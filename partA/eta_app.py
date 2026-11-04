# ETA service. Request-id middleware + validation handler idea adapted from the Module 4 lecture notebook.
import os
import time
import uuid
import logging
import threading
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, model_validator

log = logging.getLogger("eta_app")

MODEL_PATH = os.environ.get("ETA_MODEL", "data/eta_v1.joblib")
if not os.path.exists(MODEL_PATH):  # not started from repo root, try the same path relative to it
    MODEL_PATH = Path(__file__).resolve().parents[1] / MODEL_PATH
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
FEATURES = bundle["feature_columns"]
MODEL_VERSION = bundle["version"]


class Delivery(BaseModel):
    distance_km: float = Field(ge=0.1, le=500)
    n_stops: int = Field(ge=1, le=60)
    hour: int = Field(ge=0, le=23)
    weekday: int = Field(ge=0, le=6)
    traffic_index: float = Field(ge=0, le=1)
    rain: int = Field(ge=0, le=1)
    weight_kg: float = Field(gt=0, le=1000)

    @model_validator(mode="after")
    def check_feasible(self):
        if self.distance_km > 150 and self.n_stops > 20:
            raise ValueError("infeasible delivery: distance_km > 150 with n_stops > 20")
        return self


class PredictRequest(BaseModel):
    instances: list[Delivery] = Field(min_length=1, max_length=500)


BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.08, 0.1, 0.25, 0.5, 1.0, 2.5]
lock = threading.Lock()
req_counts = {}
bucket_counts = [0] * (len(BUCKETS) + 1)  # last slot = +Inf
lat_sum, lat_count = 0.0, 0


def observe(path, status, secs):
    global lat_sum, lat_count
    with lock:
        req_counts[(path, status)] = req_counts.get((path, status), 0) + 1
        lat_sum += secs
        lat_count += 1
        i = next((i for i, b in enumerate(BUCKETS) if secs <= b), len(BUCKETS))
        bucket_counts[i] += 1


def render_metrics():
    out = ["# HELP eta_requests_total Total requests by path and status.",
           "# TYPE eta_requests_total counter"]
    with lock:
        for (path, status), n in sorted(req_counts.items()):
            out.append(f'eta_requests_total{{path="{path}",status="{status}"}} {n}')
        out += ["# HELP eta_request_latency_seconds Request latency.",
                "# TYPE eta_request_latency_seconds histogram"]
        cum = 0
        for le, n in zip(BUCKETS + ["+Inf"], bucket_counts):
            cum += n  # buckets are cumulative
            out.append(f'eta_request_latency_seconds_bucket{{le="{le}"}} {cum}')
        out.append(f"eta_request_latency_seconds_sum {lat_sum:.6f}")
        out.append(f"eta_request_latency_seconds_count {lat_count}")
    return "\n".join(out) + "\n"


def server_error(rid):
    return JSONResponse(status_code=500, content={"error": "internal_error", "request_id": rid})


app = FastAPI(title="SwiftParcel ETA")


@app.middleware("http")
async def request_id_and_metrics(request: Request, call_next):
    rid = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = rid
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("unhandled error rid=%s", rid)
        response = server_error(rid)
    response.headers["X-Request-ID"] = rid
    if request.url.path != "/metrics":
        observe(request.url.path, response.status_code, time.perf_counter() - t0)
    return response


@app.exception_handler(RequestValidationError)
async def on_validation_error(request: Request, exc: RequestValidationError):
    details = [{"field": ".".join(str(x) for x in e["loc"] if x != "body") or "body", "msg": e["msg"]}
               for e in exc.errors()]
    return JSONResponse(status_code=422, content={
        "error": "validation_failed", "details": details, "request_id": request.state.request_id})


@app.post("/v1/models/eta:predict")
def predict(req: PredictRequest, request: Request):
    rid = request.state.request_id
    try:
        X = pd.DataFrame([d.model_dump() for d in req.instances])[FEATURES]
        preds = model.predict(X)
    except Exception:
        log.exception("inference failed rid=%s", rid)  # details stay in the log, not the response
        return server_error(rid)
    return {"predictions": [round(float(p), 1) for p in preds],
            "model_version": MODEL_VERSION, "request_id": rid}


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.get("/metrics")
def metrics():
    return PlainTextResponse(render_metrics(), media_type="text/plain; version=0.0.4")
