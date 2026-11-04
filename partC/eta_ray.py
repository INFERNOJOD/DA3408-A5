# Part C: Ray Serve app with a sticky canary (v1 = prod, v2 = candidate).
# Deployment / handle pattern adapted from the Module 4 lecture notebook.
import time
import hashlib
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel
from ray import serve

DATA = Path(__file__).resolve().parents[1] / "data"
api = FastAPI()


class CanaryRequest(BaseModel):
    customer_id: str
    instances: list[dict]


def bucket(customer_id: str) -> float:
    # stable across processes and restarts, unlike hash()
    h = hashlib.sha256(customer_id.encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


@serve.deployment(ray_actor_options={"num_cpus": 0.25})
class EtaModel:
    def __init__(self, path: str, extra_work_ms: float = 0.0):
        b = joblib.load(path)
        self.model, self.features, self.version = b["model"], b["feature_columns"], b["version"]
        self.extra_s = extra_work_ms / 1000

    def predict(self, instances: list[dict]) -> dict:
        if self.extra_s:
            time.sleep(self.extra_s)  # simulates a heavier model, used to trigger autoscaling
        preds = self.model.predict(pd.DataFrame(instances)[self.features])
        return {"predictions": [round(float(p), 1) for p in preds], "model_version": self.version}


@serve.deployment(ray_actor_options={"num_cpus": 0.25})
@serve.ingress(api)
class Router:
    def __init__(self, prod, candidate, weight: float):
        self.prod, self.candidate, self.weight = prod, candidate, weight

    @api.post("/v1/models/eta:predict")
    async def predict(self, req: CanaryRequest, request: Request):
        to_candidate = request.headers.get("x-beta") == "1" or bucket(req.customer_id) < self.weight
        handle = self.candidate if to_candidate else self.prod
        out = await handle.predict.remote(req.instances)
        return {**out, "arm": "v2" if to_candidate else "v1"}


def build(weight=0.1, extra_work_ms=0.0):
    # CPU budget: router 0.25 + prod 2 x 0.25 + candidate up to 3 x 0.25 = 1.5 CPUs  (<= 2)
    prod = EtaModel.options(name="eta_v1", num_replicas=2).bind(str(DATA / "eta_v1.joblib"), extra_work_ms)
    candidate = EtaModel.options(
        name="eta_v2",
        autoscaling_config={
            "min_replicas": 1,
            "max_replicas": 3,
            "target_ongoing_requests": 2,
            "upscale_delay_s": 5,
            "downscale_delay_s": 30,
        },
    ).bind(str(DATA / "eta_v2.joblib"), extra_work_ms)
    return Router.bind(prod, candidate, weight)
