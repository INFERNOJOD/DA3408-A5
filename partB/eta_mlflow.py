# Part B: MLflow registry, promotion gate.
# usage (from repo root):
#   python partB/eta_mlflow.py register
#   python partB/eta_mlflow.py gate [--min-rel-gain 0.60]
import os
import sys
import time
import json
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import eta_data  # noqa: E402

MODEL_NAME = "eta-model"
RUSH_HOURS = [7, 8, 9, 16, 17, 18, 19]
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}"))


class EtaModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        b = joblib.load(context.artifacts["bundle"])
        self.model, self.features, self.version = b["model"], b["feature_columns"], b["version"]

    def predict(self, context, model_input, params=None):
        preds = self.model.predict(model_input[self.features])
        return pd.DataFrame({"eta_minutes": np.round(preds, 1), "model_version": self.version})


def register():
    mlflow.set_experiment("eta-registry")
    fresh = eta_data.make_data(2000, seed=99)
    example = fresh[eta_data.FEATURES].head(5)
    reqs = [f"mlflow=={mlflow.__version__}", f"scikit-learn=={sklearn.__version__}",
            f"pandas=={pd.__version__}", f"numpy=={np.__version__}", f"joblib=={joblib.__version__}"]
    versions = {}
    for v in ["v1", "v2"]:
        path = ROOT / "data" / f"eta_{v}.joblib"
        b = joblib.load(path)
        out = pd.DataFrame({"eta_minutes": np.round(b["model"].predict(example), 1), "model_version": v})
        mae = float(np.mean(np.abs(b["model"].predict(fresh[eta_data.FEATURES]) - fresh.eta_minutes)))
        with mlflow.start_run(run_name=f"eta-{v}"):
            mlflow.log_params({"bundle_version": v, "estimator": type(b["model"]).__name__})
            mlflow.log_metric("fresh_mae", mae)
            info = mlflow.pyfunc.log_model(
                name="model", python_model=EtaModel(), artifacts={"bundle": str(path)},
                signature=infer_signature(example, out), input_example=example,
                pip_requirements=reqs, registered_model_name=MODEL_NAME)
        versions[v] = info.registered_model_version

    client = MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, "prod", versions["v1"])
    client.set_registered_model_alias(MODEL_NAME, "candidate", versions["v2"])

    print("version | alias     | run_id")
    for mv in client.search_model_versions(f"name='{MODEL_NAME}'"):
        alias = ",".join(client.get_model_version(MODEL_NAME, mv.version).aliases) or "-"
        print(f"{mv.version:>7} | {alias:<9} | {mv.run_id[:8]}")
    return versions


def decide(prod, cand, min_rel_gain=0.03, max_p95_ms=10.0, max_slice_regression=0.05):
    # slice rule is relative: cand slice MAE may be at most (1 + max_slice_regression) * prod slice MAE
    rel_gain = (prod["mae"] - cand["mae"]) / prod["mae"]
    criteria = {
        "mae_gain": rel_gain >= min_rel_gain,
        "latency_p95": cand["p95_ms"] <= max_p95_ms,
        "rush_slice": cand["mae_rush"] <= prod["mae_rush"] * (1 + max_slice_regression),
        "rain_slice": cand["mae_rain"] <= prod["mae_rain"] * (1 + max_slice_regression),
    }
    return {"criteria": criteria, "promote": all(criteria.values())}


def evaluate(model, df):
    pred = model.predict(df[eta_data.FEATURES])["eta_minutes"].to_numpy()
    err = np.abs(pred - df.eta_minutes.to_numpy())
    rush = df.hour.isin(RUSH_HOURS).to_numpy()
    rain = (df.rain == 1).to_numpy()
    times = []
    for i in range(200):
        row = df[eta_data.FEATURES].iloc[[i]]
        t0 = time.perf_counter()
        model.predict(row)
        times.append((time.perf_counter() - t0) * 1000)
    return {"mae": round(float(err.mean()), 3), "mae_rush": round(float(err[rush].mean()), 3),
            "mae_rain": round(float(err[rain].mean()), 3), "p95_ms": round(float(np.percentile(times, 95)), 3)}


def promotion_gate(**thresholds):
    thresholds = {"min_rel_gain": 0.03, "max_p95_ms": 10.0, "max_slice_regression": 0.05, **thresholds}
    prod = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@prod")
    cand = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@candidate")
    fresh = eta_data.make_data(2000, seed=99)
    pm, cm = evaluate(prod, fresh), evaluate(cand, fresh)
    result = decide(pm, cm, **thresholds)
    result.update({"thresholds": thresholds, "prod": pm, "candidate": cm,
                   "rel_gain": round((pm["mae"] - cm["mae"]) / pm["mae"], 4)})
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["register", "gate"])
    ap.add_argument("--min-rel-gain", type=float)
    ap.add_argument("--max-p95-ms", type=float)
    ap.add_argument("--max-slice-regression", type=float)
    a = ap.parse_args()
    if a.cmd == "register":
        register()
    else:
        th = {k: v for k, v in [("min_rel_gain", a.min_rel_gain), ("max_p95_ms", a.max_p95_ms),
                                ("max_slice_regression", a.max_slice_regression)] if v is not None}
        print(json.dumps(promotion_gate(**th), indent=2))
