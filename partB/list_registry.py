# B1: read-only listing of the eta-model registry, does not register anything
# usage (from repo root): python partB/list_registry.py > partB/register_output.txt
import os
from pathlib import Path

import mlflow
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}"))
client = MlflowClient()

print("version | alias     | run_id")
for mv in sorted(client.search_model_versions("name='eta-model'"), key=lambda m: int(m.version)):
    alias = ",".join(client.get_model_version("eta-model", mv.version).aliases) or "-"
    print(f"{mv.version:>7} | {alias:<9} | {mv.run_id[:8]}")
