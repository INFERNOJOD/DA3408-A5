# DA3408 Module 4 — SwiftParcel ETA (Krish Yadav, DA24B043)

## Setup

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python eta_data.py            # writes data/eta_train.csv, data/eta_fresh.csv, data/eta_v1.joblib, data/eta_v2.joblib
    python -c "import pandas as pd, eta_data; pd.read_csv('data/eta_fresh.csv')[eta_data.FEATURES].head(200).to_csv('data/eta_sample.csv', index=False)"

Run everything below from the repo root.

## How to reproduce

### Tests

    pytest partA/test_eta_app.py partB/test_gate.py

### Part A

    uvicorn --app-dir partA eta_app:app --port 8000            # serves v1 (ETA_MODEL default)

In another terminal:

    python load_test.py --url http://localhost:8000/v1/models/eta:predict --data data/eta_sample.csv \
           --concurrency 1 2 4 8 16 --duration 4 --out partA/loadtest_results.json
    python load_test.py --url http://localhost:8000/v1/models/eta:predict --data data/eta_sample.csv \
           --concurrency 32 48 --duration 4 --out partA/loadtest_slo_probe.json
    python load_test.py --url http://localhost:8000/v1/models/eta:predict --data data/eta_sample.csv \
           --concurrency 1 --duration 4 --batch 1 --out partA/loadtest_batch1.json
    python load_test.py --url http://localhost:8000/v1/models/eta:predict --data data/eta_sample.csv \
           --concurrency 1 --duration 4 --batch 100 --out partA/loadtest_batch100.json

For the A3.4 change, restart with 2 workers and rerun the first load test:

    uvicorn --app-dir partA eta_app:app --port 8000 --workers 2
    python load_test.py --url http://localhost:8000/v1/models/eta:predict --data data/eta_sample.csv \
           --concurrency 1 2 4 8 16 --duration 4 --out partA/loadtest_workers2.json

### Part B

    python partB/eta_mlflow.py register          # only once on a fresh checkout, rerunning adds v3/v4 and moves the aliases
    python partB/list_registry.py > partB/register_output.txt
    python partB/eta_mlflow.py gate
    python partB/eta_mlflow.py gate --min-rel-gain 0.60

Shadow comparison (two extra terminals for the servers):

    export MLFLOW_TRACKING_URI=sqlite:///$PWD/mlflow.db
    mlflow models serve -m "models:/eta-model@prod" -p 5101 --env-manager local
    mlflow models serve -m "models:/eta-model@candidate" -p 5102 --env-manager local
    python partB/shadow_compare.py > partB/shadow_output.txt

### Part C

    python partC/verify_ray.py | tee partC/verify_output.txt    # starts a 2-CPU local Ray, shuts it down at the end

### Part D

    uvicorn --app-dir partA eta_app:app --port 8000
    python partD/smoke_eta.py --url http://127.0.0.1:8000 --expect-version v1
    ETA_MODEL=data/eta_v2.joblib uvicorn --app-dir partA eta_app:app --port 8001
    python partD/smoke_eta.py --url http://127.0.0.1:8001 --expect-version v2

Exit code is 0 if all checks pass, 1 otherwise. Output is saved in partD/smoke_output.txt.

## For TAs: what to check

| Rubric item | File | Screenshot (evidence/) |
|---|---|---|
| Setup | requirements.txt, eta_data.py | 01_env_setup.png |
| A1 service | partA/eta_app.py | 03_partA_requests_and_422.png, 04_partA_live_checks_metrics.png |
| A2 tests | partA/test_eta_app.py | 02_partA_pytest_passed.png |
| A3 load test | partA/loadtest_*.json, REPORT.md | 05_partA_loadtest_tables.png |
| B1 registration | partB/eta_mlflow.py (register), partB/list_registry.py, partB/register_output.txt | 06_partB_registry_aliases.png, 12_partB_mlflow_ui_registry_aliases.png, 16_partB_registry_listing.png |
| B2 gate | partB/eta_mlflow.py (decide, promotion_gate), partB/test_gate.py, partB/gate_output.txt | 07_partB_gate_tests_passed.png, 08_partB_gate_default.png, 09_partB_gate_min_rel_gain_060.png |
| B3 shadow | partB/shadow_compare.py, partB/shadow_output.txt, REPORT.md | 10_partB_shadow_output.png, 11_partB_alias_does_not_change_running_server.png |
| C1 canary app | partC/eta_ray.py | 13_partC_verify_ray_all_passed.png |
| C2 verification | partC/verify_ray.py, partC/verify_output.txt | 13_partC_verify_ray_all_passed.png |
| C3 short answers | REPORT.md | – |
| D1 release gate | partD/smoke_eta.py, partD/smoke_output.txt, REPORT.md (abort criteria) | 14_partD_smoke_go_and_nogo.png, 15_partD_smoke_v2_go.png |

## Evidence

`evidence/` has 16 screenshots taken on my VM while running the commands above (date/time bar visible in each).
They are numbered in rubric order and mapped in the table above.
