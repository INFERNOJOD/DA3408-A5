# DA3408 Module 4 — Krish Yadav (DA24B043)

## Machine

4 CPUs (lscpu doesnt report the model name), 7.2 GiB RAM, Python 3.12.15, Linux 7.0.0-38-generic.
Load generator and server ran on the same VM.
Source: machine.txt

## Part A — REST service

### A3.1 Load test (v1, 1 uvicorn worker, batch 1, 4 s per level)

| concurrency | requests | rps | p50 ms | p95 ms | p99 ms | errors |
|---|---|---|---|---|---|---|
| 1 | 3496 | 874.0 | 1.1 | 1.3 | 1.6 | 0 |
| 2 | 3912 | 978.0 | 2.0 | 2.5 | 3.9 | 0 |
| 4 | 3346 | 836.5 | 4.3 | 8.0 | 11.6 | 0 |
| 8 | 3440 | 860.0 | 8.7 | 15.5 | 20.7 | 0 |
| 16 | 3424 | 856.0 | 17.4 | 31.7 | 44.4 | 0 |

Source: partA/loadtest_results.json

### A3.2 Saturation point and capacity under SLO (p95 <= 80 ms)

| concurrency | requests | rps | p50 ms | p95 ms | p99 ms | errors |
|---|---|---|---|---|---|---|
| 32 | 2953 | 738.2 | 38.2 | 74.4 | 163.3 | 0 |
| 48 | 3374 | 843.5 | 55.0 | 84.7 | 106.9 | 0 |

Source: partA/loadtest_slo_probe.json

- Saturation:- rps stops growing after c=2 (978.0) and stays flat at 836.5–860.0 while latency keeps rising,
  so it saturates at about c=2 (one worker is the bottleneck).
- Little's Law (L = λW):- at c=16, W = 16 / 856.0 = 18.7 ms mean latency, which sits between the
  measured p50 17.4 and p95 31.7 ms. a request alone takes ~1.1 ms, so the rest is queueing.
- Capacity under SLO:- capacity = highest throughput with p95 <= 80 ms. every run up to c=32 meets it
  (p95 74.4 ms at 32), c=48 fails (84.7 ms). rps doesnt grow past saturation, so capacity is ~850 rps
  sustained (978 peak at c=2), and the SLO holds up to ~32 requests in flight.

### A3.3 Cost per row, batch 1 vs batch 100 (concurrency 1)

| batch | requests | rps (requests/s) | p50 ms | p95 ms | p99 ms | errors |
|---|---|---|---|---|---|---|
| 1 | 3699 | 924.8 | 1.0 | 1.2 | 1.5 | 0 |
| 100 | 2611 | 652.8 | 1.5 | 1.7 | 2.0 | 0 |

Source: partA/loadtest_batch1.json, partA/loadtest_batch100.json

- at c=1, time per request = 1000 / rps
  - batch 1:- 1000 / 924.8 = 1.08 ms per request = 1.08 ms/row
  - batch 100:- 1000 / 652.8 = 1.53 ms per request = 0.015 ms/row
- so ~70x cheaper per row. most of the ~1 ms is fixed per request (HTTP, JSON, pydantic, DataFrame, metrics)
  and Ridge predict is one matrix multiply, so 100 rows cost almost the same as 1.

### A3.4 Change to double capacity: 2 uvicorn workers

| concurrency | requests | rps | p50 ms | p95 ms | p99 ms | errors |
|---|---|---|---|---|---|---|
| 1 | 3650 | 912.5 | 1.1 | 1.2 | 1.5 | 0 |
| 2 | 4100 | 1025.0 | 1.9 | 2.2 | 3.0 | 0 |
| 4 | 6331 | 1582.8 | 2.4 | 4.6 | 5.9 | 0 |
| 8 | 6947 | 1736.8 | 4.2 | 8.0 | 10.6 | 0 |
| 16 | 7096 | 1774.0 | 8.5 | 14.9 | 19.5 | 0 |

Source: partA/loadtest_workers2.json ("replicas": 1 in that file counts the served_by field from load_test.py, not uvicorn workers)

- one worker = one python process, so with the GIL it uses ~1 core. i restarted with `--workers 2`.
- c=16:- 856.0 -> 1774.0 rps (2.07x), c=8:- 860.0 -> 1736.8 (2.02x), p95 at c=16 31.7 -> 14.9 ms. so it did double.
- c=1 doesnt change (874.0 vs 912.5) since one request at a time only uses one worker.

### A3.5 Why p95/p99 and not the mean

The mean hides the tail, a few slow requests barely move it but those users still wait (at c=32 p50 is
38.2 ms but p99 is 163.3 ms). p95/p99 bounds what almost every user sees, and a user making many requests
will hit the tail sooner or later.

## Part B — MLflow

### B1 Registry

| version | alias | run_id |
|---|---|---|
| 1 | prod | 99e68750 |
| 2 | candidate | 5966c180 |

Source: partB/register_output.txt

- custom pyfunc (EtaModel in partB/eta_mlflow.py) returns eta_minutes (1 decimal) and model_version, logged
  with signature, 5-row input example and pinned pip_requirements. SQLite store (mlflow.db).

### B2 Promotion gate

| | default thresholds | min_rel_gain=0.60 |
|---|---|---|
| prod mae / mae_rush / mae_rain | 11.955 / 19.68 / 11.623 | 11.955 / 19.68 / 11.623 |
| candidate mae / mae_rush / mae_rain | 6.105 / 8.58 / 6.285 | 6.105 / 8.58 / 6.285 |
| prod p95_ms | 0.859 | 0.793 |
| candidate p95_ms | 0.715 | 0.881 |
| rel_gain | 0.4893 | 0.4893 |
| mae_gain / latency_p95 / rush_slice / rain_slice | true / true / true / true | false / true / true / true |
| promote | true | false |

Source: partB/gate_output.txt

- default:- passes all 4 -> promote. min_rel_gain=0.60:- fails only mae_gain (0.4893) -> no promote.
  (p95 differs between runs since each run re-times 200 predictions)
- slice rule:- i took max_slice_regression as relative like min_rel_gain, cand slice MAE <= prod x 1.05.
  an absolute 0.05 min is below the noise on a slice. test_gate.py checks +4.9 % (pass) and +5.1 % (fail).

### B3 Shadow comparison (same 200 rows from data/eta_fresh.csv, prod on 5101, candidate on 5102)

| metric | value |
|---|---|
| requests sent to each server | 200 |
| model versions answering | prod=v1, candidate=v2 |
| mean abs difference | 9.48 min |
| share differing by > 10 min | 35.0 % |
| max abs difference | 51.7 min |
| prod latency p50 / p95 | 2.2 / 2.5 ms |
| candidate latency p50 / p95 | 2.3 / 2.8 ms |

Source: partB/shadow_output.txt (lines 1-7)

Alias test:- with the 5101 server running i moved prod to version 2, sent a request, and moved it back.

    registry now says prod = version 2
    running 5101 server still answers: {"predictions": [{"eta_minutes": 38.0, "model_version": "v1"}]}
    alias moved back: prod = version 1

Source: partB/shadow_output.txt (lines 9-12)

- Decision:- promote v2, but via a canary not a straight swap. the gate shows it is better overall (MAE
  6.105 vs 11.955) and on both slices, and latency is about the same (p95 2.8 vs 2.5 ms). shadow has no labels
  so it only shows they disagree, but a lot (35.0 % by > 10 min, worst 51.7 min), so customers will notice.
- Why the alias doesnt change a running server:- `models:/eta-model@prod` is resolved to a version once at
  startup and that model sits in memory. the server never checks the registry again (alias test above).
- Two ways to pick up v2 without dropping requests:-
  1. rolling / blue-green:- start new instances (resolve alias -> v2), wait for health check, shift traffic,
     drain the old ones.
  2. hot reload:- a background thread polls the alias, loads the new model alongside and swaps the
     reference in one assignment, in-flight requests finish on the old one.

## Part C — Ray Serve sticky canary

- prod (v1) 2 replicas, candidate (v2) autoscales 1-3, everything reserves 0.25 CPU so worst case is
  0.25 x (1 + 2 + 3) = 1.5 CPUs, fits in 2.
- customer goes to candidate if sha256(customer_id) mapped to [0, 1] < weight, or if `X-Beta: 1`.

### C2 Verification (2-CPU Ray, 1000 customer ids, weight 0.2)

| check | result |
|---|---|
| (a) share on candidate at weight 0.2 | 0.203 |
| (b) 200 customers, second call | 0 changed arm |
| (c) after delete + redeploy | 0 of 200 changed arm |
| (d) weight 0.2 -> 0.5 | share 0.513, 203 were on candidate, 0 moved back to prod |
| (e) X-Beta: 1 on 50 prod customers | 50/50 went to candidate |
| replicas at end | eta_v1 2, eta_v2 1, Router 1 |

Source: partC/verify_output.txt

### C3.1 Why hash() is the wrong tool

python salts hash() for strings per process (PYTHONHASHSEED), so the same customer_id hashes differently in
each replica and after each restart and customers would switch arms. sha256 is the same everywhere (check (c)).

### C3.2 Why stickiness matters

- product:- if the ETA jumps between v1 and v2 on refresh the customer stops trusting it, and 35.0 % of
  shadow requests differed by > 10 min so it would happen a lot.
- analysis:- each customer is in exactly one arm, so outcomes (complaints, cancellations) map to one model
  and arms dont mix. raising the weight only adds customers (check (d)).

### C3.3 Autoscaling parameters

target_ongoing_requests=2 keeps the queue short since with 0.25 CPU a queue becomes latency fast, so scale
early. upscale_delay_s=5 reacts in a few seconds but ignores one spike, downscale_delay_s=30 is longer so it
doesnt flap with bursty traffic. max_replicas=3 is what fits 2 CPUs (1.5 worst case), rest left for Ray itself.

## Part D — Release gate

### D1 Smoke test runs

| server | expected | checks passed | result | exit code | p95 |
|---|---|---|---|---|---|
| v1 (port 8000) | v1 | 7/7 | GO | 0 | 1.4 ms |
| v1 (port 8000) | v2 | 5/7 | NO-GO | 1 | 1.3 ms |
| v2 (port 8001) | v2 | 7/7 | GO | 0 | 1.8 ms |

Source: partD/smoke_output.txt

- 7 checks:- healthz version, predict version + sane values, 3 sanity checks (longer distance / more traffic /
  more stops must not give a shorter ETA), invalid input -> 422, single-row p95 under --max-p95-ms.
- 2nd run (expecting v2 on the v1 server) shows a wrong version blocks the release with exit 1.

### D1 Abort criteria for a 10 % canary of v2

| # | metric (v2 arm) | abort if | window |
|---|---|---|---|
| 1 | p95 latency | > 80 ms (the SLO) | 5 min |
| 2 | 5xx error rate | > 1 % of requests | 5 min |
| 3 | mean abs difference to v1 on the same requests | > 15 min (shadow baseline was 9.48 min) | 1 h |
| 4 | MAE on rush-hour or rain deliveries once actual times arrive | worse than v1 by more than 5 % (same as the gate) | 24 h |

## Not tested

- candidate autoscaling under load:- verify_ray.py never sets extra_work_ms, and eta_v2 stayed at 1 replica
- smoke_eta.py against the Ray app or the MLflow servers (different request format, no /healthz)
- load tests against v2, the MLflow servers or the Ray app
- the abort criteria in D1 are not implemented as alerts anywhere

## Citations / AI use

- request-id middleware and validation handler idea adapted from the Module 4 lecture notebook (partA/eta_app.py).
- Ray Serve deployment / handle pattern adapted from the Module 4 lecture notebook (partC/eta_ray.py).
- used Claude Code for a final check against the spec, a couple of small fixes (model path, registry listing
  script) .
