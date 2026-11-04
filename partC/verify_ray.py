# C2: deploy on a 2-CPU Ray and check share, stickiness, redeploy, weight raise, header override.
# usage (from repo root):  python partC/verify_ray.py | tee partC/verify_output.txt
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import ray
import requests
from ray import serve

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eta_ray import build  # noqa: E402

URL = "http://127.0.0.1:8000/v1/models/eta:predict"
ROW = {"distance_km": 12.0, "n_stops": 4, "hour": 8, "weekday": 2,
       "traffic_index": 0.4, "rain": 0, "weight_kg": 6.5}
IDS = [f"cust-{i:04d}" for i in range(1000)]


def deploy(weight):
    serve.run(build(weight=weight), name="eta", route_prefix="/")


def ask(cid, beta=False):
    headers = {"X-Beta": "1"} if beta else {}
    r = requests.post(URL, json={"customer_id": cid, "instances": [ROW]}, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()["arm"]


def arms(ids, beta=False):
    with ThreadPoolExecutor(8) as ex:
        return dict(zip(ids, ex.map(lambda c: ask(c, beta), ids)))


def check(label, ok):
    print(f"[{'PASS' if ok else 'FAIL'}] {label}")
    assert ok, label


ray.init(num_cpus=2, runtime_env={"env_vars": {"PYTHONPATH": str(HERE)}}, log_to_driver=False)
print("ray cluster CPUs:", ray.cluster_resources().get("CPU"))

deploy(0.2)
first = arms(IDS)
share = sum(a == "v2" for a in first.values()) / len(IDS)
print(f"\n(a) weight=0.2 -> {share:.3f} of 1000 customers on candidate")
check("(a) candidate share within 0.20 +/- 0.04", abs(share - 0.2) <= 0.04)

sample = IDS[:200]
second = arms(sample)
changed = [c for c in sample if second[c] != first[c]]
print(f"\n(b) second call for 200 customers: {len(changed)} changed arm")
check("(b) same arm on second call", not changed)

serve.delete("eta")
time.sleep(2)
deploy(0.2)
again = arms(sample)
changed = [c for c in sample if again[c] != first[c]]
print(f"\n(c) after delete + redeploy of identical app: {len(changed)} of 200 changed arm")
check("(c) same arm after redeploy", not changed)

deploy(0.5)
raised = arms(IDS)
was_cand = [c for c in IDS if first[c] == "v2"]
moved_back = [c for c in was_cand if raised[c] == "v1"]
new_share = sum(a == "v2" for a in raised.values()) / len(IDS)
print(f"\n(d) weight 0.2 -> 0.5: share now {new_share:.3f}; "
      f"{len(was_cand)} were on candidate, {len(moved_back)} moved back to prod")
check("(d) nobody moves candidate -> prod when weight rises", not moved_back)

prod_ids = [c for c in IDS if raised[c] == "v1"][:50]
forced = arms(prod_ids, beta=True)
print(f"\n(e) X-Beta: 1 on 50 customers normally routed to prod -> "
      f"{sum(a == 'v2' for a in forced.values())}/50 went to candidate")
check("(e) header override sends all 50 to candidate", all(a == "v2" for a in forced.values()))

print("\ndeployments:")
for name, d in serve.status().applications["eta"].deployments.items():
    print(f"  {name:<8} status={d.status.value:<8} replicas={dict(d.replica_states)}")
print("\nall checks passed")

serve.shutdown()
ray.shutdown()
