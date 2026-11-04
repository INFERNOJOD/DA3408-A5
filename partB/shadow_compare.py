# B3: send the same 200 single-row requests to prod (5101) and candidate (5102)
# usage: python partB/shadow_compare.py > partB/shadow_output.txt
import time
import numpy as np
import pandas as pd
import requests

FEATURES = ["distance_km", "n_stops", "hour", "weekday", "traffic_index", "rain", "weight_kg"]
SERVERS = {"prod": "http://127.0.0.1:5101/invocations", "candidate": "http://127.0.0.1:5102/invocations"}

rows = pd.read_csv("data/eta_fresh.csv")[FEATURES].head(200).to_dict(orient="records")
preds = {k: [] for k in SERVERS}
lat = {k: [] for k in SERVERS}
versions = {k: set() for k in SERVERS}
sess = requests.Session()

for row in rows:
    for name, url in SERVERS.items():
        t0 = time.perf_counter()
        r = sess.post(url, json={"dataframe_records": [row]}, timeout=10)
        lat[name].append((time.perf_counter() - t0) * 1000)
        r.raise_for_status()
        out = r.json()["predictions"][0]
        preds[name].append(out["eta_minutes"])
        versions[name].add(out["model_version"])

diff = np.abs(np.array(preds["prod"]) - np.array(preds["candidate"]))
print(f"requests sent to each server : {len(rows)}")
print(f"model versions answering     : prod={sorted(versions['prod'])} candidate={sorted(versions['candidate'])}")
print(f"mean |prod - candidate|      : {diff.mean():.2f} min")
print(f"share differing by > 10 min  : {(diff > 10).mean() * 100:.1f} %")
print(f"max |prod - candidate|       : {diff.max():.1f} min")
for name in SERVERS:
    print(f"{name:<9} latency p50 / p95  : {np.percentile(lat[name], 50):.1f} / {np.percentile(lat[name], 95):.1f} ms")
