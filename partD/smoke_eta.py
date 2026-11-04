# Part D: release smoke test. Exit 0 if every check passes, 1 otherwise.
# usage: python partD/smoke_eta.py --url http://127.0.0.1:8000 --expect-version v1 [--max-p95-ms 80]
import sys
import time
import argparse

import numpy as np
import requests

BASE = {"distance_km": 20.0, "n_stops": 5, "hour": 11, "weekday": 2,
        "traffic_index": 0.3, "rain": 0, "weight_kg": 8.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="service base url, e.g. http://127.0.0.1:8000")
    ap.add_argument("--expect-version", required=True)
    ap.add_argument("--max-p95-ms", type=float, default=80.0)
    a = ap.parse_args()

    base = a.url.rstrip("/").removesuffix("/v1/models/eta:predict")
    predict_url = base + "/v1/models/eta:predict"
    s = requests.Session()
    results = []

    def predict(rows):
        r = s.post(predict_url, json={"instances": rows}, timeout=10)
        r.raise_for_status()
        return r.json()

    def check(name, fn):
        try:
            ok, info = fn()
        except Exception as e:
            ok, info = False, f"error: {e}"
        results.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {info}")

    def health():
        j = s.get(base + "/healthz", timeout=5).json()
        return j.get("status") == "ok" and j.get("model_version") == a.expect_version, j

    def version_and_shape():
        j = predict([BASE, BASE])
        ok = (j["model_version"] == a.expect_version and len(j["predictions"]) == 2
              and all(np.isfinite(p) and 0 < p < 2000 for p in j["predictions"]))
        return ok, f"version={j['model_version']} predictions={j['predictions']}"

    def longer_distance_not_faster():
        p = predict([{**BASE, "distance_km": d} for d in (5, 50, 150)])["predictions"]
        return p[0] <= p[1] <= p[2], f"5/50/150 km -> {p}"

    def heavier_traffic_not_faster():
        p = predict([{**BASE, "traffic_index": t} for t in (0.1, 0.9)])["predictions"]
        return p[0] <= p[1], f"traffic 0.1/0.9 -> {p}"

    def more_stops_not_faster():
        p = predict([{**BASE, "n_stops": n} for n in (2, 20)])["predictions"]
        return p[0] <= p[1], f"stops 2/20 -> {p}"

    def invalid_is_422():
        r = s.post(predict_url, json={"instances": [{**BASE, "hour": 30}]}, timeout=10)
        return r.status_code == 422, f"status={r.status_code}"

    def latency():
        lat = []
        for _ in range(50):
            t0 = time.perf_counter()
            predict([BASE])
            lat.append((time.perf_counter() - t0) * 1000)
        p95 = float(np.percentile(lat, 95))
        return p95 <= a.max_p95_ms, f"p95={p95:.1f} ms (limit {a.max_p95_ms})"

    check("healthz reports expected version", health)
    check("predict returns expected version, sane values", version_and_shape)
    check("sanity: longer distance never gives shorter ETA", longer_distance_not_faster)
    check("sanity: heavier traffic never gives shorter ETA", heavier_traffic_not_faster)
    check("sanity: more stops never gives shorter ETA", more_stops_not_faster)
    check("invalid input returns 422", invalid_is_422)
    check("single-row p95 latency within limit", latency)

    passed = sum(results)
    print(f"\n{passed}/{len(results)} checks passed -> {'GO' if all(results) else 'NO-GO'}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
