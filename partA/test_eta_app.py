import os
import re
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("ETA_MODEL", str(ROOT / "data" / "eta_v1.joblib"))

from fastapi.testclient import TestClient  # noqa: E402
import eta_app  # noqa: E402

client = TestClient(eta_app.app)
URL = "/v1/models/eta:predict"
ROW = {"distance_km": 12.0, "n_stops": 4, "hour": 8, "weekday": 2,
       "traffic_index": 0.4, "rain": 0, "weight_kg": 6.5}


def row(**kw):
    return {**ROW, **kw}


def post(instances, **kw):
    return client.post(URL, json={"instances": instances}, **kw)


def counter(text, path, status):
    m = re.search(rf'eta_requests_total{{path="{re.escape(path)}",status="{status}"}} (\d+)', text)
    return int(m.group(1)) if m else 0


def test_valid_request():
    r = post([ROW])
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"] == "v1"
    assert len(body["predictions"]) == 1
    p = body["predictions"][0]
    assert isinstance(p, float) and p == round(p, 1)


def test_batch_order_preserved():
    rows = [row(distance_km=d) for d in (2, 40, 120, 8, 75)]
    singles = [post([x]).json()["predictions"][0] for x in rows]
    assert post(rows).json()["predictions"] == singles
    assert post(rows[::-1]).json()["predictions"] == singles[::-1]


@pytest.mark.parametrize("field,value", [
    ("distance_km", 0.05), ("distance_km", 501), ("n_stops", 0), ("hour", 24),
    ("weekday", 7), ("traffic_index", 1.2), ("rain", 2), ("weight_kg", 0),
])
def test_out_of_range(field, value):
    r = post([row(**{field: value})])
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "validation_failed"
    assert any(field in d["field"] for d in body["details"])


def test_missing_field():
    bad = dict(ROW)
    del bad["hour"]
    r = post([bad])
    assert r.status_code == 422
    assert any(d["field"].endswith("hour") for d in r.json()["details"])


def test_wrong_type():
    r = post([row(n_stops="many")])
    assert r.status_code == 422
    assert any("n_stops" in d["field"] for d in r.json()["details"])


def test_malformed_json():
    r = client.post(URL, content=b'{"instances": [', headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    assert r.json()["error"] == "validation_failed"


def test_infeasible_rejected():
    r = post([row(distance_km=151, n_stops=21)])
    assert r.status_code == 422
    assert "infeasible" in r.text


def test_feasible_at_boundary():
    # rule is strict > on both, so exactly 150 km / 20 stops is allowed
    assert post([row(distance_km=150, n_stops=20)]).status_code == 200
    assert post([row(distance_km=300, n_stops=20)]).status_code == 200


def test_batch_size_limits():
    assert post([]).status_code == 422
    assert post([ROW] * 501).status_code == 422
    r = post([ROW] * 500)
    assert r.status_code == 200 and len(r.json()["predictions"]) == 500


def test_request_id_echoed():
    r = post([ROW], headers={"X-Request-ID": "abc-123"})
    assert r.headers["X-Request-ID"] == "abc-123"
    assert r.json()["request_id"] == "abc-123"


def test_request_id_generated():
    r = post([ROW])
    rid = r.headers["X-Request-ID"]
    uuid.UUID(rid)
    assert r.json()["request_id"] == rid


def test_request_id_on_error():
    r = post([row(hour=99)], headers={"X-Request-ID": "err-1"})
    assert r.status_code == 422
    assert r.headers["X-Request-ID"] == "err-1" and r.json()["request_id"] == "err-1"

    r = post([row(hour=99)])
    assert r.json()["request_id"] == r.headers["X-Request-ID"]


def test_metrics():
    before = client.get("/metrics").text
    post([ROW])
    post([row(rain=5)])
    text = client.get("/metrics").text
    assert counter(text, URL, 200) == counter(before, URL, 200) + 1
    assert counter(text, URL, 422) == counter(before, URL, 422) + 1
    assert "# TYPE eta_request_latency_seconds histogram" in text
    assert len(re.findall(r'eta_request_latency_seconds_bucket\{le="[^"]+"\}', text)) >= 5
    inf = int(re.search(r'_bucket\{le="\+Inf"\} (\d+)', text).group(1))
    count = int(re.search(r"eta_request_latency_seconds_count (\d+)", text).group(1))
    assert inf == count
    assert "eta_request_latency_seconds_sum" in text
    assert 'path="/metrics"' not in text


def test_generic_500(monkeypatch):
    class Broken:
        def predict(self, X):
            raise RuntimeError("secret internal detail")

    monkeypatch.setattr(eta_app, "model", Broken())
    r = post([ROW], headers={"X-Request-ID": "boom"})
    assert r.status_code == 500
    assert "secret" not in r.text and "Traceback" not in r.text
    assert r.json()["request_id"] == "boom" and r.headers["X-Request-ID"] == "boom"


def test_healthz():
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_version": "v1"}
