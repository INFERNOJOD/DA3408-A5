import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eta_mlflow import decide  # noqa: E402

PROD = {"mae": 10.0, "mae_rush": 12.0, "mae_rain": 11.0, "p95_ms": 3.0}


def cand(**kw):
    return {"mae": 9.0, "mae_rush": 12.0, "mae_rain": 11.0, "p95_ms": 5.0, **kw}


def test_clear_winner_promoted():
    r = decide(PROD, cand(mae=6.0, mae_rush=8.0, mae_rain=7.0))
    assert r["promote"] is True
    assert all(r["criteria"].values())


def test_gain_just_inside_threshold():
    assert decide(PROD, cand(mae=9.69))["criteria"]["mae_gain"] is True    # 3.1 % gain


def test_gain_just_outside_threshold():
    r = decide(PROD, cand(mae=9.71))                                      # 2.9 % gain
    assert r["criteria"]["mae_gain"] is False and r["promote"] is False


def test_worse_model_rejected():
    assert decide(PROD, cand(mae=10.5))["promote"] is False


def test_latency_boundary():
    assert decide(PROD, cand(p95_ms=10.0))["criteria"]["latency_p95"] is True
    r = decide(PROD, cand(p95_ms=10.01))
    assert r["criteria"]["latency_p95"] is False and r["promote"] is False


def test_rush_slice_regression():
    assert decide(PROD, cand(mae_rush=12.59))["criteria"]["rush_slice"] is True    # +4.9 %
    r = decide(PROD, cand(mae_rush=12.61))                                         # +5.1 %
    assert r["criteria"]["rush_slice"] is False and r["promote"] is False


def test_rain_slice_regression():
    assert decide(PROD, cand(mae_rain=11.54))["criteria"]["rain_slice"] is True    # +4.9 %
    r = decide(PROD, cand(mae_rain=11.56))                                         # +5.1 %
    assert r["criteria"]["rain_slice"] is False and r["promote"] is False


def test_overall_gain_does_not_hide_slice_regression():
    r = decide(PROD, cand(mae=5.0, mae_rain=15.0))
    assert r["criteria"]["mae_gain"] is True and r["promote"] is False


def test_custom_thresholds():
    c = cand(mae=6.0, mae_rush=8.0, mae_rain=7.0, p95_ms=15.0)
    assert decide(PROD, c)["promote"] is False
    assert decide(PROD, c, max_p95_ms=20.0)["promote"] is True
    assert decide(PROD, c, max_p95_ms=20.0, min_rel_gain=0.60)["promote"] is False
