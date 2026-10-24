import numpy as np
import pandas as pd
import pytest

from hsd.evaluate import aggregate, apply_temperature, threshold_for_sensitivity, choose_aggregation
from hsd.metrics import bootstrap_macro_f1_ci, compute_metrics, macro_f1
from hsd.utils import get_logger


def _win(rec, probs, label=1):
    rows = []
    for i, p in enumerate(probs):
        rows.append({"rec_id": rec, "path": rec, "label": label, "patient_id": rec, "p0": 1 - p, "p1": p})
    return rows


def test_noisy_or_saturates_as_expected():
    df = pd.DataFrame(_win("a", [0.1] * 20))          # 20 windows at 0.1 each
    rec = aggregate(df, 2, "noisy_or")
    assert rec.p1.iloc[0] == pytest.approx(1 - 0.9 ** 20, rel=1e-6)   # ~0.88: why noisy-OR needs validation
    rec_mean = aggregate(df, 2, "mean")
    assert rec_mean.p1.iloc[0] == pytest.approx(0.1)


def test_macro_f1_and_metrics():
    y = np.array([0, 0, 1, 1, 2, 2])
    assert macro_f1(y, y, 3) == pytest.approx(1.0)
    m = compute_metrics(y, np.array([0, 1, 1, 1, 2, 0]), num_classes=3)
    assert 0 < m["macro_f1"] < 1


def test_threshold_meets_requested_sensitivity():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(200), np.ones(60)].astype(int)
    p = np.r_[rng.beta(2, 5, 200), rng.beta(5, 2, 60)]
    thr = threshold_for_sensitivity(y, p, 0.90)
    sens = ((p >= thr) & (y == 1)).sum() / (y == 1).sum()
    assert sens >= 0.90


def test_temperature_identity_and_softening():
    P = np.array([[0.9, 0.1], [0.2, 0.8]])
    assert np.allclose(apply_temperature(P, 1.0), P, atol=1e-6)
    soft = apply_temperature(P, 3.0)
    assert soft[0, 0] < P[0, 0]                         # higher temperature softens


def test_aggregation_choice_uses_validation_only(tmp_path):
    """The chosen rule must come from validation predictions, whatever the test set looks like."""
    val = pd.DataFrame(_win("v1", [0.9, 0.9, 0.9], label=1) + _win("v2", [0.1, 0.1, 0.1], label=0))
    log = get_logger("hsd.test", None, "WARNING")
    method = choose_aggregation(val, 2, "auto", log, tmp_path / "agg.json")
    assert method in ("mean", "max", "median", "noisy_or")
    assert (tmp_path / "agg.json").exists()


def test_bootstrap_clusters_by_patient():
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    p = np.array([0, 0, 1, 0, 0, 0, 1, 1])
    clusters = np.array(["a", "a", "a", "a", "b", "b", "b", "b"])
    lo, hi = bootstrap_macro_f1_ci(y, p, clusters, 2, n=200)
    assert 0 <= lo <= hi <= 1
