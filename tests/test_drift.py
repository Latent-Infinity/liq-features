from liq.features.drift import ks_drift


def test_ks_drift_detects_shift() -> None:
    ref = [1, 1, 1, 1]
    live = [10, 10, 10, 10]
    res = ks_drift(live, ref, feature="f", threshold=0.5)
    assert res.drifted


def test_ks_drift_no_drift_small_shift() -> None:
    ref = [1, 1, 1]
    live = [1.05, 0.95, 1.0]
    res = ks_drift(live, ref, feature="f", threshold=1.0)
    assert not res.drifted
