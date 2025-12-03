import polars as pl

from liq.features.labels import TripleBarrierConfig, triple_barrier_labels


def test_triple_barrier_labels_hits_tp_and_sl() -> None:
    df = pl.DataFrame({"close": [100, 102, 90, 105]})
    cfg = TripleBarrierConfig(take_profit=0.02, stop_loss=0.05, max_holding=3)
    labels = triple_barrier_labels(df, cfg)
    assert labels[0] == 1  # tp hit at 102
    assert labels[1] == -1  # sl hit at 90 for entry at 102
    assert labels[2] in (-1, 0, 1)
