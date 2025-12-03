from liq.features.entropy import sample_entropy


def test_sample_entropy_basic() -> None:
    series = [1, 1, 1, 1, 1, 1, 1]
    ent = sample_entropy(series, m=2, r=0.2)
    assert ent == 0 or ent >= 0
