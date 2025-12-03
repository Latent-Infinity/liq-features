from liq.features.stationarity import StationarityTransformer, _fracdiff_weights


def test_fracdiff_weights_monotonic_decay() -> None:
    weights = _fracdiff_weights(0.4, max_lags=10, tol=1e-6)
    assert weights[0] == 1.0
    assert all(abs(weights[i]) >= abs(weights[i + 1]) for i in range(len(weights) - 1))


def test_stationarity_transformer_fit_transform_requires_fit() -> None:
    st = StationarityTransformer(d=0.4)
    data = [1, 2, 3, 4, 5]
    out = st.fit_transform(data)
    assert len(out) == len(data)
    # ensure transform only uses fitted flag
    st2 = StationarityTransformer()
    try:
        st2.transform(data)
        assert False, "expected runtime error when transform before fit"
    except RuntimeError:
        pass
