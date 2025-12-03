from liq.features.scaling import ModelAwareScaler


def test_scaler_nn_standardizes() -> None:
    scaler = ModelAwareScaler(model_type="nn")
    data = [1.0, 2.0, 3.0]
    out = scaler.fit_transform(data)
    assert abs(sum(out) / len(out)) < 1e-6  # centered


def test_scaler_tree_noop() -> None:
    scaler = ModelAwareScaler(model_type="tree")
    data = [5, 6, 7]
    out = scaler.fit_transform(data)
    assert out == data


def test_scaler_diffusion_range() -> None:
    scaler = ModelAwareScaler(model_type="diffusion")
    data = [0.0, 10.0]
    out = scaler.fit_transform(data)
    assert min(out) >= -1 - 1e-9
    assert max(out) <= 1 + 1e-9
