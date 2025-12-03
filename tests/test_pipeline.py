from liq.features.pipeline import FeaturePipeline


def test_pipeline_fit_transform_and_serialize() -> None:
    data = [1.0, 2.0, 3.0]
    pipe = FeaturePipeline(model_type="nn", d=0.3)
    out = pipe.fit_transform(data)
    assert len(out) == len(data)
    serialized = pipe.to_dict()
    restored = FeaturePipeline.from_dict(serialized)
    transformed = restored.transform(data)
    assert len(transformed) == len(data)


def test_pipeline_requires_fit_before_transform() -> None:
    pipe = FeaturePipeline(model_type="nn")
    try:
        pipe.transform([1, 2, 3])
        assert False, "expected runtime error"
    except RuntimeError:
        pass
