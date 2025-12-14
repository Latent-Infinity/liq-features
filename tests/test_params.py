"""Tests for liq.features.params module."""

from liq.features.params import format_params_key, hash_params, normalize_params


class TestNormalizeParams:
    """Tests for normalize_params function."""

    def test_sorts_keys(self) -> None:
        """Test keys are sorted alphabetically."""
        params = {"z": 1, "a": 2, "m": 3}

        result = normalize_params(params)

        assert list(result.keys()) == ["a", "m", "z"]

    def test_recursive_normalization(self) -> None:
        """Test nested dicts are also normalized."""
        params = {"outer": {"z": 1, "a": 2}}

        result = normalize_params(params)

        assert list(result["outer"].keys()) == ["a", "z"]

    def test_preserves_values(self) -> None:
        """Test values are preserved."""
        params = {"period": 14, "signal": 9}

        result = normalize_params(params)

        assert result["period"] == 14
        assert result["signal"] == 9

    def test_handles_non_dict(self) -> None:
        """Test non-dict input returns as-is."""
        assert normalize_params(42) == 42
        assert normalize_params("test") == "test"
        assert normalize_params([1, 2, 3]) == [1, 2, 3]

    def test_empty_dict(self) -> None:
        """Test empty dict returns empty dict."""
        assert normalize_params({}) == {}


class TestHashParams:
    """Tests for hash_params function."""

    def test_consistent_hash(self) -> None:
        """Test same params produce same hash."""
        params = {"period": 14, "signal": 9}

        hash1 = hash_params(params)
        hash2 = hash_params(params)

        assert hash1 == hash2

    def test_order_independent(self) -> None:
        """Test different key order produces same hash."""
        params1 = {"period": 14, "signal": 9}
        params2 = {"signal": 9, "period": 14}

        assert hash_params(params1) == hash_params(params2)

    def test_hash_length(self) -> None:
        """Test hash is 16 characters."""
        params = {"period": 14}

        result = hash_params(params)

        assert len(result) == 16

    def test_different_params_different_hash(self) -> None:
        """Test different params produce different hash."""
        params1 = {"period": 14}
        params2 = {"period": 20}

        assert hash_params(params1) != hash_params(params2)

    def test_empty_params(self) -> None:
        """Test empty params produces valid hash."""
        result = hash_params({})

        assert len(result) == 16
        assert result.isalnum()


class TestFormatParamsKey:
    """Tests for format_params_key function."""

    def test_basic_formatting(self) -> None:
        """Test basic key=value formatting."""
        params = {"period": 14, "signal": 9}

        result = format_params_key(params)

        assert "period=14" in result
        assert "signal=9" in result

    def test_sorted_output(self) -> None:
        """Test output is sorted by key."""
        params = {"z": 1, "a": 2}

        result = format_params_key(params)

        assert result == "a=2,z=1"

    def test_empty_params(self) -> None:
        """Test empty params returns 'default'."""
        assert format_params_key({}) == "default"

    def test_nested_dict(self) -> None:
        """Test nested dict is JSON formatted."""
        params = {"config": {"a": 1}}

        result = format_params_key(params)

        assert 'config={"a":1}' in result

    def test_list_values(self) -> None:
        """Test list values are joined with semicolons."""
        params = {"values": [1, 2, 3]}

        result = format_params_key(params)

        assert "values=1;2;3" in result
