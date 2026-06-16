"""Contract tests for the volatility logging facade.

The estimator emits structured INFO/ERROR events on the
``liq.features.volatility`` logger. Every event MUST carry the fields
called out in research plan §1.9, and decision branches MUST emit
exactly once per bar.

Tests pin:

- Required fields on every emitted record.
- No-secrets sentinel — a fake `secret=xyz` in the bound logger context
  is redacted, never present in the formatted message.
- Per-bar debouncing — multiple quality flags on one bar emit ONE
  ``quality_flag_set`` event (with all flags), not N events.
- Event catalog completeness — all canonical events register through a
  single helper so the catalog cannot drift.
"""

from __future__ import annotations

import logging

import pytest

from liq.features.volatility.logging import (
    LOGGER_NAME,
    VolEventEmitter,
    build_emitter,
)


@pytest.fixture
def log_records(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    return caplog


def _make_emitter(**overrides: object) -> VolEventEmitter:
    defaults: dict[str, object] = {
        "symbol": "AAPL",
        "asof": "2024-06-03T20:00:00Z",
        "window": 21,
        "spec_hash": "deadbeef",
        "estimator": "yang_zhang",
        "correlation_id": None,
    }
    defaults.update(overrides)
    return build_emitter(**defaults)  # type: ignore[arg-type]


class TestRequiredFields:
    def test_every_event_has_required_fields(self, log_records) -> None:
        emitter = _make_emitter()
        emitter.estimator_selected()
        emitter.estimator_fallback_applied(
            from_="yang_zhang", to="rogers_satchell", reason="missing_prev_close"
        )
        emitter.quality_flag_set(bar_index=3, flags=["HIGH_LOW_OUTLIER"])
        emitter.data_quality_rejected(rule="failure_rate_exceeded", details={"rate": 0.02})

        required = {
            "timestamp",
            "level",
            "event",
            "correlation_id",
            "estimator_used",
            "spec_hash",
            "symbol",
            "asof",
            "quality_flags",
        }
        # caplog records have the canonical context-binding the emitter sets.
        for record in log_records.records:
            extra = getattr(record, "structured", {})
            missing = required - set(extra)
            assert not missing, f"record missing fields {missing}: {extra}"

    def test_per_estimate_bound_fields(self, log_records) -> None:
        emitter = _make_emitter(symbol="MSFT", window=63)
        emitter.estimator_selected()
        record = log_records.records[-1]
        extra = getattr(record, "structured", {})
        assert extra["symbol"] == "MSFT"
        assert extra["window"] == 63
        assert extra["asof"] == "2024-06-03T20:00:00Z"

    def test_correlation_id_is_stable_within_emitter(self, log_records) -> None:
        emitter = _make_emitter()
        emitter.estimator_selected()
        emitter.quality_flag_set(bar_index=1, flags=["LOW_CONFIDENCE"])
        ids = {getattr(r, "structured", {}).get("correlation_id") for r in log_records.records}
        assert len(ids) == 1, f"correlation_id leaked across events: {ids}"
        assert next(iter(ids)) is not None

    def test_explicit_correlation_id_is_honored(self, log_records) -> None:
        emitter = _make_emitter(correlation_id="op-12345")
        emitter.estimator_selected()
        record = log_records.records[-1]
        assert getattr(record, "structured", {}).get("correlation_id") == "op-12345"


class TestNoSecrets:
    def test_secret_in_extra_is_redacted(self, log_records) -> None:
        emitter = _make_emitter()
        emitter.estimator_selected(secret="xyz")
        record = log_records.records[-1]
        structured = getattr(record, "structured", {})
        # The sentinel `secret` key must not leak; if the emitter
        # surfaces it at all, the value must be redacted.
        assert "secret" not in structured or structured.get("secret") == "***"
        assert "xyz" not in record.getMessage()


class TestPerBarDebouncing:
    def test_multiple_flags_same_bar_emit_one_event(self, log_records) -> None:
        emitter = _make_emitter()
        emitter.quality_flag_set(bar_index=2, flags=["HIGH_LOW_OUTLIER", "GAP_DOMINATED_VOL"])
        events = [
            r
            for r in log_records.records
            if getattr(r, "structured", {}).get("event") == "quality_flag_set"
        ]
        assert len(events) == 1
        extra = getattr(events[0], "structured", {})
        assert set(extra["quality_flags"]) == {"HIGH_LOW_OUTLIER", "GAP_DOMINATED_VOL"}

    def test_calling_quality_flag_twice_for_same_bar_is_idempotent(self, log_records) -> None:
        emitter = _make_emitter()
        emitter.quality_flag_set(bar_index=2, flags=["HIGH_LOW_OUTLIER"])
        emitter.quality_flag_set(bar_index=2, flags=["GAP_DOMINATED_VOL"])
        events = [
            r
            for r in log_records.records
            if getattr(r, "structured", {}).get("event") == "quality_flag_set"
        ]
        # Second call for same bar adds the flag set; total events for
        # bar 2 stays at 1 (debouncing in effect).
        assert len(events) == 1
        extra = getattr(events[0], "structured", {})
        assert set(extra["quality_flags"]) == {"HIGH_LOW_OUTLIER", "GAP_DOMINATED_VOL"}


class TestEventCatalog:
    def test_data_quality_rejected_is_error_level(self, log_records) -> None:
        emitter = _make_emitter()
        emitter.data_quality_rejected(rule="high_lt_low", details={"bar_index": 5})
        record = log_records.records[-1]
        assert record.levelno == logging.ERROR

    def test_fallback_applied_includes_from_to_reason(self, log_records) -> None:
        emitter = _make_emitter()
        emitter.estimator_fallback_applied(
            from_="yang_zhang", to="rogers_satchell", reason="missing_prev_close"
        )
        extra = getattr(log_records.records[-1], "structured", {})
        assert extra["from"] == "yang_zhang"
        assert extra["to"] == "rogers_satchell"
        assert extra["reason"] == "missing_prev_close"
