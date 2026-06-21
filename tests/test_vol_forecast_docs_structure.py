"""Doc-structure tests for the consolidated vol-forecast doc.

These tests assert the existence + section headers of
``docs/vol-forecast.md`` so the consolidated surface stays present and
shaped consistently across edits. They do not police prose — only the
load-bearing anchors a downstream consumer is expected to find.
"""

from __future__ import annotations

from pathlib import Path


_DOC_PATH = Path(__file__).parent.parent / "docs" / "vol-forecast.md"
_REQUIRED_HEADERS = (
    "# Volatility forecast features",
    "## Forecast contracts",
    "## Targets",
    "## Multiscale features",
    "## Regime features",
    "## Serving clocks",
    "## Universes",
)
_REQUIRED_SYMBOLS = (
    "VolForecastFeatures",
    "ForecastTarget",
    "build_target_rv_total",
    "build_intraday_reversal_target",
    "compute_semivariance",
    "compute_asymmetry_regression",
    "derive_gap_jump_labels",
    "resolve_multi_label",
    "GAP_DOMINATED_VOL",
    "INTRADAY_RANGE_DOMINATED_VOL",
    "JUMP_DAY",
    "compute_universe_membership",
)


def _read_doc() -> str:
    assert _DOC_PATH.exists(), f"missing consolidated doc at {_DOC_PATH}"
    return _DOC_PATH.read_text(encoding="utf-8")


def test_consolidated_doc_exists() -> None:
    assert _DOC_PATH.exists()


def test_consolidated_doc_carries_every_required_section_header() -> None:
    body = _read_doc()
    for header in _REQUIRED_HEADERS:
        assert header in body, f"missing section header: {header!r}"


def test_consolidated_doc_references_every_load_bearing_symbol() -> None:
    body = _read_doc()
    for symbol in _REQUIRED_SYMBOLS:
        assert symbol in body, f"missing symbol reference: {symbol!r}"


def test_consolidated_doc_does_not_carry_phase_language() -> None:
    body = _read_doc()
    forbidden = ("F0", "F1H", "F2", "F2H", "F3", "F3H", "F4", "F4H", "F5", "F5H", "FF+1")
    for token in forbidden:
        assert f" {token} " not in body, f"phase language leaked: {token!r}"
        assert f"({token})" not in body, f"phase language leaked: {token!r}"
