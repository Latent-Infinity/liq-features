from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest

from liq.features.vol_forecast import build_target_rv_total


def _bars() -> pl.DataFrame:
    previous_ts = datetime(2024, 1, 2, 20, 59, tzinfo=UTC)
    start = datetime(2024, 1, 3, 14, 30, tzinfo=UTC)
    rows = [
        {
            "timestamp": previous_ts,
            "open": 99.0,
            "close": 100.0,
            "corporate_action_effective_ts": None,
            "corporate_action_event_id": None,
        }
    ]
    price = 101.0
    for idx in range(21):
        rows.append(
            {
                "timestamp": start + timedelta(minutes=idx),
                "open": price,
                "close": price * 1.001,
                "corporate_action_effective_ts": datetime(2024, 1, 4, tzinfo=UTC)
                if idx == 0
                else None,
                "corporate_action_event_id": "split_a" if idx == 0 else None,
            }
        )
        price *= 1.001
    return pl.DataFrame(rows)


def test_target_rv_total_adds_overnight_gap_and_intraday_rv() -> None:
    target = build_target_rv_total(_bars(), date(2024, 1, 3), symbol="AAPL")
    expected_gap = math.log(101.0 / 100.0) ** 2
    expected_intraday = 21 * (math.log(1.001) ** 2)

    assert target.overnight_gap_var == pytest.approx(expected_gap)
    assert target.intraday_bar_rv == pytest.approx(expected_intraday)
    assert target.target_rv_total == pytest.approx(expected_gap + expected_intraday)
    assert target.raw_price_start == 101.0
    assert target.raw_price_end == pytest.approx(101.0 * (1.001**21))


def test_ca_adjacent_rows_are_reason_coded_and_not_promotion_eligible() -> None:
    target = build_target_rv_total(_bars(), date(2024, 1, 3), symbol="AAPL")

    assert not target.promotion_eligible
    assert [reason.code for reason in target.reason_codes] == ["CA_ADJACENT"]
    assert target.corporate_action_event_id == "split_a"
