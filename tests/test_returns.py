from __future__ import annotations

import pandas as pd
import pytest

from pyportflowopt.errors import FatalPipelineError
from pyportflowopt.returns import detect_periods_per_year, prices_to_simple_returns


def test_prices_to_simple_returns_formula():
    prices = pd.DataFrame({"AAA": [100.0, 110.0, 99.0]})
    returns = prices_to_simple_returns(prices)
    assert len(returns) == 2
    assert returns["AAA"].iloc[0] == pytest.approx(0.10)
    assert returns["AAA"].iloc[1] == pytest.approx(-0.10)


@pytest.mark.parametrize(
    "freq, expected_periods",
    [
        ("D", 252),
        ("7D", 52),
        ("30D", 12),
        ("90D", 4),
        ("365D", 1),
    ],
)
def test_detect_periods_per_year_bands(freq, expected_periods):
    dates = pd.date_range("2020-01-01", periods=10, freq=freq)
    assert detect_periods_per_year(pd.DatetimeIndex(dates)) == expected_periods


def test_detect_periods_per_year_too_few_dates_is_fatal():
    with pytest.raises(FatalPipelineError):
        detect_periods_per_year(pd.DatetimeIndex(["2020-01-01"]))


def test_detect_periods_per_year_boundary_exactly_four_days():
    dates = pd.DatetimeIndex(["2020-01-01", "2020-01-05", "2020-01-09"])
    assert detect_periods_per_year(dates) == 252
