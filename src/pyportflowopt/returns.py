from __future__ import annotations

import pandas as pd

from .errors import FatalPipelineError

# (max median day-gap inclusive, periods per year)
_ANNUALIZATION_BANDS: tuple[tuple[float, int], ...] = (
    (4.0, 252),
    (10.0, 52),
    (45.0, 12),
    (135.0, 4),
    (float("inf"), 1),
)


def prices_to_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """R_t = P_t/P_{t-1} - 1; drops the first (all-NaN) row."""
    returns = prices.pct_change()
    return returns.iloc[1:]


def detect_periods_per_year(dates: pd.DatetimeIndex) -> int:
    """Median day-gap between consecutive rows -> nearest of {252, 52, 12, 4, 1}."""
    if len(dates) < 2:
        raise FatalPipelineError("Cannot detect data frequency from fewer than 2 dates.")
    gaps = pd.Series(dates).diff().dropna().dt.days
    median_gap = float(gaps.median())
    for threshold, periods in _ANNUALIZATION_BANDS:
        if median_gap <= threshold:
            return periods
    return 1
