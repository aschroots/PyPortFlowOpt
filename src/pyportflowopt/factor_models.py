from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .errors import FactorAlignmentError, FatalPipelineError


@dataclass(frozen=True)
class RegressionResult:
    alpha: float  # period-native (unannualized) -- see run_ols_point_estimates
    betas: dict[str, float]  # factor_name -> point-estimate beta


def build_design_matrix(factor_returns: pd.DataFrame, enabled_factor_names: Sequence[str]) -> np.ndarray:
    """[1 | factor_returns[enabled_factor_names]] -- intercept column first."""
    missing = [name for name in enabled_factor_names if name not in factor_returns.columns]
    if missing:
        raise FatalPipelineError(
            f"Factor column(s) required by enabled config flags not found in factor file: {missing}"
        )
    n = len(factor_returns)
    intercept = np.ones((n, 1))
    factor_block = factor_returns[list(enabled_factor_names)].to_numpy(dtype=float)
    return np.hstack([intercept, factor_block])


def run_ols_point_estimates(
    y: np.ndarray,
    x_with_intercept: np.ndarray,
    factor_names: Sequence[str],
    periods_per_year: int,
) -> RegressionResult:
    """periods_per_year is accepted (not used) to keep this call site, and
    compute_joint_factor_regression's, identical to PyPortFlow2's original -- alpha is
    intentionally left period-native here (no `* periods_per_year`), unlike PyPortFlow2's
    alpha_annualized, since this project needs everything in matching period-native units."""
    coef, *_ = np.linalg.lstsq(x_with_intercept, y, rcond=None)
    alpha = float(coef[0])
    betas = {name: float(coef[i + 1]) for i, name in enumerate(factor_names)}
    return RegressionResult(alpha=alpha, betas=betas)


def compute_joint_factor_regression(
    excess_security_returns: pd.Series,
    factor_returns: pd.DataFrame,
    enabled_factor_names: Sequence[str],
    periods_per_year: int,
) -> RegressionResult:
    """Joint multiple regression over whichever factors a named model specifies.

    A single-factor model (e.g. CAPM, enabled_factor_names == ("Mkt-RF",)) is just
    the degenerate univariate case of this same regression -- no special-casing
    needed for CAPM vs. any other named factor model.
    """
    frame = {"y": excess_security_returns}
    for name in enabled_factor_names:
        frame[name] = factor_returns[name]
    aligned = pd.concat(frame, axis=1).dropna()
    x = build_design_matrix(aligned[list(enabled_factor_names)], enabled_factor_names)
    return run_ols_point_estimates(
        aligned["y"].to_numpy(dtype=float), x, enabled_factor_names, periods_per_year
    )


def realign_monthly_factor_dates_to_price_dates(
    factor_values: pd.DataFrame, target_dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """Remaps factor rows from their own month-end date convention onto the
    price/return file's own dates, matched by calendar (year, month) -- for
    monthly data, a trading-day month-end price file and a calendar-month-end
    factor file can't be expected to share the exact same date (e.g. July 2021:
    a price file's last trading day is 2021-07-30, a Friday, while the calendar
    month-end is 2021-07-31, a Saturday). Matching by month sidesteps that
    mismatch entirely rather than requiring exact date equality.

    A target date whose month has no corresponding factor row becomes an
    all-NaN row (equivalent to a missing date under exact-match comparison).
    If a month has more than one factor row (shouldn't normally happen), the
    chronologically last one is used.
    """
    factor_by_period = factor_values.groupby(factor_values.index.to_period("M")).last()
    aligned = factor_by_period.reindex(target_dates.to_period("M"))
    aligned.index = target_dates
    return aligned


def align_factor_window(
    factor_values: pd.DataFrame, window_dates: pd.DatetimeIndex, periods_per_year: int
) -> pd.DataFrame:
    """Aligns the factor file's own dates onto a single window's dates -- monthly data
    (periods_per_year == 12) matches by (year, month) via realign_monthly_factor_dates_to_price_dates,
    daily/weekly data requires exact date equality (a plain reindex). Raises
    FactorAlignmentError only when a window date is WHOLLY missing from the factor file
    (every column NaN for that date); partial per-column NaN (e.g. SMB5Fac starting later
    than Mkt-RF) is tolerated here and left to compute_joint_factor_regression's own
    .dropna(). Called once per window (dozens of times per run), unlike PyPortFlow2's
    single global alignment.
    """
    if periods_per_year == 12:
        aligned = realign_monthly_factor_dates_to_price_dates(factor_values, window_dates)
        missing_dates = aligned.index[aligned.isna().all(axis=1)]
    else:
        aligned = factor_values.reindex(window_dates)
        missing_dates = window_dates.difference(factor_values.index)

    if len(missing_dates) > 0:
        raise FactorAlignmentError(
            f"Factor file is missing data for {len(missing_dates)} window date(s), "
            f"e.g. {missing_dates[0].date()}."
        )
    return aligned


def compute_long_run_factor_expected_returns(
    factor_values: pd.DataFrame, as_of_date: pd.Timestamp
) -> dict[str, float]:
    """Period-native average return per factor column, each using its OWN independently
    available factor-file history up to (and including) as_of_date -- independent of
    any regression window and independent of every other factor's own history length.

    This is deliberately decoupled from the (typically much shorter) regression
    window used to estimate beta: beta should track recent risk exposure, but the
    expected factor return is a long-run premium best estimated from as much
    history as is available, not the short window's own (noisy) average.

    Used only for the FF5Mod model. Every other model uses
    compute_common_start_long_run_factor_expected_returns instead, which forces a
    single shared start date across a model's factors rather than letting each
    factor independently use its own (potentially much longer) history.
    """
    history = factor_values.loc[factor_values.index <= as_of_date]
    return {name: float(history[name].dropna().mean()) for name in factor_values.columns}


def compute_common_start_long_run_factor_expected_returns(
    factor_values: pd.DataFrame,
    factor_names: Sequence[str],
    as_of_date: pd.Timestamp,
) -> dict[str, float]:
    """Period-native average return per factor in factor_names, all computed over the
    SAME shared date range: from the latest (most restrictive) of each factor's own
    oldest available date, up to (and including) as_of_date.

    Example: if Mkt-RF has data back to 1930-01 but SMB5Fac only back to 1950-03,
    every factor in the model -- including Mkt-RF -- is averaged starting 1950-03,
    not each independently from its own earliest date. This keeps every factor in
    a given model comparable over an identical period, unlike
    compute_long_run_factor_expected_returns (used only for FF5Mod), where each
    factor independently uses as much of its own history as is available.
    """
    missing = [name for name in factor_names if name not in factor_values.columns]
    if missing:
        raise FatalPipelineError(
            f"Factor column(s) required by enabled config flags not found in factor file: {missing}"
        )

    history = factor_values.loc[factor_values.index <= as_of_date]
    oldest_dates = [history[name].first_valid_index() for name in factor_names]
    valid_oldest_dates = [d for d in oldest_dates if d is not None]
    if not valid_oldest_dates:
        raise FatalPipelineError(
            f"No data available at or before {as_of_date.date()} for any of {list(factor_names)}."
        )
    common_start = max(valid_oldest_dates)
    shared_history = history.loc[history.index >= common_start]
    return {name: float(shared_history[name].dropna().mean()) for name in factor_names}


def latest_value_in_window(series: pd.Series, window_start: pd.Timestamp, window_end: pd.Timestamp) -> float:
    """Most recent non-null value within [window_start, window_end] -- bounded strictly
    to the window's own date range, no annualization. Replaces PyPortFlow2's
    get_annualized_value_as_of (unbounded look-back over the entire file, then
    `* periods_per_year`) -- a deliberate behavioral deviation, not an oversight, since
    the spec calls for "the latest date available risk-free rate value within the
    perfWndw date range"."""
    eligible = series.loc[(series.index >= window_start) & (series.index <= window_end)].dropna()
    if eligible.empty:
        raise FatalPipelineError(
            f"No data available in window [{window_start.date()}, {window_end.date()}] for '{series.name}'."
        )
    return float(eligible.iloc[-1])


def compute_factor_date_ranges(factor_values: pd.DataFrame) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """Oldest and newest non-null date per factor column, since different factors
    commonly have different amounts of history in the same file (e.g. Mkt-RF back
    to 1930, SMB only from 1960 on)."""
    return {
        name: (factor_values[name].first_valid_index(), factor_values[name].last_valid_index())
        for name in factor_values.columns
    }


def compute_expected_return(
    rf: float, betas: dict[str, float], long_run_expected_factor_returns: dict[str, float]
) -> float:
    """E[R_i] = RF + sum(beta_k * E[factor_k]), using window-estimated betas but
    long-run (full available history) expected factor returns -- both period-native."""
    return rf + sum(betas[name] * long_run_expected_factor_returns[name] for name in betas)
