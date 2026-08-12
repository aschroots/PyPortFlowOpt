from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForecastAnnualization:
    """Forecast side stays arithmetic (preserves MVO's mu_p = w^T mu linearity --
    compounding it would break consistency between per-asset and portfolio annualized
    figures). bckTestExpR/bckTestExpVol are NaN when bck_test_wndw is None (no
    bckTestWndw configured at all, i.e. doBcktest: false for the whole run)."""

    YrExpR: float
    bckTestExpR: float
    YrExpVol: float
    bckTestExpVol: float
    YrExpShrp: float


@dataclass(frozen=True)
class RealizedPerformance:
    """Realized side is geometric (measures what actually happened)."""

    YrRealR: float
    bckTestRealR: float
    YrRealVol: float
    bckTestRealVol: float


NAN_REALIZED_PERFORMANCE = RealizedPerformance(
    YrRealR=float("nan"), bckTestRealR=float("nan"), YrRealVol=float("nan"), bckTestRealVol=float("nan")
)


def compute_forecast_annualization(
    mu_p: float, sigma_p: float, rf: float, periods_per_year: int, bck_test_wndw: int | None
) -> ForecastAnnualization:
    p = periods_per_year
    yr_exp_r = mu_p * p
    yr_exp_vol = sigma_p * math.sqrt(p)
    yr_exp_shrp = (mu_p - rf) * math.sqrt(p) / sigma_p
    if bck_test_wndw is None:
        bcktest_exp_r = float("nan")
        bcktest_exp_vol = float("nan")
    else:
        bcktest_exp_r = mu_p * bck_test_wndw
        bcktest_exp_vol = sigma_p * math.sqrt(bck_test_wndw)
    return ForecastAnnualization(
        YrExpR=yr_exp_r,
        bckTestExpR=bcktest_exp_r,
        YrExpVol=yr_exp_vol,
        bckTestExpVol=bcktest_exp_vol,
        YrExpShrp=yr_exp_shrp,
    )


def compute_realized_performance(
    portfolio_period_returns: pd.Series | np.ndarray, periods_per_year: int
) -> RealizedPerformance:
    """Applies the window's weights to the ACTUAL bcktest-period returns (already
    computed by the caller) -> portfolio realized period returns r_p,t (t=1..B)."""
    r = np.asarray(portfolio_period_returns, dtype=float)
    b = len(r)
    p = periods_per_year
    bcktest_real_r = float(np.prod(1.0 + r) - 1.0)
    yr_real_r = (1.0 + bcktest_real_r) ** (p / b) - 1.0  # CAGR annualization
    stdev = float(np.std(r, ddof=1))
    bcktest_real_vol = stdev * math.sqrt(b)
    yr_real_vol = stdev * math.sqrt(p)
    return RealizedPerformance(
        YrRealR=yr_real_r,
        bckTestRealR=bcktest_real_r,
        YrRealVol=yr_real_vol,
        bckTestRealVol=bcktest_real_vol,
    )


def residual_del_fore_real(expected: float, realized: float) -> float:
    """Expected - Realized -- NaN propagates automatically when realized is NaN
    (e.g. Final window, doBcktest: false, or a failed optimization), which is what makes
    'no Real*/DelForeReal* columns anywhere' fall out of this one shared code path."""
    return expected - realized


def residual_prem_real_mkt(portfolio_realized: float, market_realized: float) -> float:
    """Return[portfolio] - Return[MarketProxy], both realized, same annualization scale.
    NaN when either side is NaN -- including whenever no market proxy is configured."""
    return portfolio_realized - market_realized
