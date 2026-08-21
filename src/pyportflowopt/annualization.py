from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForecastAnnualization:
    """Forecast side stays arithmetic (preserves MVO's mu_p = w^T mu linearity --
    compounding it would break consistency between per-asset and portfolio annualized
    figures). bkTstExpR/bkTstExpVol are NaN when bck_test_wndw is None (no bckTestWndw
    configured at all, i.e. doBcktest: false for the whole run). YrExpRF is rf arithmetically
    annualized (rf * p) -- algebraically the exact RF term already implicit in YrExpShrp's
    (mu_p - rf) * sqrt(p) / sigma_p, since p / sqrt(p) == sqrt(p)."""

    YrExpR: float
    bkTstExpR: float
    YrExpVol: float
    bkTstExpVol: float
    YrExpRF: float
    YrExpShrp: float


@dataclass(frozen=True)
class RealizedPerformance:
    """Realized side is geometric (measures what actually happened)."""

    YrRealR: float
    bkTstRealR: float
    YrRealVol: float
    bkTstRealVol: float


NAN_REALIZED_PERFORMANCE = RealizedPerformance(
    YrRealR=float("nan"), bkTstRealR=float("nan"), YrRealVol=float("nan"), bkTstRealVol=float("nan")
)


def compute_forecast_annualization(
    mu_p: float, sigma_p: float, rf: float, periods_per_year: int, bck_test_wndw: int | None
) -> ForecastAnnualization:
    p = periods_per_year
    yr_exp_r = mu_p * p
    yr_exp_vol = sigma_p * math.sqrt(p)
    yr_exp_rf = rf * p
    yr_exp_shrp = (mu_p - rf) * math.sqrt(p) / sigma_p
    if bck_test_wndw is None:
        bcktest_exp_r = float("nan")
        bcktest_exp_vol = float("nan")
    else:
        bcktest_exp_r = mu_p * bck_test_wndw
        bcktest_exp_vol = sigma_p * math.sqrt(bck_test_wndw)
    return ForecastAnnualization(
        YrExpR=yr_exp_r,
        bkTstExpR=bcktest_exp_r,
        YrExpVol=yr_exp_vol,
        bkTstExpVol=bcktest_exp_vol,
        YrExpRF=yr_exp_rf,
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
        bkTstRealR=bcktest_real_r,
        YrRealVol=yr_real_vol,
        bkTstRealVol=bcktest_real_vol,
    )


def compute_realized_rf(rf_period_returns: pd.Series | np.ndarray, periods_per_year: int) -> float:
    """Same geometric-compounding-then-CAGR-annualization treatment as
    compute_realized_performance's YrRealR, applied to the bcktest window's RF series
    instead of portfolio returns -- so YrRealShrp compares two figures built the same way."""
    r = np.asarray(rf_period_returns, dtype=float)
    b = len(r)
    p = periods_per_year
    bcktest_real_rf = float(np.prod(1.0 + r) - 1.0)
    return (1.0 + bcktest_real_rf) ** (p / b) - 1.0


def compute_realized_sharpe(yr_real_r: float, yr_real_rf: float, yr_real_vol: float) -> float:
    """Annualized realized Sharpe ratio -- mirrors YrExpShrp's excess-return-over-vol shape,
    but with both the return and the risk-free rate already CAGR-annualized from the
    bcktest window's actual compounded values, rather than derived from a single
    point-in-time rf."""
    return (yr_real_r - yr_real_rf) / yr_real_vol


def residual_del_exp_real(expected: float, realized: float) -> float:
    """Expected - Realized -- NaN propagates automatically when realized is NaN
    (e.g. Final window, doBcktest: false, or a failed optimization), which is what makes
    'no Real*/DelExpReal* columns anywhere' fall out of this one shared code path."""
    return expected - realized


def residual_prem_real_mkt(portfolio_realized: float, market_realized: float) -> float:
    """Return[portfolio] - Return[MarketProxy], both realized, same annualization scale.
    NaN when either side is NaN -- including whenever no market proxy is configured."""
    return portfolio_realized - market_realized
