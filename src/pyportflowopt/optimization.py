from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pypfopt import exceptions as pypfopt_exceptions
from pypfopt.efficient_frontier import EfficientFrontier

from .config import FACTOR_MODEL_CATALOG
from .factor_models import (
    RegressionResult,
    compute_common_start_long_run_factor_expected_returns,
    compute_expected_return,
    compute_joint_factor_regression,
    compute_long_run_factor_expected_returns,
)

SIMP_MODEL_NAME = "Simp"
LONG_PORTFOLIO_TYPE = "Lng"
SHORT_PORTFOLIO_TYPE = "Shrt"


@dataclass(frozen=True)
class OptimizationOutcome:
    """Never raised as an exception -- a failed max_sharpe() (no asset beats rf, or the
    solver doesn't converge) is a routine per-(window, model, portfolio-type) occurrence,
    not an exceptional one."""

    success: bool
    weights: dict[str, float] | None
    mu_p: float | None
    sigma_p: float | None
    sharpe: float | None
    failure_reason: str | None


def compute_long_run_cutoff(perf_end: pd.Timestamp, periods_per_year: int) -> pd.Timestamp:
    """Rolls the window's perf-end date forward to the calendar month-end for monthly
    data, so the window's own final month isn't spuriously excluded from a factor's
    long-run history (mirrors PyPortFlow2's pipeline.py long_run_cutoff handling)."""
    if periods_per_year == 12:
        return perf_end + pd.offsets.MonthEnd(0)
    return perf_end


def compute_simp_mu(covered_returns: pd.DataFrame) -> pd.Series:
    """Simp model: arithmetic mean of the window's covered-universe returns."""
    return covered_returns.mean()


def compute_factor_model_mu(
    model_name: str,
    covered_returns: pd.DataFrame,
    aligned_factors: pd.DataFrame,
    raw_factor_values: pd.DataFrame,
    rf: float,
    long_run_cutoff: pd.Timestamp,
    periods_per_year: int,
) -> tuple[pd.Series, dict[str, RegressionResult]]:
    """Per-ticker joint factor regression (excess return = ticker return - RF from the
    aligned factor window), then E[R_i] = rf + sum(beta_k * long-run E[factor_k]).

    Preserves the FF5Mod-vs-everyone-else branch exactly as PyPortFlow2's pipeline.py
    orchestrates it: FF5Mod averages each factor over its own independent full history;
    every other model averages all its factors over one shared common-start date.
    """
    factor_names = FACTOR_MODEL_CATALOG[model_name]
    if model_name == "FF5Mod":
        long_run_expected_returns = compute_long_run_factor_expected_returns(
            raw_factor_values, long_run_cutoff
        )
    else:
        long_run_expected_returns = compute_common_start_long_run_factor_expected_returns(
            raw_factor_values, factor_names, long_run_cutoff
        )

    rf_series = aligned_factors["RF"]
    betas_for_model: dict[str, RegressionResult] = {}
    mu_values: dict[str, float] = {}
    for ticker in covered_returns.columns:
        excess = covered_returns[ticker] - rf_series
        result = compute_joint_factor_regression(excess, aligned_factors, factor_names, periods_per_year)
        betas_for_model[ticker] = result
        mu_values[ticker] = compute_expected_return(rf, result.betas, long_run_expected_returns)

    return pd.Series(mu_values), betas_for_model


def optimize_portfolio(
    mu: pd.Series, cov: pd.DataFrame, rf: float, weight_bounds: tuple[float, float]
) -> OptimizationOutcome:
    """EfficientFrontier(mu, cov, weight_bounds).max_sharpe(risk_free_rate=rf) IS the
    tangent portfolio, no separate step. Never calls pypfopt.expected_returns.* or
    pypfopt.risk_models.* -- mu/cov are always precomputed and passed in directly, so
    pypfopt's own geometric-compounding trap (confined to expected_returns.*) never
    applies here."""
    try:
        ef = EfficientFrontier(mu, cov, weight_bounds=weight_bounds)
        ef.max_sharpe(risk_free_rate=rf)
        weights = dict(zip(mu.index, ef.weights, strict=True))
        mu_p, sigma_p, sharpe = ef.portfolio_performance(risk_free_rate=rf)
        return OptimizationOutcome(
            success=True, weights=weights, mu_p=mu_p, sigma_p=sigma_p, sharpe=sharpe, failure_reason=None
        )
    except (ValueError, pypfopt_exceptions.OptimizationError) as exc:
        return OptimizationOutcome(
            success=False, weights=None, mu_p=None, sigma_p=None, sharpe=None, failure_reason=str(exc)
        )


def apply_weight_epsilon(weights: dict[str, float], epsilon: float) -> dict[str, float]:
    """Zeroes out any weight with abs(w) < epsilon (dust cleanup), then renormalizes the
    remainder to sum to 1. Guards total <= 0 by returning the zeroed-but-unnormalized dict
    unchanged -- dividing by a non-positive total would crash or, worse for a Shrt portfolio,
    silently flip every position's sign. Should be rare in practice: the optimizer's own
    constraint already sums raw weights to 1, and epsilon is meant to catch only dust."""
    cleaned = {ticker: (w if abs(w) >= epsilon else 0.0) for ticker, w in weights.items()}
    total = sum(cleaned.values())
    if total <= 0:
        return cleaned
    return {ticker: w / total for ticker, w in cleaned.items()}


def compute_portfolio_performance(
    weights: dict[str, float], mu: pd.Series, cov: pd.DataFrame
) -> tuple[float, float]:
    """Recomputes (mu_p, sigma_p) directly from the given weights -- needed after
    apply_weight_epsilon, since pypfopt's ef.portfolio_performance() reflects the raw
    pre-epsilon weights and must not be reused once weights are cleaned."""
    w = pd.Series(weights).reindex(mu.index)
    mu_p = float((w * mu).sum())
    sigma_p = float(np.sqrt(w.to_numpy() @ cov.loc[w.index, w.index].to_numpy() @ w.to_numpy()))
    return mu_p, sigma_p
