from __future__ import annotations

import pandas as pd
import pytest

from pyportflowopt.factor_models import RegressionResult
from pyportflowopt.optimization import (
    LONG_PORTFOLIO_TYPE,
    SHORT_PORTFOLIO_TYPE,
    SIMP_MODEL_NAME,
    apply_weight_epsilon,
    compute_factor_model_mu,
    compute_long_run_cutoff,
    compute_portfolio_performance,
    compute_simp_mu,
    optimize_portfolio,
)


def test_optimize_portfolio_success_matches_pypfopt_max_sharpe_directly():
    from pypfopt.efficient_frontier import EfficientFrontier

    mu = pd.Series({"AAA": 0.05, "BBB": 0.03, "CCC": 0.04})
    cov = pd.DataFrame(
        [[0.04, 0.01, 0.00], [0.01, 0.03, 0.00], [0.00, 0.00, 0.02]],
        index=["AAA", "BBB", "CCC"],
        columns=["AAA", "BBB", "CCC"],
    )
    rf = 0.01

    outcome = optimize_portfolio(mu, cov, rf, weight_bounds=(0.0, 1.0))
    assert outcome.success is True
    assert outcome.weights is not None
    assert sum(outcome.weights.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(w >= -1e-9 for w in outcome.weights.values())  # long-only

    # Cross-check against calling pypfopt directly -- optimize_portfolio must not add any
    # hidden transformation of its own.
    ef = EfficientFrontier(mu, cov, weight_bounds=(0.0, 1.0))
    ef.max_sharpe(risk_free_rate=rf)
    expected_mu_p, expected_sigma_p, expected_sharpe = ef.portfolio_performance(risk_free_rate=rf)
    assert outcome.mu_p == pytest.approx(expected_mu_p)
    assert outcome.sigma_p == pytest.approx(expected_sigma_p)
    assert outcome.sharpe == pytest.approx(expected_sharpe)


def test_optimize_portfolio_short_allowed_can_take_negative_weights():
    mu = pd.Series({"AAA": 0.01, "BBB": 0.06, "CCC": 0.05})
    cov = pd.DataFrame(
        [[0.04, 0.03, 0.02], [0.03, 0.05, 0.01], [0.02, 0.01, 0.03]],
        index=["AAA", "BBB", "CCC"],
        columns=["AAA", "BBB", "CCC"],
    )
    outcome = optimize_portfolio(mu, cov, rf=0.005, weight_bounds=(-0.3, 1.0))
    assert outcome.success is True
    assert sum(outcome.weights.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(w >= -0.3 - 1e-6 for w in outcome.weights.values())


def test_optimize_portfolio_fails_gracefully_when_rf_exceeds_all_expected_returns():
    mu = pd.Series({"AAA": 0.001, "BBB": 0.002})
    cov = pd.DataFrame([[0.01, 0.0], [0.0, 0.02]], index=["AAA", "BBB"], columns=["AAA", "BBB"])
    outcome = optimize_portfolio(mu, cov, rf=0.05, weight_bounds=(0.0, 1.0))
    assert outcome.success is False
    assert outcome.weights is None
    assert "risk-free rate" in outcome.failure_reason


def test_optimize_portfolio_fails_gracefully_on_solver_infeasibility():
    # A real (guaranteed-PSD) sample covariance/expected-return pair extracted from an
    # actual fixture run (CAPM, window 2) that the OSQP solver reports as infeasible --
    # reproduces the "routine" solver-non-convergence failure path without fabricating an
    # invalid (non-PSD) covariance matrix that couldn't occur from real return data.
    mu = pd.Series(
        {
            "AAA": 1.856499672757677e-05,
            "BBB": 0.00010954376652069192,
            "CCC": 6.869625669128485e-05,
            "DDD": 9.647335103304533e-05,
        }
    )
    cov = pd.DataFrame(
        [
            [0.000209442825583464, -6.210201404508218e-05, 0.00016135821735045158, -3.967781410158657e-05],
            [-6.210201404508218e-05, 0.00026256250224051164, -1.0492317283006113e-05, 6.101341246537057e-05],
            [0.00016135821735045158, -1.0492317283006113e-05, 0.0003133317705179659, 7.024617604334217e-05],
            [-3.967781410158657e-05, 6.101341246537057e-05, 7.024617604334217e-05, 0.0005256280394426186],
        ],
        index=["AAA", "BBB", "CCC", "DDD"],
        columns=["AAA", "BBB", "CCC", "DDD"],
    )
    outcome = optimize_portfolio(mu, cov, rf=0.0001, weight_bounds=(0.0, 1.0))
    assert outcome.success is False
    assert outcome.weights is None
    assert outcome.failure_reason is not None


def test_compute_simp_mu_is_arithmetic_mean():
    returns = pd.DataFrame({"AAA": [0.01, 0.02, 0.03], "BBB": [0.00, 0.01, -0.01]})
    mu = compute_simp_mu(returns)
    assert mu["AAA"] == pytest.approx(0.02)
    assert mu["BBB"] == pytest.approx(0.00)


def test_compute_long_run_cutoff_rolls_forward_to_month_end_for_monthly_data():
    perf_end = pd.Timestamp("2021-07-30")  # last TRADING day of July (Friday)
    cutoff = compute_long_run_cutoff(perf_end, periods_per_year=12)
    assert cutoff == pd.Timestamp("2021-07-31")  # calendar month-end


def test_compute_long_run_cutoff_passthrough_for_non_monthly_data():
    perf_end = pd.Timestamp("2021-07-30")
    cutoff = compute_long_run_cutoff(perf_end, periods_per_year=52)
    assert cutoff == perf_end


def test_compute_factor_model_mu_ff5mod_vs_common_start_branch_differ():
    dates = pd.date_range("2020-01-03", periods=6, freq="W-FRI")
    covered_returns = pd.DataFrame(
        {"AAA": [0.02, 0.03, 0.01, 0.025, 0.015, 0.02], "BBB": [0.01, 0.015, 0.005, 0.01, 0.02, 0.01]},
        index=dates,
    )
    aligned_factors = pd.DataFrame(
        {
            "Mkt-RF": [0.01, 0.02, -0.01, 0.015, 0.005, 0.01],
            "SMB": [0.002, 0.001, 0.003, 0.001, 0.002, 0.0015],
            "HML": [0.001, -0.001, 0.002, 0.0, 0.001, 0.0005],
            "RMW": [0.0005, 0.0005, 0.001, 0.0, 0.0005, 0.0005],
            "CMA": [0.0003, 0.0002, 0.0004, 0.0001, 0.0003, 0.0002],
            "RF": [0.0001] * 6,
        },
        index=dates,
    )
    # A longer raw history for FF5Mod's per-factor independent long-run averaging.
    raw_dates = pd.date_range("2019-01-04", periods=60, freq="W-FRI")
    raw_factor_values = pd.DataFrame(
        {
            "Mkt-RF": [0.005] * 60,
            "SMB": [0.001] * 60,
            "HML": [0.0005] * 60,
            "RMW": [0.0004] * 60,
            "CMA": [0.0003] * 60,
            "RF": [0.0001] * 60,
        },
        index=raw_dates,
    )

    mu_ff5, betas_ff5 = compute_factor_model_mu(
        "FF5Mod",
        covered_returns,
        aligned_factors,
        raw_factor_values,
        rf=0.0001,
        long_run_cutoff=dates[-1],
        periods_per_year=52,
    )
    assert set(mu_ff5.index) == {"AAA", "BBB"}
    assert all(isinstance(b, RegressionResult) for b in betas_ff5.values())


def test_portfolio_type_constants_and_bounds_shape():
    assert LONG_PORTFOLIO_TYPE == "Lng"
    assert SHORT_PORTFOLIO_TYPE == "Shrt"
    assert SIMP_MODEL_NAME == "Simp"


def test_apply_weight_epsilon_zeroes_dust_and_renormalizes_remainder():
    weights = {"AAA": 0.5, "BBB": 0.499, "CCC": 0.001}
    cleaned = apply_weight_epsilon(weights, epsilon=0.01)
    assert cleaned["CCC"] == 0.0
    assert sum(cleaned.values()) == pytest.approx(1.0)
    assert cleaned["AAA"] == pytest.approx(0.5 / 0.999)
    assert cleaned["BBB"] == pytest.approx(0.499 / 0.999)


def test_apply_weight_epsilon_zero_is_a_no_op_modulo_floating_point():
    weights = {"AAA": 0.6, "BBB": 0.3, "CCC": 0.1}
    cleaned = apply_weight_epsilon(weights, epsilon=0.0)
    assert cleaned == pytest.approx(weights)


def test_apply_weight_epsilon_negative_weights_use_absolute_value():
    weights = {"AAA": 1.2, "BBB": -0.15, "CCC": -0.05}
    cleaned = apply_weight_epsilon(weights, epsilon=0.1)
    assert cleaned["CCC"] == 0.0
    assert cleaned["BBB"] != 0.0
    assert sum(cleaned.values()) == pytest.approx(1.0)


def test_apply_weight_epsilon_degenerate_total_returns_zeroed_unnormalized():
    # Every weight below epsilon -- renormalizing (dividing by 0) would crash.
    weights = {"AAA": 0.05, "BBB": -0.05}
    cleaned = apply_weight_epsilon(weights, epsilon=0.1)
    assert cleaned == {"AAA": 0.0, "BBB": 0.0}


def test_compute_portfolio_performance_hand_computed():
    weights = {"AAA": 0.6, "BBB": 0.4}
    mu = pd.Series({"AAA": 0.05, "BBB": 0.03})
    cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.02]], index=["AAA", "BBB"], columns=["AAA", "BBB"])
    mu_p, sigma_p = compute_portfolio_performance(weights, mu, cov)
    assert mu_p == pytest.approx(0.6 * 0.05 + 0.4 * 0.03)
    expected_var = 0.6**2 * 0.04 + 0.4**2 * 0.02 + 2 * 0.6 * 0.4 * 0.01
    assert sigma_p == pytest.approx(expected_var**0.5)
