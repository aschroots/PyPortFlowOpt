from __future__ import annotations

import math

import pytest

from pyportflowopt.annualization import (
    NAN_REALIZED_PERFORMANCE,
    compute_forecast_annualization,
    compute_realized_performance,
    compute_realized_rf,
    compute_realized_sharpe,
    residual_del_exp_real,
    residual_prem_real_mkt,
)


def test_forecast_annualization_hand_computed():
    mu_p = 0.002
    sigma_p = 0.01
    rf = 0.0001
    p = 52
    b = 6

    forecast = compute_forecast_annualization(mu_p, sigma_p, rf, p, b)

    assert forecast.YrExpR == pytest.approx(mu_p * p)
    assert forecast.bkTstExpR == pytest.approx(mu_p * b)
    assert forecast.YrExpVol == pytest.approx(sigma_p * math.sqrt(p))
    assert forecast.bkTstExpVol == pytest.approx(sigma_p * math.sqrt(b))
    assert forecast.YrExpRF == pytest.approx(rf * p)
    assert forecast.YrExpShrp == pytest.approx((mu_p - rf) * math.sqrt(p) / sigma_p)
    # YrExpShrp is algebraically (YrExpR - YrExpRF) / YrExpVol.
    assert forecast.YrExpShrp == pytest.approx((forecast.YrExpR - forecast.YrExpRF) / forecast.YrExpVol)


def test_forecast_annualization_bcktest_wndw_none_gives_nan_bcktest_fields_only():
    forecast = compute_forecast_annualization(0.002, 0.01, 0.0001, periods_per_year=52, bck_test_wndw=None)
    assert math.isnan(forecast.bkTstExpR)
    assert math.isnan(forecast.bkTstExpVol)
    # Yr* fields only depend on P, mu_p, sigma_p, rf -- still populated with no B at all.
    assert not math.isnan(forecast.YrExpR)
    assert not math.isnan(forecast.YrExpVol)
    assert not math.isnan(forecast.YrExpRF)
    assert not math.isnan(forecast.YrExpShrp)


def test_realized_performance_hand_computed():
    r = [0.01, -0.02, 0.03, 0.005, -0.01, 0.02]  # B=6 periods
    p = 52
    b = len(r)

    realized = compute_realized_performance(r, p)

    expected_bcktest_real_r = 1.0
    for period_return in r:
        expected_bcktest_real_r *= 1 + period_return
    expected_bcktest_real_r -= 1.0

    assert realized.bkTstRealR == pytest.approx(expected_bcktest_real_r)
    assert realized.YrRealR == pytest.approx((1 + expected_bcktest_real_r) ** (p / b) - 1)

    import statistics

    stdev = statistics.stdev(r)
    assert realized.bkTstRealVol == pytest.approx(stdev * math.sqrt(b))
    assert realized.YrRealVol == pytest.approx(stdev * math.sqrt(p))


def test_nan_realized_performance_constant_is_all_nan():
    assert math.isnan(NAN_REALIZED_PERFORMANCE.YrRealR)
    assert math.isnan(NAN_REALIZED_PERFORMANCE.bkTstRealR)
    assert math.isnan(NAN_REALIZED_PERFORMANCE.YrRealVol)
    assert math.isnan(NAN_REALIZED_PERFORMANCE.bkTstRealVol)


def test_realized_rf_hand_computed_matches_realized_performance_treatment():
    rf = [0.0001, 0.0002, 0.00015, 0.0001, 0.00018, 0.0002]  # B=6 periods
    p = 52
    b = len(rf)

    expected_bcktest_real_rf = 1.0
    for period_rf in rf:
        expected_bcktest_real_rf *= 1 + period_rf
    expected_bcktest_real_rf -= 1.0
    expected_yr_real_rf = (1 + expected_bcktest_real_rf) ** (p / b) - 1

    assert compute_realized_rf(rf, p) == pytest.approx(expected_yr_real_rf)


def test_realized_sharpe_is_excess_realized_return_over_realized_vol():
    assert compute_realized_sharpe(yr_real_r=0.12, yr_real_rf=0.02, yr_real_vol=0.20) == pytest.approx(0.5)


def test_residual_del_exp_real_is_expected_minus_realized():
    assert residual_del_exp_real(0.10, 0.07) == pytest.approx(0.03)


def test_residual_del_exp_real_nan_realized_propagates_to_nan():
    assert math.isnan(residual_del_exp_real(0.10, float("nan")))


def test_residual_prem_real_mkt_is_portfolio_minus_market():
    assert residual_prem_real_mkt(0.12, 0.09) == pytest.approx(0.03)


def test_residual_prem_real_mkt_nan_market_propagates_to_nan():
    assert math.isnan(residual_prem_real_mkt(0.12, float("nan")))
