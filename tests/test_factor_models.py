from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyportflowopt.errors import FactorAlignmentError, FatalPipelineError
from pyportflowopt.factor_models import (
    align_factor_window,
    build_design_matrix,
    compute_common_start_long_run_factor_expected_returns,
    compute_expected_return,
    compute_factor_date_ranges,
    compute_joint_factor_regression,
    compute_long_run_factor_expected_returns,
    latest_value_in_window,
    realign_monthly_factor_dates_to_price_dates,
    run_ols_point_estimates,
)


def test_build_design_matrix_prepends_intercept_column():
    factor_returns = pd.DataFrame({"Mkt-RF": [0.01, 0.02, -0.01]})
    x = build_design_matrix(factor_returns, ("Mkt-RF",))
    assert x.shape == (3, 2)
    assert (x[:, 0] == 1.0).all()
    np.testing.assert_allclose(x[:, 1], [0.01, 0.02, -0.01])


def test_build_design_matrix_missing_factor_column_is_fatal():
    factor_returns = pd.DataFrame({"Mkt-RF": [0.01, 0.02]})
    with pytest.raises(FatalPipelineError):
        build_design_matrix(factor_returns, ("Mkt-RF", "SMB"))


def test_run_ols_point_estimates_alpha_is_period_native_not_annualized():
    # y = 0.002 + 1.5 * x, no noise -- OLS should recover alpha=0.002 exactly regardless
    # of periods_per_year, since PyPortFlowOpt no longer multiplies alpha by it (unlike
    # PyPortFlow2's alpha_annualized).
    rng = np.random.default_rng(0)
    x_factor = rng.normal(0, 0.01, 50)
    y = 0.002 + 1.5 * x_factor
    x = np.column_stack([np.ones(50), x_factor])

    result_p1 = run_ols_point_estimates(y, x, ("Mkt-RF",), periods_per_year=1)
    result_p252 = run_ols_point_estimates(y, x, ("Mkt-RF",), periods_per_year=252)

    assert result_p1.alpha == pytest.approx(0.002, abs=1e-9)
    assert result_p252.alpha == pytest.approx(0.002, abs=1e-9)
    assert result_p1.alpha == pytest.approx(result_p252.alpha)  # periods_per_year has zero effect
    assert result_p1.betas["Mkt-RF"] == pytest.approx(1.5, abs=1e-9)


def test_compute_joint_factor_regression_capm_degenerate_case():
    dates = pd.date_range("2020-01-03", periods=5, freq="W-FRI")
    factor_returns = pd.DataFrame({"Mkt-RF": [0.01, 0.02, -0.01, 0.03, 0.00]}, index=dates)
    excess = pd.Series([0.02, 0.04, -0.02, 0.06, 0.00], index=dates)  # exactly 2x Mkt-RF, alpha=0
    result = compute_joint_factor_regression(excess, factor_returns, ("Mkt-RF",), periods_per_year=52)
    assert result.betas["Mkt-RF"] == pytest.approx(2.0, abs=1e-6)
    assert result.alpha == pytest.approx(0.0, abs=1e-6)


def test_realign_monthly_factor_dates_matches_by_year_month():
    factor_dates = pd.DatetimeIndex(["2021-01-31", "2021-02-28", "2021-03-31"])
    factor_values = pd.DataFrame({"Mkt-RF": [0.01, 0.02, 0.03]}, index=factor_dates)
    target_dates = pd.DatetimeIndex(["2021-01-29", "2021-02-26"])  # trading-day month-ends
    aligned = realign_monthly_factor_dates_to_price_dates(factor_values, target_dates)
    assert list(aligned.index) == list(target_dates)
    assert aligned["Mkt-RF"].iloc[0] == pytest.approx(0.01)
    assert aligned["Mkt-RF"].iloc[1] == pytest.approx(0.02)


def test_realign_monthly_factor_dates_missing_month_is_nan_row():
    factor_dates = pd.DatetimeIndex(["2021-01-31", "2021-03-31"])  # no February row
    factor_values = pd.DataFrame({"Mkt-RF": [0.01, 0.03]}, index=factor_dates)
    target_dates = pd.DatetimeIndex(["2021-01-29", "2021-02-26", "2021-03-30"])
    aligned = realign_monthly_factor_dates_to_price_dates(factor_values, target_dates)
    assert pd.isna(aligned["Mkt-RF"].iloc[1])


def test_align_factor_window_monthly_uses_realign(monkeypatch=None):
    factor_dates = pd.DatetimeIndex(["2021-01-31", "2021-02-28"])
    factor_values = pd.DataFrame({"Mkt-RF": [0.01, 0.02], "RF": [0.001, 0.001]}, index=factor_dates)
    window_dates = pd.DatetimeIndex(["2021-01-29", "2021-02-26"])
    aligned = align_factor_window(factor_values, window_dates, periods_per_year=12)
    assert list(aligned.index) == list(window_dates)
    assert aligned["Mkt-RF"].iloc[1] == pytest.approx(0.02)


def test_align_factor_window_monthly_wholly_missing_month_is_fatal():
    factor_dates = pd.DatetimeIndex(["2021-01-31"])
    factor_values = pd.DataFrame({"Mkt-RF": [0.01], "RF": [0.001]}, index=factor_dates)
    window_dates = pd.DatetimeIndex(["2021-01-29", "2021-02-26"])
    with pytest.raises(FactorAlignmentError):
        align_factor_window(factor_values, window_dates, periods_per_year=12)


def test_align_factor_window_non_monthly_requires_exact_date_match():
    factor_dates = pd.DatetimeIndex(["2021-01-01", "2021-01-08", "2021-01-15"])
    factor_values = pd.DataFrame({"Mkt-RF": [0.01, 0.02, 0.03], "RF": [0.0] * 3}, index=factor_dates)
    window_dates = pd.DatetimeIndex(["2021-01-01", "2021-01-08"])
    aligned = align_factor_window(factor_values, window_dates, periods_per_year=52)
    assert list(aligned.index) == list(window_dates)


def test_align_factor_window_non_monthly_missing_exact_date_is_fatal():
    factor_dates = pd.DatetimeIndex(["2021-01-01", "2021-01-08"])
    factor_values = pd.DataFrame({"Mkt-RF": [0.01, 0.02], "RF": [0.0, 0.0]}, index=factor_dates)
    window_dates = pd.DatetimeIndex(["2021-01-01", "2021-01-15"])  # 01-15 is wholly missing
    with pytest.raises(FactorAlignmentError):
        align_factor_window(factor_values, window_dates, periods_per_year=52)


def test_align_factor_window_tolerates_partial_column_nan():
    factor_dates = pd.DatetimeIndex(["2021-01-01", "2021-01-08"])
    factor_values = pd.DataFrame(
        {"Mkt-RF": [0.01, 0.02], "SMB5Fac": [np.nan, 0.03], "RF": [0.0, 0.0]}, index=factor_dates
    )
    window_dates = pd.DatetimeIndex(["2021-01-01", "2021-01-08"])
    # Not wholly missing (Mkt-RF/RF present for 01-01), so this must NOT raise even though
    # SMB5Fac is partially NaN.
    aligned = align_factor_window(factor_values, window_dates, periods_per_year=52)
    assert pd.isna(aligned["SMB5Fac"].iloc[0])


def test_compute_long_run_factor_expected_returns_is_period_native_mean():
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    factor_values = pd.DataFrame({"Mkt-RF": [0.01, 0.02, 0.03, 0.04]}, index=dates)
    result = compute_long_run_factor_expected_returns(factor_values, as_of_date=dates[2])
    assert result["Mkt-RF"] == pytest.approx((0.01 + 0.02 + 0.03) / 3)


def test_compute_common_start_long_run_uses_most_restrictive_start_date():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    factor_values = pd.DataFrame(
        {
            "Mkt-RF": [0.01, 0.02, 0.03, 0.04, 0.05],
            "SMB": [np.nan, np.nan, 0.10, 0.20, 0.30],  # only starts on day 3
        },
        index=dates,
    )
    result = compute_common_start_long_run_factor_expected_returns(
        factor_values, ("Mkt-RF", "SMB"), as_of_date=dates[4]
    )
    # Common start = day 3 (SMB's first valid date) -> average over days 3,4,5 for BOTH factors.
    assert result["Mkt-RF"] == pytest.approx((0.03 + 0.04 + 0.05) / 3)
    assert result["SMB"] == pytest.approx((0.10 + 0.20 + 0.30) / 3)


def test_compute_common_start_long_run_missing_factor_column_is_fatal():
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    factor_values = pd.DataFrame({"Mkt-RF": [0.01, 0.02, 0.03]}, index=dates)
    with pytest.raises(FatalPipelineError):
        compute_common_start_long_run_factor_expected_returns(
            factor_values, ("Mkt-RF", "SMB"), as_of_date=dates[2]
        )


def test_compute_common_start_long_run_no_data_before_as_of_date_is_fatal():
    dates = pd.date_range("2020-06-01", periods=3, freq="D")
    factor_values = pd.DataFrame({"Mkt-RF": [0.01, 0.02, 0.03]}, index=dates)
    with pytest.raises(FatalPipelineError):
        compute_common_start_long_run_factor_expected_returns(
            factor_values, ("Mkt-RF",), as_of_date=pd.Timestamp("2020-01-01")
        )


def test_latest_value_in_window_returns_most_recent_non_null():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    series = pd.Series([0.01, np.nan, 0.03, np.nan, np.nan], index=dates, name="RF")
    value = latest_value_in_window(series, dates[0], dates[3])
    assert value == pytest.approx(0.03)  # day 3 is the latest non-null within [day0, day3]


def test_latest_value_in_window_bounded_ignores_data_outside_range():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    series = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05], index=dates, name="RF")
    value = latest_value_in_window(series, dates[0], dates[1])
    assert value == pytest.approx(0.02)  # not 0.05 -- bounded look-back, unlike PyPortFlow2's original


def test_latest_value_in_window_empty_is_fatal():
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    series = pd.Series([np.nan, np.nan, np.nan], index=dates, name="RF")
    with pytest.raises(FatalPipelineError):
        latest_value_in_window(series, dates[0], dates[2])


def test_compute_factor_date_ranges_per_column():
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    factor_values = pd.DataFrame(
        {"Mkt-RF": [0.01, 0.02, 0.03, 0.04], "SMB": [np.nan, np.nan, 0.10, 0.20]}, index=dates
    )
    ranges = compute_factor_date_ranges(factor_values)
    assert ranges["Mkt-RF"] == (dates[0], dates[3])
    assert ranges["SMB"] == (dates[2], dates[3])


def test_compute_expected_return_formula():
    rf = 0.001
    betas = {"Mkt-RF": 1.2, "SMB": 0.3}
    long_run = {"Mkt-RF": 0.05, "SMB": 0.02}
    expected = compute_expected_return(rf, betas, long_run)
    assert expected == pytest.approx(0.001 + 1.2 * 0.05 + 0.3 * 0.02)
