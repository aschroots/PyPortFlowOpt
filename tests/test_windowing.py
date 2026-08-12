from __future__ import annotations

import pandas as pd
import pytest

from pyportflowopt.errors import InsufficientHistoryError
from pyportflowopt.windowing import (
    Window,
    build_window_plan,
    check_window_coverage,
    generate_backtest_windows,
    generate_final_window,
    generate_shift_only_windows,
    resolve_window_dates,
    window_row_dates,
)


def test_verified_backtest_windowing_example():
    """Literal assertion of the plan's verified example: nPerfRows=40, perfWndw=10,
    bckTestWndw=6, perfWndwShft=10 -> nBacktests=4; window 3 = perf rows 21-30 /
    bcktest rows 31-36; window 4 (anchored) = perf rows 25-34 / bcktest rows 35-40."""
    windows, n_backtests, is_anchored = generate_backtest_windows(
        n_perf_rows=40, perf_wndw=10, perf_wndw_shft=10, bck_test_wndw=6
    )

    assert n_backtests == 4
    assert len(windows) == 4
    assert is_anchored is True

    w1, w2, w3, w4 = windows

    assert w1.perf_start_row == 1
    assert w1.perf_end_row == 10
    assert w1.bcktest_start_row == 11
    assert w1.bcktest_end_row == 16
    assert w1.is_anchored is False

    assert w2.perf_start_row == 11
    assert w2.perf_end_row == 20
    assert w2.bcktest_start_row == 21
    assert w2.bcktest_end_row == 26
    assert w2.is_anchored is False

    assert w3.perf_start_row == 21
    assert w3.perf_end_row == 30
    assert w3.bcktest_start_row == 31
    assert w3.bcktest_end_row == 36
    assert w3.is_anchored is False

    assert w4.perf_start_row == 25
    assert w4.perf_end_row == 34
    assert w4.bcktest_start_row == 35
    assert w4.bcktest_end_row == 40
    assert w4.is_anchored is True


def test_backtest_windowing_no_anchoring_needed_when_evenly_divisible():
    # nPerfRows - perfWndw - bckTestWndw = 40 - 10 - 10 = 20, evenly divisible by shft=10.
    windows, n_backtests, is_anchored = generate_backtest_windows(
        n_perf_rows=40, perf_wndw=10, perf_wndw_shft=10, bck_test_wndw=10
    )
    assert is_anchored is False
    assert n_backtests == 3
    assert len(windows) == 3
    assert windows[-1].bcktest_end_row == 40


def test_backtest_windowing_insufficient_history_is_fatal():
    with pytest.raises(InsufficientHistoryError):
        generate_backtest_windows(n_perf_rows=10, perf_wndw=10, perf_wndw_shft=5, bck_test_wndw=6)


def test_backtest_windowing_exact_zero_remaining_is_allowed():
    # nPerfRows - perfWndw - bckTestWndw == 0 is the boundary (>= 0, not fatal).
    windows, n_backtests, is_anchored = generate_backtest_windows(
        n_perf_rows=16, perf_wndw=10, perf_wndw_shft=10, bck_test_wndw=6
    )
    assert n_backtests == 1
    assert is_anchored is False
    assert windows[0].bcktest_end_row == 16


def test_shift_only_windows_no_bcktest_columns():
    windows = generate_shift_only_windows(n_perf_rows=25, perf_wndw=10, perf_wndw_shft=10)
    assert len(windows) == 2
    assert windows[0].perf_start_row == 1
    assert windows[0].perf_end_row == 10
    assert windows[0].bcktest_start_row is None
    assert windows[0].bcktest_end_row is None
    assert windows[0].has_bcktest is False
    assert windows[1].perf_start_row == 11
    assert windows[1].perf_end_row == 20


def test_shift_only_windows_insufficient_history_is_fatal():
    with pytest.raises(InsufficientHistoryError):
        generate_shift_only_windows(n_perf_rows=5, perf_wndw=10, perf_wndw_shft=5)


def test_final_window_always_uses_the_most_recent_rows():
    window = generate_final_window(n_perf_rows=40, perf_wndw=10)
    assert window.perf_wndw_num == "Final"
    assert window.perf_start_row == 31
    assert window.perf_end_row == 40
    assert window.bcktest_start_row is None
    assert window.has_bcktest is False


def test_build_window_plan_do_bcktest_true_appends_final_and_tracks_n_backtests():
    plan = build_window_plan(
        n_perf_rows=40, perf_wndw=10, perf_wndw_shft=10, do_bcktest=True, bck_test_wndw=6
    )
    assert plan.n_backtests == 4
    assert plan.n_perf_windows is None
    assert plan.is_anchored is True
    assert len(plan.windows) == 5  # 4 backtest windows + Final
    assert plan.windows[-1].perf_wndw_num == "Final"


def test_build_window_plan_do_bcktest_false_uses_shift_only_windows():
    plan = build_window_plan(
        n_perf_rows=25, perf_wndw=10, perf_wndw_shft=10, do_bcktest=False, bck_test_wndw=None
    )
    assert plan.n_backtests is None
    assert plan.n_perf_windows == 2
    assert len(plan.windows) == 3  # 2 shift-only windows + Final
    assert plan.windows[-1].perf_wndw_num == "Final"


def test_resolve_window_dates_maps_rows_to_timestamps():
    dates = pd.date_range("2020-01-03", periods=40, freq="W-FRI")
    window = Window(
        perf_wndw_num=1,
        perf_start_row=1,
        perf_end_row=10,
        bcktest_start_row=11,
        bcktest_end_row=16,
        is_anchored=False,
    )
    resolved = resolve_window_dates(window, dates)
    assert resolved.perf_start == dates[0]
    assert resolved.perf_end == dates[9]
    assert resolved.bcktest_start == dates[10]
    assert resolved.bcktest_end == dates[15]


def test_resolve_window_dates_none_bcktest_when_no_bcktest():
    dates = pd.date_range("2020-01-03", periods=10, freq="W-FRI")
    window = Window(
        perf_wndw_num="Final",
        perf_start_row=1,
        perf_end_row=10,
        bcktest_start_row=None,
        bcktest_end_row=None,
        is_anchored=False,
    )
    resolved = resolve_window_dates(window, dates)
    assert resolved.bcktest_start is None
    assert resolved.bcktest_end is None


def test_window_row_dates_is_inclusive_1_indexed_slice():
    dates = pd.date_range("2020-01-03", periods=10, freq="W-FRI")
    sliced = window_row_dates(dates, 2, 4)
    assert list(sliced) == list(dates[1:4])


def test_check_window_coverage_full_coverage_true():
    dates = pd.date_range("2020-01-03", periods=4, freq="W-FRI")
    returns = pd.DataFrame({"AAA": [0.01, 0.02, 0.03, 0.04]}, index=dates)
    coverage = check_window_coverage(returns, dates)
    assert coverage == {"AAA": True}


def test_check_window_coverage_mid_window_gap_excludes_ticker_from_just_that_window():
    dates = pd.date_range("2020-01-03", periods=4, freq="W-FRI")
    returns = pd.DataFrame({"AAA": [0.01, 0.02, 0.03, 0.04], "BBB": [0.01, None, 0.03, 0.04]}, index=dates)
    full_window_coverage = check_window_coverage(returns, dates)
    assert full_window_coverage == {"AAA": True, "BBB": False}

    # But BBB IS fully covered over a different, narrower window that skips the gap.
    narrow_window_coverage = check_window_coverage(returns, dates[[0, 2, 3]])
    assert narrow_window_coverage["BBB"] is True
