from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from .errors import InsufficientHistoryError

FINAL_WINDOW_ID = "Final"


@dataclass(frozen=True)
class Window:
    """All row numbers are 1-indexed, ascending, row 1 = oldest."""

    perf_wndw_num: int | str  # 1-based sequential int, or the literal "Final"
    perf_start_row: int
    perf_end_row: int
    bcktest_start_row: int | None
    bcktest_end_row: int | None
    is_anchored: bool

    @property
    def has_bcktest(self) -> bool:
        return self.bcktest_start_row is not None


@dataclass(frozen=True)
class WindowDates:
    perf_start: pd.Timestamp
    perf_end: pd.Timestamp
    bcktest_start: pd.Timestamp | None
    bcktest_end: pd.Timestamp | None


@dataclass(frozen=True)
class WindowPlan:
    windows: tuple[Window, ...]  # regular/anchored (or shift-only) windows, then the Final window
    n_backtests: int | None  # set when doBcktest: true
    n_perf_windows: int | None  # set when doBcktest: false
    is_anchored: bool  # whether the last backtest window skipped perfWndwShft to reach EOF


def generate_backtest_windows(
    n_perf_rows: int, perf_wndw: int, perf_wndw_shft: int, bck_test_wndw: int
) -> tuple[list[Window], int, bool]:
    remaining = n_perf_rows - perf_wndw - bck_test_wndw
    if remaining < 0:
        raise InsufficientHistoryError(
            f"Not enough history for backtest windows: nPerfRows={n_perf_rows}, "
            f"perfWndw={perf_wndw}, bckTestWndw={bck_test_wndw}."
        )
    n_backtests = math.ceil(remaining / perf_wndw_shft + 1)

    windows: list[Window] = []
    i = 0
    while True:
        perf_start = 1 + i * perf_wndw_shft
        perf_end = perf_start + perf_wndw - 1
        bck_start = perf_end + 1
        bck_end = bck_start + bck_test_wndw - 1
        if bck_end > n_perf_rows:
            break
        windows.append(
            Window(
                perf_wndw_num=i + 1,
                perf_start_row=perf_start,
                perf_end_row=perf_end,
                bcktest_start_row=bck_start,
                bcktest_end_row=bck_end,
                is_anchored=False,
            )
        )
        i += 1

    is_anchored = remaining % perf_wndw_shft != 0
    if is_anchored:
        bck_end = n_perf_rows
        bck_start = bck_end - bck_test_wndw + 1
        perf_end = bck_start - 1
        perf_start = perf_end - perf_wndw + 1
        windows.append(
            Window(
                perf_wndw_num=len(windows) + 1,
                perf_start_row=perf_start,
                perf_end_row=perf_end,
                bcktest_start_row=bck_start,
                bcktest_end_row=bck_end,
                is_anchored=True,
            )
        )

    return windows, n_backtests, is_anchored


def generate_shift_only_windows(n_perf_rows: int, perf_wndw: int, perf_wndw_shft: int) -> list[Window]:
    if n_perf_rows < perf_wndw:
        raise InsufficientHistoryError(
            f"Not enough history for a single window: nPerfRows={n_perf_rows}, perfWndw={perf_wndw}."
        )
    windows: list[Window] = []
    i = 0
    while True:
        perf_start = 1 + i * perf_wndw_shft
        perf_end = perf_start + perf_wndw - 1
        if perf_end > n_perf_rows:
            break
        windows.append(
            Window(
                perf_wndw_num=i + 1,
                perf_start_row=perf_start,
                perf_end_row=perf_end,
                bcktest_start_row=None,
                bcktest_end_row=None,
                is_anchored=False,
            )
        )
        i += 1
    return windows


def generate_final_window(n_perf_rows: int, perf_wndw: int) -> Window:
    """Always generated regardless of doBcktest -- the live-forecast window."""
    if n_perf_rows < perf_wndw:
        raise InsufficientHistoryError(
            f"Not enough history for the Final window: nPerfRows={n_perf_rows}, perfWndw={perf_wndw}."
        )
    perf_end = n_perf_rows
    perf_start = n_perf_rows - perf_wndw + 1
    return Window(
        perf_wndw_num=FINAL_WINDOW_ID,
        perf_start_row=perf_start,
        perf_end_row=perf_end,
        bcktest_start_row=None,
        bcktest_end_row=None,
        is_anchored=False,
    )


def build_window_plan(
    n_perf_rows: int,
    perf_wndw: int,
    perf_wndw_shft: int,
    do_bcktest: bool,
    bck_test_wndw: int | None,
) -> WindowPlan:
    final_window = generate_final_window(n_perf_rows, perf_wndw)
    if do_bcktest:
        assert bck_test_wndw is not None
        windows, n_backtests, is_anchored = generate_backtest_windows(
            n_perf_rows, perf_wndw, perf_wndw_shft, bck_test_wndw
        )
        return WindowPlan(
            windows=(*windows, final_window),
            n_backtests=n_backtests,
            n_perf_windows=None,
            is_anchored=is_anchored,
        )

    windows = generate_shift_only_windows(n_perf_rows, perf_wndw, perf_wndw_shft)
    return WindowPlan(
        windows=(*windows, final_window),
        n_backtests=None,
        n_perf_windows=len(windows),
        is_anchored=False,
    )


def resolve_window_dates(window: Window, dates: pd.DatetimeIndex) -> WindowDates:
    """Maps 1-indexed rows to pd.Timestamps -- PerfStart/PerfEnd/bckTestStart/bckTestEnd
    in every output are dates, not row numbers, so they're self-describing without
    cross-referencing the source file's row numbering."""

    def _at(row: int) -> pd.Timestamp:
        return dates[row - 1]

    return WindowDates(
        perf_start=_at(window.perf_start_row),
        perf_end=_at(window.perf_end_row),
        bcktest_start=_at(window.bcktest_start_row) if window.bcktest_start_row is not None else None,
        bcktest_end=_at(window.bcktest_end_row) if window.bcktest_end_row is not None else None,
    )


def window_row_dates(dates: pd.DatetimeIndex, start_row: int, end_row: int) -> pd.DatetimeIndex:
    """1-indexed, inclusive row-range slice of an ascending DatetimeIndex."""
    return dates[start_row - 1 : end_row]


def check_window_coverage(returns: pd.DataFrame, window_dates: pd.DatetimeIndex) -> dict[str, bool]:
    """Whether each ticker has a non-null value for every row spanned by window_dates.

    Ported verbatim from PyPortFlow2's stats.py:check_window_coverage.
    """
    window_returns = returns.loc[returns.index.isin(window_dates)]
    return {
        ticker: bool(window_returns[ticker].notna().all()) and len(window_returns) == len(window_dates)
        for ticker in returns.columns
    }
