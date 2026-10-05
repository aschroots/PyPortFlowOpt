from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from pyportflowopt.config import PyPortFlowOptConfig
from pyportflowopt.results import VALUE_FIELDS, OptimizationRecord
from pyportflowopt.run_summary import RunSummary
from pyportflowopt.windowing import Window, WindowPlan

_CATALOG_ORDER = ("CAPM", "FF3", "FFC4", "FF5", "FF5Mod", "FFC6")


def _config(**overrides) -> PyPortFlowOptConfig:
    base = dict(
        marketProxyTicker="SPY",
        marketProxyName="S&P 500",
        priceInputKind="prices",
        perfWndw=10,
        perfWndwShft=10,
        doBcktest=True,
        bckTestWndw=6,
        shortPortfolio=False,
        shortLimit=None,
        factorModels=("CAPM",),
        wtsEpsilon=0.0,
    )
    base.update(overrides)
    return PyPortFlowOptConfig(**base)


def _record(perf_wndw_num, model, portfolio_type="Lng", **overrides) -> OptimizationRecord:
    values = dict.fromkeys(VALUE_FIELDS, float("nan"))
    values.update(overrides)
    return OptimizationRecord(
        perfWndwNum=perf_wndw_num,
        PerfStart=dt.date(2020, 1, 1),
        PerfEnd=dt.date(2020, 3, 1),
        bkTstStart=dt.date(2020, 3, 2),
        bkTstEnd=dt.date(2020, 4, 1),
        Model=model,
        PortfolioType=portfolio_type,
        **values,
    )


def _render(config, records, **kwargs):
    summary = RunSummary()
    window_plan = kwargs.pop(
        "window_plan",
        WindowPlan(
            windows=(Window(1, 1, 10, 11, 16, False),),
            n_backtests=1,
            n_perf_windows=None,
            is_anchored=False,
        ),
    )
    return summary.render(
        config,
        Path("config.yaml"),
        window_plan,
        Path("securities.csv"),
        52,
        4,
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-01")),
        40,
        Path("factors.csv"),
        52,
        {"Mkt-RF": (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-01"))},
        records,
        _CATALOG_ORDER,
        Path("out"),
        **kwargs,
    )


def test_title_and_self_computed_underline():
    rendered = _render(_config(), [])
    lines = rendered.splitlines()
    assert lines[0] == "PyPortFlowOpt Run Summary"
    assert lines[1] == "=" * len("PyPortFlowOpt Run Summary")
    assert len(lines[1]) == len(lines[0])


def test_section_headers_have_no_trailing_colon():
    lines = _render(_config(), [_record(1, "Simp", YrExpShrp=1.0)]).splitlines()
    for header in ("Configuration", "Input Securities Summary", "Input Factors Summary", "Output Summary"):
        assert header in lines
        assert f"{header}:" not in lines


def test_market_proxy_none_renders_as_none_literal():
    config = _config(marketProxyTicker=None, marketProxyName=None)
    rendered = _render(config, [])
    assert "marketProxyTicker = (none)" in rendered
    assert "marketProxyName = (none)" in rendered


def test_market_proxy_configured_renders_actual_values():
    rendered = _render(_config(), [])
    assert "marketProxyTicker = SPY" in rendered
    assert "marketProxyName = S&P 500" in rendered


def test_do_bcktest_true_renders_n_backtests_and_output_summary():
    window_plan = WindowPlan(
        windows=(Window(1, 1, 10, 11, 16, False),), n_backtests=4, n_perf_windows=None, is_anchored=True
    )
    rendered = _render(
        _config(),
        [_record(1, "Simp", YrExpShrp=1.5, YrDelExpReal=0.02)],
        window_plan=window_plan,
    )
    assert "nBacktests = 4" in rendered
    assert "anchored" in rendered.lower()
    lines = rendered.splitlines()
    i = lines.index("Output Summary")
    assert lines[i + 1] == f"  Outputs Location: {Path('out').resolve()}"
    assert lines[i + 2].startswith("  Simp Long: max YrExpShrp")


def test_do_bcktest_false_renders_n_perf_windows_and_location_only_output_summary():
    config = _config(doBcktest=False, bckTestWndw=None, marketProxyTicker=None, marketProxyName=None)
    window_plan = WindowPlan(
        windows=(Window(1, 1, 10, None, None, False),), n_backtests=None, n_perf_windows=3, is_anchored=False
    )
    rendered = _render(config, [_record(1, "Simp", YrExpShrp=1.5)], window_plan=window_plan)
    assert "nPerfWindows = 3" in rendered
    assert "nBacktests" not in rendered
    # Output Summary still appears (for the output location) but has no backtest stats.
    lines = rendered.splitlines()
    i = lines.index("Output Summary")
    assert lines[i + 1 :] == [f"  Outputs Location: {Path('out').resolve()}"]
    assert "max YrExpShrp" not in rendered


def test_output_summary_omits_market_proxy_stats_when_unconfigured():
    config = _config(marketProxyTicker=None, marketProxyName=None)
    rendered = _render(
        config,
        [_record(1, "Simp", YrExpShrp=1.5, YrDelExpReal=0.02, YrPremRealMkt=float("nan"))],
    )
    assert "YrDelExpReal mean" in rendered
    assert "YrPremRealMkt" not in rendered


def test_notes_and_flags_sections_render_accumulated_messages():
    summary = RunSummary()
    summary.note("a note")
    summary.flag("a flag")
    window_plan = WindowPlan(windows=(), n_backtests=1, n_perf_windows=None, is_anchored=False)
    rendered = summary.render(
        _config(),
        Path("config.yaml"),
        window_plan,
        Path("securities.csv"),
        52,
        4,
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-01")),
        40,
        Path("factors.csv"),
        52,
        {},
        [],
        _CATALOG_ORDER,
        Path("out"),
    )
    assert "Notes" in rendered
    assert "  - a note" in rendered
    assert "Flags" in rendered
    assert "  - a flag" in rendered
