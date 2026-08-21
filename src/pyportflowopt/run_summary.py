from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from .config import PyPortFlowOptConfig, market_proxy_display_name
from .results import OptimizationRecord
from .windowing import WindowPlan

_TITLE = "PyPortFlowOpt Run Summary"

_CONFIG_FIELDS_TO_RENDER = (
    "marketProxyTicker",
    "marketProxyName",
    "priceInputKind",
    "perfWndw",
    "perfWndwShft",
    "doBcktest",
    "bckTestWndw",
    "shortPortfolio",
    "shortLimit",
    "factorModels",
    "wtsEpsilon",
)


def _render_config_field(config: PyPortFlowOptConfig, field_name: str) -> str:
    value = getattr(config, field_name)
    if field_name in ("marketProxyTicker", "marketProxyName"):
        return str(value) if value is not None else "(none)"
    if isinstance(value, tuple):
        return ", ".join(value) if value else "(none)"
    return str(value)


class RunSummary:
    """Accumulates informational notes and validation flags for the timestamped
    run_summary_YYYYMMDD_HHMMSS.log. Ported from PyPortFlow2's logging_utils.py:RunSummary
    with two formatting fixes: the title underline is self-computed ("=" * len(title))
    rather than a hardcoded length, and every section header uses a consistent trailing
    colon (PyPortFlow2's own two summary sections are inconsistent with each other on
    this point)."""

    def __init__(self) -> None:
        self._notes: list[str] = []
        self._flags: list[str] = []

    def note(self, message: str) -> None:
        self._notes.append(message)

    def flag(self, message: str) -> None:
        self._flags.append(message)

    def render(
        self,
        config: PyPortFlowOptConfig,
        config_path: Path,
        window_plan: WindowPlan,
        securities_path: Path,
        periods_per_year: int,
        ticker_count: int,
        securities_date_range: tuple[pd.Timestamp, pd.Timestamp],
        n_perf_rows: int,
        factors_path: Path,
        factor_periods_per_year: int,
        factor_date_ranges: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
        records: Sequence[OptimizationRecord],
        catalog_order: Sequence[str],
    ) -> str:
        lines = [_TITLE, "=" * len(_TITLE), ""]

        lines.append("Configuration:")
        lines.append(f"  Input File: {config_path.name}")
        for field_name in _CONFIG_FIELDS_TO_RENDER:
            lines.append(f"  {field_name} = {_render_config_field(config, field_name)}")
        if config.doBcktest:
            lines.append(f"  nBacktests = {window_plan.n_backtests}")
            if window_plan.is_anchored:
                lines.append(
                    "  Note: the final backtest window is anchored to the end of history and "
                    "does not follow perfWndwShft."
                )
        else:
            lines.append(f"  nPerfWindows = {window_plan.n_perf_windows}")
        lines.append("")

        lines.append("Input Securities Summary:")
        lines.append(f"  Input File: {securities_path.name}")
        lines.append(f"  Detected periods per year: {periods_per_year}")
        lines.append(f"  tickerCount = {ticker_count}")
        lines.append(f"  priceInputKind = {config.priceInputKind}")
        lines.append(f"  dateRange = {securities_date_range[0].date()} to {securities_date_range[1].date()}")
        lines.append(f"  nPerfRows = {n_perf_rows}")
        lines.append("")

        lines.append("Input Factors Summary:")
        lines.append(f"  Input File: {factors_path.name}")
        lines.append(f"  Detected periods per year: {factor_periods_per_year}")
        for name, (oldest, newest) in factor_date_ranges.items():
            lines.append(f"  {name}: {oldest.date()} to {newest.date()}")
        lines.append("")

        if config.doBcktest:
            output_summary_lines = _render_output_summary(config, records, catalog_order)
            if output_summary_lines:
                lines.append("Output Summary:")
                lines.extend(output_summary_lines)
                lines.append("")

        if self._notes:
            lines.append("Notes:")
            lines.extend(f"  - {n}" for n in self._notes)
            lines.append("")

        if self._flags:
            lines.append("Flags:")
            lines.extend(f"  - {f}" for f in self._flags)
            lines.append("")

        return "\n".join(lines)


def _render_output_summary(
    config: PyPortFlowOptConfig, records: Sequence[OptimizationRecord], catalog_order: Sequence[str]
) -> list[str]:
    lines: list[str] = []
    present_models = [name for name in ("Simp", *catalog_order) if any(r.Model == name for r in records)]
    market_proxy_configured = market_proxy_display_name(config) is not None

    for model in present_models:
        model_records = [r for r in records if r.Model == model]

        for portfolio_type, label in (("Lng", "Long"), ("Shrt", "Short")):
            typed_records = [
                r for r in model_records if r.PortfolioType == portfolio_type and not math.isnan(r.YrExpShrp)
            ]
            if not typed_records:
                continue
            best = max(typed_records, key=lambda r: r.YrExpShrp)
            lines.append(
                f"  {model} {label}: max YrExpShrp = {best.YrExpShrp:.4f} "
                f"(window {best.perfWndwNum}, PerfStart={best.PerfStart} PerfEnd={best.PerfEnd}, "
                f"bkTstStart={best.bkTstStart} bkTstEnd={best.bkTstEnd})"
            )

        del_exp_vals = [r.YrDelExpReal for r in model_records if not math.isnan(r.YrDelExpReal)]
        if del_exp_vals:
            mean = statistics.mean(del_exp_vals)
            stdev = statistics.stdev(del_exp_vals) if len(del_exp_vals) > 1 else float("nan")
            lines.append(f"  {model}: YrDelExpReal mean = {mean:.6f}, stdev = {stdev:.6f}")

        if market_proxy_configured:
            prem_vals = [r.YrPremRealMkt for r in model_records if not math.isnan(r.YrPremRealMkt)]
            if prem_vals:
                mean = statistics.mean(prem_vals)
                stdev = statistics.stdev(prem_vals) if len(prem_vals) > 1 else float("nan")
                lines.append(f"  {model}: YrPremRealMkt mean = {mean:.6f}, stdev = {stdev:.6f}")

    return lines


def write_run_summary(path: Path, rendered: str) -> None:
    path.write_text(rendered, encoding="utf-8")
