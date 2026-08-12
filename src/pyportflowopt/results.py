from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

ID_FIELDS: tuple[str, ...] = ("perfWndwNum", "PerfStart", "PerfEnd", "bckTestStart", "bckTestEnd")

VALUE_FIELDS: tuple[str, ...] = (
    "YrExpR",
    "bckTestExpR",
    "YrExpVol",
    "bckTestExpVol",
    "YrExpShrp",
    "YrRealR",
    "bckTestRealR",
    "YrRealVol",
    "bckTestRealVol",
    "YrDelForeReal",
    "bckTestDelForeReal",
    "YrPremRealMkt",
    "bckTestPremRealMkt",
)


@dataclass(frozen=True)
class OptimizationRecord:
    """One tidy record per (window, model, portfolio_type). Field names for the 13 value
    fields match VALUE_FIELDS' exact order, which is also the CSV output order -- using
    the literal output-column spelling as the field name (rather than snake_case) avoids
    a separate name-translation table for what is otherwise the single highest-risk area
    for a naming transcription bug in this whole port."""

    perfWndwNum: int | str
    PerfStart: dt.date
    PerfEnd: dt.date
    bckTestStart: dt.date | None
    bckTestEnd: dt.date | None
    Model: str
    PortfolioType: str  # "Lng" | "Shrt"
    YrExpR: float
    bckTestExpR: float
    YrExpVol: float
    bckTestExpVol: float
    YrExpShrp: float
    YrRealR: float
    bckTestRealR: float
    YrRealVol: float
    bckTestRealVol: float
    YrDelForeReal: float
    bckTestDelForeReal: float
    YrPremRealMkt: float
    bckTestPremRealMkt: float


def _model_order(records: Sequence[OptimizationRecord], catalog_order: Sequence[str]) -> list[str]:
    present = {r.Model for r in records}
    return [name for name in ("Simp", *catalog_order) if name in present]


def build_results_tables(
    records: Sequence[OptimizationRecord], portfolio_type: str, catalog_order: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (pmod_pval_df, pval_pmod_df) for the given portfolio type ('Lng'/'Shrt').

    Computes one wide table (id columns + every "{Model}{ValueField}" column, in
    whatever order first encountered) and derives BOTH output orderings as pure column
    selections of it -- same column names in both files, just reordered (model-outer vs.
    value-outer), never two independent pivots.
    """
    filtered = [r for r in records if r.PortfolioType == portfolio_type]
    models = _model_order(filtered, catalog_order)

    id_by_window: dict[int | str, dict[str, object]] = {}
    values_by_window: dict[int | str, dict[str, float]] = {}
    for r in filtered:
        key = r.perfWndwNum
        id_by_window.setdefault(key, {f: getattr(r, f) for f in ID_FIELDS})
        row = values_by_window.setdefault(key, {})
        for value_field in VALUE_FIELDS:
            row[f"{r.Model}{value_field}"] = getattr(r, value_field)

    base_rows = []
    for key, id_cols in id_by_window.items():
        row: dict[str, object] = dict(id_cols)
        row.update(values_by_window[key])
        base_rows.append(row)
    base_df = pd.DataFrame(base_rows)

    pmod_pval_cols = [*ID_FIELDS, *(f"{m}{vf}" for m in models for vf in VALUE_FIELDS)]
    pval_pmod_cols = [*ID_FIELDS, *(f"{m}{vf}" for vf in VALUE_FIELDS for m in models)]

    pmod_pval_df = base_df.reindex(columns=pmod_pval_cols)
    pval_pmod_df = base_df.reindex(columns=pval_pmod_cols)
    return pmod_pval_df, pval_pmod_df


def build_sharpe_history_table(
    records: Sequence[OptimizationRecord], catalog_order: Sequence[str]
) -> pd.DataFrame:
    """id columns, then <model>LngShrp per model (catalog order), then <model>ShrtShrp per
    model if any Shrt records are present -- annualized YrExpShrp only, no bckTestWndw-
    scale Sharpe is tracked."""
    lng_records = [r for r in records if r.PortfolioType == "Lng"]
    shrt_records = [r for r in records if r.PortfolioType == "Shrt"]
    models_lng = _model_order(lng_records, catalog_order)
    models_shrt = _model_order(shrt_records, catalog_order)

    id_by_window: dict[int | str, dict[str, object]] = {}
    lng_shrp: dict[int | str, dict[str, float]] = {}
    shrt_shrp: dict[int | str, dict[str, float]] = {}
    for r in lng_records:
        id_by_window.setdefault(r.perfWndwNum, {f: getattr(r, f) for f in ID_FIELDS})
        lng_shrp.setdefault(r.perfWndwNum, {})[f"{r.Model}LngShrp"] = r.YrExpShrp
    for r in shrt_records:
        id_by_window.setdefault(r.perfWndwNum, {f: getattr(r, f) for f in ID_FIELDS})
        shrt_shrp.setdefault(r.perfWndwNum, {})[f"{r.Model}ShrtShrp"] = r.YrExpShrp

    rows = []
    for key, id_cols in id_by_window.items():
        row: dict[str, object] = dict(id_cols)
        for m in models_lng:
            row[f"{m}LngShrp"] = lng_shrp.get(key, {}).get(f"{m}LngShrp", float("nan"))
        for m in models_shrt:
            row[f"{m}ShrtShrp"] = shrt_shrp.get(key, {}).get(f"{m}ShrtShrp", float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)
