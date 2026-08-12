from __future__ import annotations

import datetime as dt

import pytest

from pyportflowopt.results import (
    ID_FIELDS,
    VALUE_FIELDS,
    OptimizationRecord,
    build_results_tables,
    build_sharpe_history_table,
)

_CATALOG_ORDER = ("CAPM", "FF3", "FFC4", "FF5", "FF5Mod", "FFC6")


def _record(perf_wndw_num, model, portfolio_type, base_value=1.0, **overrides) -> OptimizationRecord:
    values = {field: base_value for field in VALUE_FIELDS}
    values.update(overrides)
    return OptimizationRecord(
        perfWndwNum=perf_wndw_num,
        PerfStart=dt.date(2020, 1, 1),
        PerfEnd=dt.date(2020, 3, 1),
        bckTestStart=dt.date(2020, 3, 2),
        bckTestEnd=dt.date(2020, 4, 1),
        Model=model,
        PortfolioType=portfolio_type,
        **values,
    )


def test_value_fields_are_the_13_fields_in_plan_order():
    assert VALUE_FIELDS == (
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
    assert len(VALUE_FIELDS) == 13


def test_id_fields_include_perf_wndw_num_literally_spelled():
    assert ID_FIELDS[0] == "perfWndwNum"
    assert "WindowNum" not in ID_FIELDS


def test_build_results_tables_pmod_pval_and_pval_pmod_have_identical_value_sets():
    records = [
        _record(1, "Simp", "Lng", base_value=1.0),
        _record(1, "CAPM", "Lng", base_value=2.0),
        _record(2, "Simp", "Lng", base_value=3.0),
        _record(2, "CAPM", "Lng", base_value=4.0),
    ]
    pmod_pval, pval_pmod = build_results_tables(records, "Lng", _CATALOG_ORDER)

    assert set(pmod_pval.columns) == set(pval_pmod.columns)
    assert len(pmod_pval) == len(pval_pmod) == 2
    for col in pmod_pval.columns:
        assert list(pmod_pval[col]) == list(pval_pmod[col]), col


def test_build_results_tables_pmod_pval_groups_by_model_first():
    records = [_record(1, "Simp", "Lng"), _record(1, "CAPM", "Lng")]
    pmod_pval, _ = build_results_tables(records, "Lng", _CATALOG_ORDER)
    value_cols = [c for c in pmod_pval.columns if c not in ID_FIELDS]
    # First 13 value columns should all be Simp's (model-outer, value-inner).
    assert all(c.startswith("Simp") for c in value_cols[:13])
    assert all(c.startswith("CAPM") for c in value_cols[13:26])


def test_build_results_tables_pval_pmod_groups_by_value_first():
    records = [_record(1, "Simp", "Lng"), _record(1, "CAPM", "Lng")]
    _, pval_pmod = build_results_tables(records, "Lng", _CATALOG_ORDER)
    value_cols = [c for c in pval_pmod.columns if c not in ID_FIELDS]
    # First 2 columns should both be "YrExpR" for the two models (value-outer, model-inner).
    assert value_cols[0] == "SimpYrExpR"
    assert value_cols[1] == "CAPMYrExpR"


def test_build_results_tables_model_order_follows_catalog_not_computation_order():
    records = [_record(1, "FF3", "Lng"), _record(1, "Simp", "Lng"), _record(1, "CAPM", "Lng")]
    pmod_pval, _ = build_results_tables(records, "Lng", _CATALOG_ORDER)
    value_cols = [c for c in pmod_pval.columns if c not in ID_FIELDS]
    assert value_cols[0].startswith("Simp")
    assert value_cols[13].startswith("CAPM")
    assert value_cols[26].startswith("FF3")


def test_build_results_tables_filters_by_portfolio_type():
    records = [_record(1, "Simp", "Lng"), _record(1, "Simp", "Shrt")]
    lng_pmod_pval, _ = build_results_tables(records, "Lng", _CATALOG_ORDER)
    shrt_pmod_pval, _ = build_results_tables(records, "Shrt", _CATALOG_ORDER)
    assert len(lng_pmod_pval) == 1
    assert len(shrt_pmod_pval) == 1


def test_build_sharpe_history_table_columns_and_grouping():
    records = [
        _record(1, "Simp", "Lng", YrExpShrp=1.5),
        _record(1, "CAPM", "Lng", YrExpShrp=1.2),
        _record(1, "Simp", "Shrt", YrExpShrp=1.8),
        _record(1, "CAPM", "Shrt", YrExpShrp=1.6),
    ]
    table = build_sharpe_history_table(records, _CATALOG_ORDER)
    assert list(table.columns) == [
        *ID_FIELDS,
        "SimpLngShrp",
        "CAPMLngShrp",
        "SimpShrtShrp",
        "CAPMShrtShrp",
    ]
    assert table["SimpLngShrp"].iloc[0] == pytest.approx(1.5)
    assert table["SimpShrtShrp"].iloc[0] == pytest.approx(1.8)


def test_build_sharpe_history_table_no_short_records_omits_shrt_columns():
    records = [_record(1, "Simp", "Lng", YrExpShrp=1.5)]
    table = build_sharpe_history_table(records, _CATALOG_ORDER)
    assert "SimpShrtShrp" not in table.columns


def test_build_sharpe_history_table_preserves_window_generation_order():
    records = [
        _record(1, "Simp", "Lng"),
        _record(2, "Simp", "Lng"),
        _record("Final", "Simp", "Lng"),
    ]
    table = build_sharpe_history_table(records, _CATALOG_ORDER)
    assert list(table["perfWndwNum"]) == [1, 2, "Final"]
