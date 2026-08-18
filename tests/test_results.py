from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from pyportflowopt.results import (
    ID_FIELDS,
    STAT_ROW_LABELS,
    VALUE_FIELDS,
    OptimizationRecord,
    build_results_tables,
    build_sharpe_history_table,
    build_weights_tables,
)

_CATALOG_ORDER = ("CAPM", "FF3", "FFC4", "FF5", "FF5Mod", "FFC6")


def _record(
    perf_wndw_num, model, portfolio_type, base_value=1.0, weights=None, **overrides
) -> OptimizationRecord:
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
        weights=weights,
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


def _ticker_rows(table):
    return table[~table["Ticker"].isin(STAT_ROW_LABELS)]


def test_build_weights_tables_distinguishes_excluded_from_zero_weight():
    records = [
        _record(1, "Simp", "Lng", weights={"AAA": 0.6, "BBB": 0.4}),
        _record(2, "Simp", "Lng", weights={"AAA": 0.0, "BBB": 0.6, "CCC": 0.4}),
    ]
    table = build_weights_tables(records, "Lng", ("AAA", "BBB", "CCC"), _CATALOG_ORDER)["Simp"]

    ccc_window1 = table.loc[table["Ticker"] == "CCC", 1].iloc[0]
    assert pd.isna(ccc_window1)  # excluded from window 1's optimization

    aaa_window2 = table.loc[table["Ticker"] == "AAA", 2].iloc[0]
    assert aaa_window2 == pytest.approx(0.0)  # included, optimizer assigned exactly 0.0


def test_build_weights_tables_failed_optimization_is_all_nan_column():
    nan = float("nan")
    records = [
        _record(1, "Simp", "Lng", weights={"AAA": 1.0}),
        _record(2, "Simp", "Lng", weights=None, base_value=nan),
    ]
    table = build_weights_tables(records, "Lng", ("AAA",), _CATALOG_ORDER)["Simp"]
    assert table[2].isna().all()


def test_build_weights_tables_ticker_never_covered_is_all_nan_row_and_sinks_to_bottom():
    records = [_record(1, "Simp", "Lng", weights={"AAA": 1.0})]
    table = build_weights_tables(records, "Lng", ("AAA", "ZZZ"), _CATALOG_ORDER)["Simp"]
    ticker_rows = _ticker_rows(table)
    assert list(ticker_rows["Ticker"]) == ["AAA", "ZZZ"]
    assert ticker_rows.loc[ticker_rows["Ticker"] == "ZZZ", 1].isna().all()


def test_build_weights_tables_nonzero_tickers_sort_before_always_zero_tickers():
    records = [
        _record(1, "Simp", "Lng", weights={"AAA": 0.0, "BBB": 0.5, "CCC": 0.0, "DDD": 0.5}),
    ]
    table = build_weights_tables(records, "Lng", ("AAA", "BBB", "CCC", "DDD"), _CATALOG_ORDER)["Simp"]
    ticker_rows = _ticker_rows(table)
    assert list(ticker_rows["Ticker"]) == ["BBB", "DDD", "AAA", "CCC"]


def test_build_weights_tables_columns_preserve_window_generation_order():
    records = [
        _record(1, "Simp", "Lng", weights={"AAA": 1.0}),
        _record(2, "Simp", "Lng", weights={"AAA": 1.0}),
        _record("Final", "Simp", "Lng", weights={"AAA": 1.0}),
    ]
    table = build_weights_tables(records, "Lng", ("AAA",), _CATALOG_ORDER)["Simp"]
    assert list(table.columns)[1:] == [1, 2, "Final"]


def test_build_weights_tables_returns_dict_per_model_in_catalog_order():
    records = [
        _record(1, "FF3", "Lng", weights={"AAA": 1.0}),
        _record(1, "Simp", "Lng", weights={"AAA": 1.0}),
        _record(1, "CAPM", "Lng", weights={"AAA": 1.0}),
    ]
    result = build_weights_tables(records, "Lng", ("AAA",), _CATALOG_ORDER)
    assert list(result.keys()) == ["Simp", "CAPM", "FF3"]


def test_build_weights_tables_filters_by_portfolio_type():
    records = [
        _record(1, "Simp", "Lng", weights={"AAA": 1.0}),
        _record(1, "Simp", "Shrt", weights={"AAA": 0.5}),
    ]
    lng_result = build_weights_tables(records, "Lng", ("AAA",), _CATALOG_ORDER)
    shrt_result = build_weights_tables(records, "Shrt", ("AAA",), _CATALOG_ORDER)
    assert lng_result["Simp"].loc[lng_result["Simp"]["Ticker"] == "AAA", 1].iloc[0] == pytest.approx(1.0)
    assert shrt_result["Simp"].loc[shrt_result["Simp"]["Ticker"] == "AAA", 1].iloc[0] == pytest.approx(0.5)


def test_build_weights_tables_first_column_is_ticker():
    records = [_record(1, "Simp", "Lng", weights={"AAA": 1.0})]
    table = build_weights_tables(records, "Lng", ("AAA",), _CATALOG_ORDER)["Simp"]
    assert table.columns[0] == "Ticker"


def test_build_weights_tables_leading_rows_are_the_forecast_stats():
    records = [
        _record(
            1,
            "Simp",
            "Lng",
            weights={"AAA": 1.0},
            YrExpR=0.08,
            YrExpVol=0.14,
            YrExpShrp=0.57,
        )
    ]
    table = build_weights_tables(records, "Lng", ("AAA",), _CATALOG_ORDER)["Simp"]
    assert list(table["Ticker"][:3]) == ["YrExpR", "YrExpVol", "YrExpShrp"]
    assert table.loc[table["Ticker"] == "YrExpR", 1].iloc[0] == pytest.approx(0.08)
    assert table.loc[table["Ticker"] == "YrExpVol", 1].iloc[0] == pytest.approx(0.14)
    assert table.loc[table["Ticker"] == "YrExpShrp", 1].iloc[0] == pytest.approx(0.57)
