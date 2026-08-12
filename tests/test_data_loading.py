from __future__ import annotations

import pandas as pd
import pytest

from pyportflowopt.data_loading import load_factor_series, load_securities_data
from pyportflowopt.errors import DataLoadError, DuplicateTickerError, MarketProxyNotFoundError

_DATES_ASCENDING = ["20200103", "20200110", "20200117", "20200124"]


def test_load_securities_data_sorts_ascending_regardless_of_file_order(write_price_csv):
    # File written most-recent-first (like PyPortFlow1's outputs) -- loader must still
    # produce ascending dates, with the internal row index assigned positionally after
    # sorting, never from the file's own row order.
    reversed_dates = list(reversed(_DATES_ASCENDING))
    path = write_price_csv(
        reversed_dates, {"AAA": [103.0, 102.0, 101.0, 100.0], "BBB": [50.0, 49.0, 48.5, 48.0]}
    )
    data = load_securities_data(path, market_proxy_ticker=None, price_input_kind="returns")
    assert list(data.dates) == sorted(data.dates)
    assert data.returns.iloc[0]["AAA"] == 100.0  # oldest row is now the ascending row 1


def test_load_securities_data_prices_converts_to_simple_returns(write_price_csv):
    path = write_price_csv(_DATES_ASCENDING, {"AAA": [100.0, 110.0, 121.0, 108.9]})
    data = load_securities_data(path, market_proxy_ticker=None, price_input_kind="prices")
    assert len(data.returns) == 3  # one row dropped (pct_change of the first row)
    assert data.returns["AAA"].iloc[0] == pytest.approx(0.10)
    assert data.returns["AAA"].iloc[1] == pytest.approx(0.10)
    assert data.returns["AAA"].iloc[2] == pytest.approx(-0.10)


def test_load_securities_data_returns_kind_no_conversion(write_price_csv):
    path = write_price_csv(_DATES_ASCENDING, {"AAA": [0.01, -0.02, 0.03, 0.005]})
    data = load_securities_data(path, market_proxy_ticker=None, price_input_kind="returns")
    assert len(data.returns) == 4
    assert data.returns["AAA"].iloc[0] == pytest.approx(0.01)


def test_load_securities_data_no_market_proxy_keeps_all_columns(write_price_csv):
    path = write_price_csv(_DATES_ASCENDING, {"AAA": [1, 2, 3, 4], "BBB": [4, 3, 2, 1]})
    data = load_securities_data(path, market_proxy_ticker=None, price_input_kind="returns")
    assert data.market_proxy_returns is None
    assert set(data.returns.columns) == {"AAA", "BBB"}


def test_load_securities_data_splits_out_configured_market_proxy(write_price_csv):
    path = write_price_csv(_DATES_ASCENDING, {"AAA": [1.0, 2.0, 3.0, 4.0], "SPY": [10.0, 20.0, 30.0, 40.0]})
    data = load_securities_data(path, market_proxy_ticker="SPY", price_input_kind="returns")
    assert "SPY" not in data.returns.columns
    assert list(data.returns.columns) == ["AAA"]
    assert data.market_proxy_returns is not None
    assert data.market_proxy_returns.iloc[0] == 10.0


def test_load_securities_data_market_proxy_not_found_is_fatal(write_price_csv):
    path = write_price_csv(_DATES_ASCENDING, {"AAA": [1, 2, 3, 4]})
    with pytest.raises(MarketProxyNotFoundError):
        load_securities_data(path, market_proxy_ticker="SPY", price_input_kind="returns")


def test_load_securities_data_duplicate_tickers_is_fatal(tmp_path):
    path = tmp_path / "dupes.csv"
    path.write_text("RowNum,Date,AAA,AAA\n1,20200103,1,2\n2,20200110,3,4\n", encoding="utf-8")
    with pytest.raises(DuplicateTickerError):
        load_securities_data(path, market_proxy_ticker=None, price_input_kind="returns")


def test_load_securities_data_too_few_columns_is_fatal(tmp_path):
    path = tmp_path / "toofew.csv"
    path.write_text("RowNum,Date\n1,20200103\n", encoding="utf-8")
    with pytest.raises(DataLoadError):
        load_securities_data(path, market_proxy_ticker=None, price_input_kind="returns")


def test_load_factor_series_whitelists_columns_and_scales_by_100(write_factor_csv):
    path = write_factor_csv(
        _DATES_ASCENDING,
        {"Mkt-RF": [1.0, 2.0, -1.0, 0.5], "RF": [0.1, 0.1, 0.1, 0.1], "SomeUnknownCol": [9, 9, 9, 9]},
    )
    data = load_factor_series(path)
    assert "SomeUnknownCol" not in data.values.columns
    assert set(data.values.columns) == {"Mkt-RF", "RF"}
    assert data.values["Mkt-RF"].iloc[0] == pytest.approx(0.01)
    assert data.values["RF"].iloc[0] == pytest.approx(0.001)


def test_load_factor_series_all_recognized_columns_present(write_factor_csv):
    path = write_factor_csv(
        _DATES_ASCENDING,
        {
            "Mkt-RF": [1, 1, 1, 1],
            "SMB": [1, 1, 1, 1],
            "SMB5Fac": [1, 1, 1, 1],
            "HML": [1, 1, 1, 1],
            "RMW": [1, 1, 1, 1],
            "CMA": [1, 1, 1, 1],
            "PR1YR": [1, 1, 1, 1],
            "RF": [1, 1, 1, 1],
        },
    )
    data = load_factor_series(path)
    assert list(data.values.columns) == ["Mkt-RF", "SMB", "SMB5Fac", "HML", "RMW", "CMA", "PR1YR", "RF"]


def test_compact_yyyymmdd_date_parses_correctly(write_price_csv):
    path = write_price_csv(_DATES_ASCENDING, {"AAA": [1, 2, 3, 4]})
    data = load_securities_data(path, market_proxy_ticker=None, price_input_kind="returns")
    assert data.dates[0] == pd.Timestamp("2020-01-03")


def test_compact_yyyymm_monthly_date_maps_to_month_end(write_price_csv):
    path = write_price_csv(["202001", "202002", "202003", "202004"], {"AAA": [1, 2, 3, 4]})
    data = load_securities_data(path, market_proxy_ticker=None, price_input_kind="returns")
    assert data.dates[0] == pd.Timestamp("2020-01-31")
    assert data.dates[1] == pd.Timestamp("2020-02-29")  # leap year


def test_mixed_compact_and_non_compact_dates_is_fatal(tmp_path):
    path = tmp_path / "mixed.csv"
    path.write_text(
        "RowNum,Date,AAA\n1,20200103,1\n2,2020-01-10,2\n",
        encoding="utf-8",
    )
    with pytest.raises(DataLoadError):
        load_securities_data(path, market_proxy_ticker=None, price_input_kind="returns")


def test_non_numeric_value_is_fatal(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("RowNum,Date,AAA\n1,20200103,notanumber\n2,20200110,2\n", encoding="utf-8")
    with pytest.raises(DataLoadError):
        load_securities_data(path, market_proxy_ticker=None, price_input_kind="returns")
