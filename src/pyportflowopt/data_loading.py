from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from .errors import DataLoadError, DuplicateTickerError, MarketProxyNotFoundError
from .returns import prices_to_simple_returns


@dataclass(frozen=True)
class LoadedSeriesData:
    dates: pd.DatetimeIndex
    values: pd.DataFrame
    source_path: Path


class TabularSeriesLoader(Protocol):
    def load(self, path: Path) -> LoadedSeriesData: ...


@dataclass(frozen=True)
class SecuritiesData:
    dates: pd.DatetimeIndex
    returns: pd.DataFrame
    market_proxy_returns: pd.Series | None


def _parse_date_column(raw_date_series: pd.Series, column_name: str, path: Path) -> pd.DatetimeIndex:
    """Parses a date column, with explicit support for compact numeric date codes
    (Ken French-style): YYYYMM for monthly data, mapped to that month's actual last
    calendar day (e.g. 202402 -> 2024-02-29), and YYYYMMDD for daily/weekly data,
    which is already an exact day. A bare integer like 202405 would otherwise be
    silently misparsed by pandas' generic to_datetime as a nanosecond-epoch
    timestamp (yielding a garbage 1970 date) rather than erroring -- so these two
    compact forms must be detected and parsed explicitly, not left to inference.

    Any column not uniformly in one of those two compact forms falls through to
    pandas' generic date inference unchanged (e.g. ordinary "2024-05-31" strings).
    A column that mixes compact-numeric-looking values with other formats is
    treated as malformed and rejected, rather than silently guessing.
    """
    as_str = raw_date_series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

    all_monthly = as_str.str.fullmatch(r"\d{6}").all()
    all_daily = as_str.str.fullmatch(r"\d{8}").all()
    any_compact = as_str.str.fullmatch(r"\d{6}|\d{8}").any()

    if all_monthly:
        return pd.DatetimeIndex(pd.to_datetime(as_str, format="%Y%m") + pd.offsets.MonthEnd(0))
    if all_daily:
        return pd.DatetimeIndex(pd.to_datetime(as_str, format="%Y%m%d"))
    if any_compact:
        raise DataLoadError(
            f"{path}: '{column_name}' column mixes compact numeric dates (YYYYMM/YYYYMMDD) "
            f"with other formats -- all rows must use the same format."
        )

    try:
        return pd.DatetimeIndex(pd.to_datetime(raw_date_series))
    except Exception as exc:
        raise DataLoadError(f"{path}: could not parse '{column_name}' column as dates: {exc}") from exc


class CsvSeriesLoader:
    """Generic loader: a date column plus one or more numeric series columns.

    `date_column` names the date column's header. `drop_columns` are dropped
    before the remaining columns are treated as numeric series columns.
    """

    def __init__(self, date_column: str, drop_columns: Sequence[str] = ()):
        self._date_column = date_column
        self._drop_columns = tuple(drop_columns)

    def load(self, path: Path) -> LoadedSeriesData:
        try:
            raw = pd.read_csv(path)
        except Exception as exc:
            raise DataLoadError(f"Could not read CSV file {path}: {exc}") from exc

        if self._date_column not in raw.columns:
            raise DataLoadError(
                f"{path}: expected a '{self._date_column}' column, found columns {list(raw.columns)}"
            )

        raw = raw.drop(columns=[c for c in self._drop_columns if c in raw.columns])

        dates = _parse_date_column(raw[self._date_column], self._date_column, path)

        value_columns = [c for c in raw.columns if c != self._date_column]
        raw_values = raw[value_columns]
        values = raw_values.apply(pd.to_numeric, errors="coerce")

        bad_mask = values.isna() & raw_values.notna()
        if bad_mask.to_numpy().any():
            bad_cols = values.columns[bad_mask.any(axis=0)]
            bad_col = bad_cols[0]
            bad_row = int(bad_mask[bad_col].idxmax())
            raise DataLoadError(f"{path}: non-numeric value in column '{bad_col}' at row {bad_row}")

        values.index = pd.DatetimeIndex(dates)
        values = values.sort_index()

        return LoadedSeriesData(dates=values.index, values=values, source_path=path)


def validate_unique_tickers(columns: Sequence[str]) -> None:
    counts: dict[str, int] = {}
    for name in columns:
        counts[name] = counts.get(name, 0) + 1
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    if duplicates:
        raise DuplicateTickerError(f"Duplicate ticker column(s) found in header: {duplicates}")


def _load_securities_series(path: Path, *, loader: TabularSeriesLoader | None = None) -> LoadedSeriesData:
    """Column 1 = row number (dropped), column 2 = date, column 3+ = one ticker per column."""
    if loader is None:
        # Read the raw header without pandas' default header=0 parsing, which silently
        # renames duplicate columns (e.g. "AAA" -> "AAA.1") before we ever get a chance
        # to see the duplication -- so duplicates must be checked on this raw row.
        header_row = pd.read_csv(path, header=None, nrows=1).iloc[0].astype(str).tolist()
        if len(header_row) < 3:
            raise DataLoadError(
                f"{path}: expected at least 3 columns (row number, date, >=1 ticker), found {len(header_row)}"
            )
        row_number_column, date_column = header_row[0], header_row[1]
        validate_unique_tickers(header_row[2:])
        loader = CsvSeriesLoader(date_column=date_column, drop_columns=[row_number_column])
        return loader.load(path)

    data = loader.load(path)
    validate_unique_tickers(data.values.columns)
    return data


def load_securities_data(
    path: Path, market_proxy_ticker: str | None, price_input_kind: str
) -> SecuritiesData:
    """Loads the securities price/return file, sorted ascending by date regardless of
    the input file's row order (PyPortFlow1 writes most-recent-first) -- the internal
    1-based window row-index is assigned purely positionally over this ascending order,
    never derived from any 'Row' column value in the file.

    Does not reject missing values -- coverage is handled per-window, not at load time.
    """
    loaded = _load_securities_series(path)
    values = loaded.values

    if market_proxy_ticker is not None and market_proxy_ticker not in values.columns:
        raise MarketProxyNotFoundError(
            f"{path}: configured marketProxyTicker '{market_proxy_ticker}' not found among "
            f"columns {list(values.columns)}"
        )

    if price_input_kind == "prices":
        values = prices_to_simple_returns(values)

    if market_proxy_ticker is not None:
        market_proxy_returns = values[market_proxy_ticker]
        returns = values.drop(columns=[market_proxy_ticker])
    else:
        market_proxy_returns = None
        returns = values

    return SecuritiesData(
        dates=pd.DatetimeIndex(values.index), returns=returns, market_proxy_returns=market_proxy_returns
    )


_FACTOR_COLUMN_NAMES = ("Mkt-RF", "SMB", "SMB5Fac", "HML", "RMW", "CMA", "PR1YR", "RF")


def load_factor_series(path: Path, *, loader: TabularSeriesLoader | None = None) -> LoadedSeriesData:
    """Column 1 header 'Date'; remaining factor/RF columns identified by header name (not position).

    Only recognized factor columns actually present in the file are loaded and normalized
    (Ken French-style factor values are published in percent; normalized to decimal here).
    Which columns are actually *required* depends on the run's config and is validated
    downstream, not here, since this loader has no knowledge of which factors are enabled.
    """
    if loader is None:
        loader = CsvSeriesLoader(date_column="Date")
    data = loader.load(path)

    present = [name for name in _FACTOR_COLUMN_NAMES if name in data.values.columns]
    normalized = data.values[present] / 100.0
    return LoadedSeriesData(dates=data.dates, values=normalized, source_path=path)
