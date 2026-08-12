from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def write_price_csv(tmp_path):
    def _write(dates, ticker_data: dict, filename: str = "prices.csv"):
        df = pd.DataFrame(ticker_data)
        df.insert(0, "Date", dates)
        df.insert(0, "RowNum", range(1, len(df) + 1))
        path = tmp_path / filename
        df.to_csv(path, index=False)
        return path

    return _write


@pytest.fixture
def write_factor_csv(tmp_path):
    def _write(dates, factor_data: dict, filename: str = "factors.csv"):
        df = pd.DataFrame(factor_data)
        df.insert(0, "Date", dates)
        path = tmp_path / filename
        df.to_csv(path, index=False)
        return path

    return _write
