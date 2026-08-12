from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_output_csv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)
