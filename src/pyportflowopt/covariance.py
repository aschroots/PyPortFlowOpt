from __future__ import annotations

import pandas as pd


def compute_covariance_matrix(window_returns: pd.DataFrame) -> pd.DataFrame:
    return window_returns.cov()
