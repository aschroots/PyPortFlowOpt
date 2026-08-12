from __future__ import annotations

import pandas as pd
import pytest

from pyportflowopt.covariance import compute_covariance_matrix


def test_compute_covariance_matrix_matches_pandas_cov():
    df = pd.DataFrame({"AAA": [0.01, 0.02, -0.01, 0.03], "BBB": [0.00, 0.01, 0.02, -0.01]})
    cov = compute_covariance_matrix(df)
    pd.testing.assert_frame_equal(cov, df.cov())


def test_compute_covariance_matrix_diagonal_is_variance():
    df = pd.DataFrame({"AAA": [0.01, 0.02, -0.01, 0.03]})
    cov = compute_covariance_matrix(df)
    assert cov.loc["AAA", "AAA"] == pytest.approx(df["AAA"].var(ddof=1))
