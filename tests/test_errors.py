from __future__ import annotations

import pytest

from pyportflowopt.errors import (
    ConfigError,
    CoverageExhaustedError,
    DataLoadError,
    DuplicateTickerError,
    FactorAlignmentError,
    FatalPipelineError,
    FrequencyMismatchError,
    InsufficientHistoryError,
    MarketProxyNotFoundError,
    PyPortFlowOptError,
)


@pytest.mark.parametrize(
    "exc_class",
    [
        FatalPipelineError,
        ConfigError,
        DataLoadError,
        DuplicateTickerError,
        MarketProxyNotFoundError,
        FrequencyMismatchError,
        InsufficientHistoryError,
        FactorAlignmentError,
        CoverageExhaustedError,
    ],
)
def test_all_errors_root_at_pyportflowopt_error(exc_class):
    assert issubclass(exc_class, PyPortFlowOptError)


@pytest.mark.parametrize(
    "exc_class",
    [
        ConfigError,
        DataLoadError,
        DuplicateTickerError,
        MarketProxyNotFoundError,
        FrequencyMismatchError,
        InsufficientHistoryError,
        FactorAlignmentError,
        CoverageExhaustedError,
    ],
)
def test_specific_errors_are_fatal_pipeline_errors(exc_class):
    assert issubclass(exc_class, FatalPipelineError)


def test_pyportflowopt_error_is_an_exception():
    assert issubclass(PyPortFlowOptError, Exception)
