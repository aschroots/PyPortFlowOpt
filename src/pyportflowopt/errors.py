class PyPortFlowOptError(Exception):
    """Base class for all PyPortFlowOpt errors."""


class FatalPipelineError(PyPortFlowOptError):
    """Raised for conditions that must halt the run before any output is written."""


class ConfigError(FatalPipelineError):
    """Raised when the YAML configuration fails validation."""


class DataLoadError(FatalPipelineError):
    """Raised when an input file is malformed (bad dates, non-numeric cells, wrong shape)."""


class DuplicateTickerError(FatalPipelineError):
    """Raised when the securities file's header contains duplicate ticker names."""


class MarketProxyNotFoundError(FatalPipelineError):
    """Raised when a configured marketProxyTicker is not found as a column in the securities CSV."""


class FrequencyMismatchError(FatalPipelineError):
    """Raised when the factor file's detected periods-per-year doesn't match the securities file's."""


class InsufficientHistoryError(FatalPipelineError):
    """Raised when there isn't enough row history in the securities file to form even one window."""


class FactorAlignmentError(FatalPipelineError):
    """Raised when a window date is wholly missing from the factor file after alignment."""


class CoverageExhaustedError(FatalPipelineError):
    """Raised when a window's covered ticker universe becomes empty after per-window coverage exclusion."""
