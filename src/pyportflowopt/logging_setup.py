import logging
from pathlib import Path

_CONSOLE_FORMAT = "%(levelname)s %(name)s: %(message)s"
_FILE_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure the root logger (idempotent: safe to call more than once).

    Configuring the root logger, rather than just pyportflowopt's own logger, means
    cvxpy's/pypfopt's internal solver warnings are captured in the same output too.
    """
    root = logging.getLogger()
    root.setLevel(level)

    has_console = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers
    )
    if not has_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT))
        root.addHandler(console_handler)

    if log_file is not None:
        already_attached = any(
            isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file.resolve()
            for h in root.handlers
        )
        if not already_attached:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
            root.addHandler(file_handler)
