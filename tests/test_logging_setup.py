from __future__ import annotations

import logging

from pyportflowopt.logging_setup import configure_logging


def _reset_root_logger():
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()


def test_configure_logging_adds_console_handler(tmp_path):
    _reset_root_logger()
    try:
        configure_logging("INFO")
        root = logging.getLogger()
        console_handlers = [
            h
            for h in root.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        assert len(console_handlers) == 1
        assert root.level == logging.INFO
    finally:
        _reset_root_logger()


def test_configure_logging_adds_file_handler_when_log_file_given(tmp_path):
    _reset_root_logger()
    try:
        log_file = tmp_path / "run.log"
        configure_logging("DEBUG", log_file)
        root = logging.getLogger()
        file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1
        assert root.level == logging.DEBUG

        logging.getLogger("pyportflowopt").debug("hello from test")
        for h in root.handlers:
            h.flush()
        assert "hello from test" in log_file.read_text(encoding="utf-8")
    finally:
        _reset_root_logger()


def test_configure_logging_is_idempotent(tmp_path):
    _reset_root_logger()
    try:
        log_file = tmp_path / "run.log"
        configure_logging("INFO", log_file)
        configure_logging("INFO", log_file)
        configure_logging("INFO", log_file)
        root = logging.getLogger()
        console_handlers = [
            h
            for h in root.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        assert len(console_handlers) == 1
        assert len(file_handlers) == 1
    finally:
        _reset_root_logger()
