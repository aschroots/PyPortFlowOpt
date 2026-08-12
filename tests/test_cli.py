from __future__ import annotations

import logging

from pyportflowopt.cli import EXIT_CONFIG_ERROR, EXIT_PIPELINE_ERROR, EXIT_SUCCESS, main


def _reset_root_logger():
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()


def test_cli_success_writes_all_outputs_and_returns_zero(fixtures_dir, tmp_path):
    _reset_root_logger()
    try:
        output_dir = tmp_path / "out"
        exit_code = main(
            [
                "--config",
                str(fixtures_dir / "config_valid_full.yaml"),
                "--securities",
                str(fixtures_dir / "securities_prices.csv"),
                "--factors",
                str(fixtures_dir / "factors.csv"),
                "--output-dir",
                str(output_dir),
                "--log-level",
                "DEBUG",
            ]
        )
        assert exit_code == EXIT_SUCCESS

        # Each run gets its own runOpt_<timestamp> subfolder under --output-dir (mirrors
        # PyPortFlow1's run_<timestamp>/ and PyPortFlow2's runStats_<timestamp>/ convention)
        # so repeated runs never clobber a prior run's outputs.
        run_dirs = list(output_dir.glob("runOpt_*"))
        assert len(run_dirs) == 1
        run_dir = run_dirs[0]

        assert (run_dir / "run.log").exists()
        assert (run_dir / "SharpeHistory.csv").exists()
        assert (run_dir / "OptResultsLngPmodPval.csv").exists()
        assert (run_dir / "OptResultsShrtPmodPval.csv").exists()
        summary_logs = list(run_dir.glob("run_summary_*.log"))
        assert len(summary_logs) == 1

        run_log_content = (run_dir / "run.log").read_text(encoding="utf-8")
        assert "DEBUG" in run_log_content  # full DEBUG-level trace, not just INFO+
    finally:
        _reset_root_logger()


def test_cli_config_error_returns_exit_code_2(fixtures_dir, tmp_path):
    _reset_root_logger()
    try:
        bad_config = tmp_path / "bad_config.yaml"
        bad_config.write_text("doBcktest: true\n", encoding="utf-8")  # missing required fields
        exit_code = main(
            [
                "--config",
                str(bad_config),
                "--securities",
                str(fixtures_dir / "securities_prices.csv"),
                "--factors",
                str(fixtures_dir / "factors.csv"),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
        assert exit_code == EXIT_CONFIG_ERROR
    finally:
        _reset_root_logger()


def test_cli_pipeline_error_returns_exit_code_1(fixtures_dir, tmp_path):
    _reset_root_logger()
    try:
        # A market proxy ticker that doesn't exist in the securities file -> fatal
        # MarketProxyNotFoundError, raised during run() rather than config loading.
        bad_config = tmp_path / "bad_proxy_config.yaml"
        bad_config.write_text(
            "marketProxyTicker: NOTATICKER\n"
            "priceInputKind: prices\n"
            "perfWndw: 10\n"
            "perfWndwShft: 10\n"
            "doBcktest: false\n"
            "shortPortfolio: false\n"
            "factorModels: []\n",
            encoding="utf-8",
        )
        exit_code = main(
            [
                "--config",
                str(bad_config),
                "--securities",
                str(fixtures_dir / "securities_prices.csv"),
                "--factors",
                str(fixtures_dir / "factors.csv"),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
        assert exit_code == EXIT_PIPELINE_ERROR
    finally:
        _reset_root_logger()


def test_cli_output_dir_created_even_when_config_is_invalid(fixtures_dir, tmp_path):
    _reset_root_logger()
    try:
        bad_config = tmp_path / "bad_config.yaml"
        bad_config.write_text("doBcktest: true\n", encoding="utf-8")
        output_dir = tmp_path / "out"
        main(
            [
                "--config",
                str(bad_config),
                "--securities",
                str(fixtures_dir / "securities_prices.csv"),
                "--factors",
                str(fixtures_dir / "factors.csv"),
                "--output-dir",
                str(output_dir),
            ]
        )
        # The runOpt_<timestamp> subfolder (and run.log inside it) is created BEFORE config
        # validation runs (matches PyPortFlow1's ordering, since the FileHandler needs the
        # directory to exist).
        run_dirs = list(output_dir.glob("runOpt_*"))
        assert len(run_dirs) == 1
        assert (run_dirs[0] / "run.log").exists()
    finally:
        _reset_root_logger()
