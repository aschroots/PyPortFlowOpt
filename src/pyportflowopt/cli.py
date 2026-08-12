from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from .config import load_config
from .errors import ConfigError, FatalPipelineError
from .logging_setup import configure_logging
from .pipeline import run

EXIT_SUCCESS = 0
EXIT_PIPELINE_ERROR = 1
EXIT_CONFIG_ERROR = 2

logger = logging.getLogger("pyportflowopt")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyportflowopt",
        description=(
            "Rolling multi-model portfolio optimization and backtest: Simple-average and "
            "Fama-French-style factor-model expected returns, optimized via pyportfolioopt's "
            "tangent portfolio, backtested against realized returns over shifting windows."
        ),
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML config file")
    parser.add_argument(
        "--securities", type=Path, required=True, help="Path to the securities price/return CSV"
    )
    parser.add_argument("--factors", type=Path, required=True, help="Path to the factor-returns CSV")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write outputs into")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Every run gets its own runOpt_<timestamp> subfolder under --output-dir, mirroring
    # PyPortFlow1's run_<timestamp>/ and PyPortFlow2's runStats_<timestamp>/ convention, so
    # repeated runs never clobber a prior run's outputs. The subfolder must be created (and
    # its timestamp fixed) here, before logging is configured, since run.log needs to live
    # inside it from the very start of the run -- unlike PyPortFlow2, which has no run.log to
    # coordinate with and so can leave this to pipeline.py.
    run_timestamp = datetime.now()
    run_output_dir = args.output_dir / f"runOpt_{run_timestamp:%Y%m%d_%H%M%S}"

    try:
        run_output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        configure_logging(args.log_level)
        logger.error("Could not create output directory %s: %s", run_output_dir, exc)
        return EXIT_PIPELINE_ERROR

    configure_logging(args.log_level, run_output_dir / "run.log")

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        logger.error(str(exc))
        return EXIT_CONFIG_ERROR

    try:
        result = run(
            config=config,
            config_path=args.config,
            securities_path=args.securities,
            factors_path=args.factors,
            output_dir=run_output_dir,
            run_timestamp=run_timestamp,
        )
    except FatalPipelineError as exc:
        logger.error(str(exc))
        return EXIT_PIPELINE_ERROR

    logger.info("Run complete. Outputs written to %s", result.output_dir)
    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
