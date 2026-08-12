from __future__ import annotations

import math

import pandas as pd
import pytest

from pyportflowopt.config import load_config
from pyportflowopt.pipeline import run

_RUN_TIMESTAMP = pd.Timestamp("2024-01-01 12:00:00").to_pydatetime()


def test_golden_path_reproduces_verified_windowing_example(fixtures_dir, tmp_path):
    config_path = fixtures_dir / "config_valid_full.yaml"
    config = load_config(config_path)

    result = run(
        config=config,
        config_path=config_path,
        securities_path=fixtures_dir / "securities_prices.csv",
        factors_path=fixtures_dir / "factors.csv",
        output_dir=tmp_path / "out",
        run_timestamp=_RUN_TIMESTAMP,
    )

    # nPerfRows=40 (41 price rows -> 40 return rows), perfWndw=10, bckTestWndw=6,
    # perfWndwShft=10 -> nBacktests=4, matching the plan's verified example exactly.
    assert result.window_plan.n_backtests == 4
    assert result.window_plan.is_anchored is True
    assert len(result.window_plan.windows) == 5  # 4 backtest windows + Final

    w1, w2, w3, w4, wfinal = result.window_plan.windows
    assert (w1.perf_start_row, w1.perf_end_row, w1.bcktest_start_row, w1.bcktest_end_row) == (1, 10, 11, 16)
    assert (w3.perf_start_row, w3.perf_end_row, w3.bcktest_start_row, w3.bcktest_end_row) == (
        21,
        30,
        31,
        36,
    )
    assert (w4.perf_start_row, w4.perf_end_row, w4.bcktest_start_row, w4.bcktest_end_row) == (
        25,
        34,
        35,
        40,
    )
    assert w4.is_anchored is True
    assert wfinal.perf_wndw_num == "Final"


def test_golden_path_writes_all_expected_output_files(fixtures_dir, tmp_path):
    config_path = fixtures_dir / "config_valid_full.yaml"
    config = load_config(config_path)
    output_dir = tmp_path / "out"

    result = run(
        config=config,
        config_path=config_path,
        securities_path=fixtures_dir / "securities_prices.csv",
        factors_path=fixtures_dir / "factors.csv",
        output_dir=output_dir,
        run_timestamp=_RUN_TIMESTAMP,
    )

    expected_files = [
        "OptResultsLngPmodPval.csv",
        "OptResultsLngPvalPmod.csv",
        "OptResultsShrtPmodPval.csv",
        "OptResultsShrtPvalPmod.csv",
        "SharpeHistory.csv",
        "run_summary_20240101_120000.log",
    ]
    for filename in expected_files:
        assert (output_dir / filename).exists(), filename

    assert result.run_summary_path == output_dir / "run_summary_20240101_120000.log"

    sharpe_history = pd.read_csv(output_dir / "SharpeHistory.csv")
    assert list(sharpe_history.columns[:5]) == [
        "perfWndwNum",
        "PerfStart",
        "PerfEnd",
        "bckTestStart",
        "bckTestEnd",
    ]
    assert len(sharpe_history) == 5  # 4 backtest windows + Final

    lng_pmod_pval = pd.read_csv(output_dir / "OptResultsLngPmodPval.csv")
    assert "SimpYrExpR" in lng_pmod_pval.columns
    assert "CAPMYrExpR" in lng_pmod_pval.columns


def test_golden_path_final_row_has_nan_realized_fields_but_populated_forecast(fixtures_dir, tmp_path):
    config_path = fixtures_dir / "config_valid_full.yaml"
    config = load_config(config_path)

    result = run(
        config=config,
        config_path=config_path,
        securities_path=fixtures_dir / "securities_prices.csv",
        factors_path=fixtures_dir / "factors.csv",
        output_dir=tmp_path / "out",
        run_timestamp=_RUN_TIMESTAMP,
    )

    final_records = [r for r in result.records if r.perfWndwNum == "Final"]
    assert final_records
    for r in final_records:
        assert math.isnan(r.YrRealR)
        assert math.isnan(r.bckTestRealR)
        assert math.isnan(r.YrDelForeReal)
        assert r.bckTestStart is None
        assert r.bckTestEnd is None
        if not math.isnan(r.YrExpR):  # only successful optimizations have a populated forecast
            assert not math.isnan(r.YrExpShrp)


def test_golden_path_backtest_windows_have_populated_realized_fields_on_success(fixtures_dir, tmp_path):
    config_path = fixtures_dir / "config_valid_full.yaml"
    config = load_config(config_path)

    result = run(
        config=config,
        config_path=config_path,
        securities_path=fixtures_dir / "securities_prices.csv",
        factors_path=fixtures_dir / "factors.csv",
        output_dir=tmp_path / "out",
        run_timestamp=_RUN_TIMESTAMP,
    )

    non_final_successful = [
        r for r in result.records if r.perfWndwNum != "Final" and not math.isnan(r.YrExpR)
    ]
    assert non_final_successful
    for r in non_final_successful:
        assert not math.isnan(r.YrRealR)
        assert not math.isnan(r.bckTestRealR)
        assert not math.isnan(r.YrDelForeReal)


def test_golden_path_no_bcktest_run_has_all_real_columns_nan(fixtures_dir, tmp_path):
    config_path = fixtures_dir / "config_valid_no_bcktest.yaml"
    config = load_config(config_path)
    output_dir = tmp_path / "out"

    result = run(
        config=config,
        config_path=config_path,
        securities_path=fixtures_dir / "securities_prices.csv",
        factors_path=fixtures_dir / "factors.csv",
        output_dir=output_dir,
        run_timestamp=_RUN_TIMESTAMP,
    )

    assert result.window_plan.n_backtests is None
    assert result.window_plan.n_perf_windows is not None
    for r in result.records:
        assert math.isnan(r.YrRealR)
        assert math.isnan(r.bckTestRealR)
        assert math.isnan(r.YrRealVol)
        assert math.isnan(r.bckTestRealVol)
        assert math.isnan(r.YrDelForeReal)
        assert math.isnan(r.bckTestDelForeReal)
        assert math.isnan(r.YrPremRealMkt)
        assert math.isnan(r.bckTestPremRealMkt)
        assert math.isnan(r.bckTestExpR)  # B is unavailable for the whole run
        assert math.isnan(r.bckTestExpVol)

    assert "nPerfWindows" in result.summary_text
    assert "nBacktests" not in result.summary_text
    assert "Output Summary:" not in result.summary_text

    lng_pmod_pval = pd.read_csv(output_dir / "OptResultsLngPmodPval.csv")
    assert lng_pmod_pval["SimpYrRealR"].isna().all()
    assert not (output_dir / "OptResultsShrtPmodPval.csv").exists()  # shortPortfolio: false


def test_frequency_mismatch_between_securities_and_factors_is_fatal(fixtures_dir, tmp_path, write_factor_csv):
    from pyportflowopt.errors import FrequencyMismatchError

    config_path = fixtures_dir / "config_valid_no_bcktest.yaml"
    config = load_config(config_path)

    # Monthly factor dates against a weekly securities file.
    monthly_dates = [f"20200{m:02d}" for m in range(1, 10)]
    bad_factor_path = write_factor_csv(
        monthly_dates, {"Mkt-RF": [0.01] * 9, "RF": [0.001] * 9}, filename="monthly_factors.csv"
    )

    with pytest.raises(FrequencyMismatchError):
        run(
            config=config,
            config_path=config_path,
            securities_path=fixtures_dir / "securities_prices.csv",
            factors_path=bad_factor_path,
            output_dir=tmp_path / "out",
        )
