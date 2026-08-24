from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import yaml

from pyportflowopt.config import load_config
from pyportflowopt.pipeline import _portfolio_bcktest_returns, run

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
        "OptResultsOverviewLng.csv",
        "OptResultsOverviewShrt.csv",
        "SharpeHistory.csv",
        "WtsLngSimp.csv",
        "WtsLngCAPM.csv",
        "WtsLngFF3.csv",
        "WtsLngFFC4.csv",
        "WtsLngFF5.csv",
        "WtsLngFF5Mod.csv",
        "WtsLngFFC6.csv",
        "WtsShrtSimp.csv",
        "WtsShrtCAPM.csv",
        "WtsShrtFF3.csv",
        "WtsShrtFFC4.csv",
        "WtsShrtFF5.csv",
        "WtsShrtFF5Mod.csv",
        "WtsShrtFFC6.csv",
        "run_summary_20240101_120000.log",
    ]
    for filename in expected_files:
        assert (output_dir / filename).exists(), filename

    assert result.run_summary_path == output_dir / "run_summary_20240101_120000.log"
    assert set(result.weights_lng_paths) == {"Simp", "CAPM", "FF3", "FFC4", "FF5", "FF5Mod", "FFC6"}
    assert result.weights_lng_paths["Simp"] == output_dir / "WtsLngSimp.csv"
    assert set(result.weights_shrt_paths) == {"Simp", "CAPM", "FF3", "FFC4", "FF5", "FF5Mod", "FFC6"}
    assert result.weights_shrt_paths["Simp"] == output_dir / "WtsShrtSimp.csv"

    sharpe_history = pd.read_csv(output_dir / "SharpeHistory.csv")
    assert list(sharpe_history.columns[:5]) == [
        "perfWndwNum",
        "PerfStart",
        "PerfEnd",
        "bkTstStart",
        "bkTstEnd",
    ]
    assert len(sharpe_history) == 5  # 4 backtest windows + Final

    lng_pmod_pval = pd.read_csv(output_dir / "OptResultsLngPmodPval.csv")
    assert "SimpYrExpR" in lng_pmod_pval.columns
    assert "CAPMYrExpR" in lng_pmod_pval.columns
    assert "SimpYrExpRF" in lng_pmod_pval.columns
    assert "SimpYrRealRF" in lng_pmod_pval.columns
    assert "SimpYrRealShrp" in lng_pmod_pval.columns

    overview_lng = pd.read_csv(output_dir / "OptResultsOverviewLng.csv")
    assert list(overview_lng.columns[:7]) == [
        "Model",
        "perfWndwNum",
        "PerfStart",
        "PerfEnd",
        "bkTstStart",
        "bkTstEnd",
        "Non-Zero Wts",
    ]
    assert len(overview_lng) == 5 * 7  # 5 windows * 7 models (Simp + 6 catalog models)

    overview_shrt = pd.read_csv(output_dir / "OptResultsOverviewShrt.csv")
    assert "Non-Zero Wts" not in overview_shrt.columns

    weights_simp = pd.read_csv(output_dir / "WtsLngSimp.csv")
    assert weights_simp.columns[0] == "Ticker"
    assert list(weights_simp["Ticker"][:5]) == ["YrExpR", "YrExpVol", "YrExpRF", "YrExpShrp", "Non-Zero Wts"]
    assert set(weights_simp["Ticker"][5:]) == {"AAA", "BBB", "CCC", "DDD"}
    assert len(weights_simp) == 5 + 4  # 5 stat rows + 4 tickers
    assert len(weights_simp.columns) == 1 + 5  # "Ticker" + 4 backtest windows + Final
    stat_labels = {"YrExpR", "YrExpVol", "YrExpRF", "YrExpShrp", "Non-Zero Wts"}
    ticker_weights = weights_simp[~weights_simp["Ticker"].isin(stat_labels)]
    window_cols = [c for c in weights_simp.columns if c != "Ticker"]
    for col in window_cols:
        values = ticker_weights[col]
        if values.notna().any():  # skip windows where this model's optimization failed
            assert values.sum(skipna=True) == pytest.approx(1.0)

    weights_shrt_simp = pd.read_csv(output_dir / "WtsShrtSimp.csv")
    assert list(weights_shrt_simp["Ticker"][:4]) == ["YrExpR", "YrExpVol", "YrExpRF", "YrExpShrp"]
    assert "Non-Zero Wts" not in set(weights_shrt_simp["Ticker"])


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
        assert math.isnan(r.bkTstRealR)
        assert math.isnan(r.YrRealRF)
        assert math.isnan(r.YrRealShrp)
        assert math.isnan(r.YrDelExpReal)
        assert r.bkTstStart is None
        assert r.bkTstEnd is None
        if not math.isnan(r.YrExpR):  # only successful optimizations have a populated forecast
            assert not math.isnan(r.YrExpRF)
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
        assert not math.isnan(r.bkTstRealR)
        assert not math.isnan(r.YrRealRF)
        assert not math.isnan(r.YrRealShrp)
        assert not math.isnan(r.YrDelExpReal)


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
        assert math.isnan(r.bkTstRealR)
        assert math.isnan(r.YrRealVol)
        assert math.isnan(r.bkTstRealVol)
        assert math.isnan(r.YrRealRF)
        assert math.isnan(r.YrRealShrp)
        assert math.isnan(r.YrDelExpReal)
        assert math.isnan(r.bkTstDelExpReal)
        assert math.isnan(r.YrPremRealMkt)
        assert math.isnan(r.bkTstPremRealMkt)
        assert math.isnan(r.bkTstExpR)  # B is unavailable for the whole run
        assert math.isnan(r.bkTstExpVol)
        if not math.isnan(r.YrExpR):  # only successful optimizations have a populated forecast
            assert not math.isnan(r.YrExpRF)

    assert "nPerfWindows" in result.summary_text
    assert "nBacktests" not in result.summary_text
    assert "Output Summary:" not in result.summary_text

    lng_pmod_pval = pd.read_csv(output_dir / "OptResultsLngPmodPval.csv")
    assert lng_pmod_pval["SimpYrRealR"].isna().all()
    assert not (output_dir / "OptResultsShrtPmodPval.csv").exists()  # shortPortfolio: false
    assert not (output_dir / "OptResultsOverviewShrt.csv").exists()  # shortPortfolio: false
    assert (output_dir / "OptResultsOverviewLng.csv").exists()
    assert not list(output_dir.glob("WtsShrt*.csv"))  # shortPortfolio: false


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


def test_wts_epsilon_zeroes_dust_and_renormalizes_remaining_weights(fixtures_dir, tmp_path):
    base_config = yaml.safe_load((fixtures_dir / "config_valid_full.yaml").read_text(encoding="utf-8"))
    # With 4 tickers and epsilon=0.5, at most 2 raw weights can each be >= 0.5 (they'd
    # otherwise sum past 1) -- a deterministic invariant to check without depending on the
    # optimizer's actual output for this fixture's data.
    base_config["wtsEpsilon"] = 0.5
    config_path = tmp_path / "config_wts_epsilon.yaml"
    config_path.write_text(yaml.safe_dump(base_config), encoding="utf-8")
    config = load_config(config_path)
    assert config.wtsEpsilon == pytest.approx(0.5)

    output_dir = tmp_path / "out"
    run(
        config=config,
        config_path=config_path,
        securities_path=fixtures_dir / "securities_prices.csv",
        factors_path=fixtures_dir / "factors.csv",
        output_dir=output_dir,
        run_timestamp=_RUN_TIMESTAMP,
    )

    overview_lng = pd.read_csv(output_dir / "OptResultsOverviewLng.csv")
    populated = overview_lng[overview_lng["Non-Zero Wts"].notna()]
    assert not populated.empty
    assert (populated["Non-Zero Wts"] <= 2).all()

    weights_simp = pd.read_csv(output_dir / "WtsLngSimp.csv")
    stat_labels = {"YrExpR", "YrExpVol", "YrExpRF", "YrExpShrp", "Non-Zero Wts"}
    ticker_weights = weights_simp[~weights_simp["Ticker"].isin(stat_labels)]
    window_cols = [c for c in weights_simp.columns if c != "Ticker"]
    for col in window_cols:
        values = ticker_weights[col]
        if values.notna().any():  # skip windows where this model's optimization failed
            assert values.sum(skipna=True) == pytest.approx(1.0)


def test_portfolio_bcktest_returns_is_buy_and_hold_not_rebalanced():
    # A and B diverge every period, so a buy-and-hold portfolio's effective weights drift
    # away from the initial 50/50 target -- a constant-mix/rebalanced calculation would not
    # show that drift and would land on a different total.
    bck_dates = pd.date_range("2020-01-01", periods=3, freq="D")
    returns = pd.DataFrame(
        {"A": [0.10, 0.10, 0.10], "B": [-0.10, -0.10, -0.10], "C": [0.05, 0.05, 0.05]},
        index=bck_dates,
    )
    weights = {"A": 0.5, "B": 0.5}

    result = _portfolio_bcktest_returns(returns, bck_dates, weights)

    assert list(result.index) == list(bck_dates)

    # Closed-form buy-and-hold total: each security compounds on its own, weights combine
    # the compounded values once, at the end.
    expected_total = 0.5 * (1.10**3) + 0.5 * (0.90**3) - 1.0
    compounded_total = float(np.prod(1.0 + result.to_numpy()) - 1.0)
    assert compounded_total == pytest.approx(expected_total)

    # The naive rebalanced-every-period calculation (re-applying the initial weights to each
    # period's raw returns) gives a different, wrong total here -- this is the regression
    # guard for the original bug.
    naive_period_returns = returns[["A", "B"]].to_numpy() @ np.array([0.5, 0.5])
    naive_total = float(np.prod(1.0 + naive_period_returns) - 1.0)
    assert naive_total == pytest.approx(0.0)
    assert compounded_total != pytest.approx(naive_total)

    # First period: no drift has happened yet, so buy-and-hold and rebalanced agree.
    assert result.iloc[0] == pytest.approx(naive_period_returns[0])
