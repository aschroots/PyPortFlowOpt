from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from .annualization import (
    NAN_REALIZED_PERFORMANCE,
    compute_forecast_annualization,
    compute_realized_performance,
    compute_realized_rf,
    compute_realized_sharpe,
    residual_del_exp_real,
    residual_prem_real_mkt,
)
from .config import FACTOR_MODEL_CATALOG, PyPortFlowOptConfig
from .covariance import compute_covariance_matrix
from .data_loading import load_factor_series, load_securities_data
from .errors import FatalPipelineError, FrequencyMismatchError
from .factor_models import align_factor_window, compute_factor_date_ranges, latest_value_in_window
from .optimization import (
    LONG_PORTFOLIO_TYPE,
    SHORT_PORTFOLIO_TYPE,
    SIMP_MODEL_NAME,
    apply_weight_epsilon,
    compute_factor_model_mu,
    compute_long_run_cutoff,
    compute_portfolio_performance,
    compute_simp_mu,
    optimize_portfolio,
)
from .output_writing import write_output_csv
from .results import (
    OptimizationRecord,
    build_overview_table,
    build_results_tables,
    build_sharpe_history_table,
    build_weights_tables,
)
from .returns import detect_periods_per_year
from .run_summary import RunSummary, write_run_summary
from .windowing import (
    Window,
    WindowPlan,
    build_window_plan,
    check_window_coverage,
    resolve_window_dates,
    window_row_dates,
)

logger = logging.getLogger("pyportflowopt")


@dataclass
class RunResult:
    output_dir: Path
    records: list[OptimizationRecord]
    window_plan: WindowPlan
    summary_text: str
    run_summary_path: Path
    sharpe_history_path: Path
    opt_results_lng_pmod_pval_path: Path
    opt_results_lng_pval_pmod_path: Path
    opt_results_shrt_pmod_pval_path: Path | None
    opt_results_shrt_pval_pmod_path: Path | None
    opt_results_overview_lng_path: Path
    opt_results_overview_shrt_path: Path | None
    weights_lng_paths: dict[str, Path]
    weights_shrt_paths: dict[str, Path]


def _nan_record(
    window: Window, perf_start, perf_end, bcktest_start, bcktest_end, model: str, portfolio_type: str
) -> OptimizationRecord:
    nan = float("nan")
    return OptimizationRecord(
        perfWndwNum=window.perf_wndw_num,
        PerfStart=perf_start,
        PerfEnd=perf_end,
        bkTstStart=bcktest_start,
        bkTstEnd=bcktest_end,
        Model=model,
        PortfolioType=portfolio_type,
        YrExpR=nan,
        bkTstExpR=nan,
        YrExpVol=nan,
        bkTstExpVol=nan,
        YrExpRF=nan,
        YrExpShrp=nan,
        YrRealR=nan,
        bkTstRealR=nan,
        YrRealVol=nan,
        bkTstRealVol=nan,
        YrRealRF=nan,
        YrRealShrp=nan,
        YrDelExpReal=nan,
        bkTstDelExpReal=nan,
        YrPremRealMkt=nan,
        bkTstPremRealMkt=nan,
    )


def _portfolio_bcktest_returns(
    returns: pd.DataFrame, bck_dates: pd.DatetimeIndex, weights: dict[str, float]
) -> pd.Series:
    weights_series = pd.Series(weights)
    bck_returns = returns.loc[bck_dates, weights_series.index]
    return pd.Series(bck_returns.to_numpy() @ weights_series.to_numpy(), index=bck_returns.index)


def run(
    config: PyPortFlowOptConfig,
    config_path: Path,
    securities_path: Path,
    factors_path: Path,
    output_dir: Path,
    *,
    run_timestamp: datetime | None = None,
) -> RunResult:
    summary = RunSummary()
    timestamp_str = (run_timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    logger.debug(
        "Starting run: config=%s securities=%s factors=%s", config_path, securities_path, factors_path
    )
    logger.debug("Config: %s", config)

    securities_data = load_securities_data(securities_path, config.marketProxyTicker, config.priceInputKind)
    periods_per_year = detect_periods_per_year(securities_data.dates)
    ticker_count = len(securities_data.returns.columns)
    securities_date_range = (securities_data.dates.min(), securities_data.dates.max())
    n_perf_rows = len(securities_data.returns)
    logger.debug(
        "Loaded securities data: tickers=%s marketProxy=%s periodsPerYear=%d nPerfRows=%d dateRange=%s..%s",
        list(securities_data.returns.columns),
        config.marketProxyTicker,
        periods_per_year,
        n_perf_rows,
        securities_date_range[0].date(),
        securities_date_range[1].date(),
    )

    factor_data = load_factor_series(factors_path)
    if "RF" not in factor_data.values.columns:
        raise FatalPipelineError(f"Factor file {factors_path} is missing the required 'RF' column.")
    factor_periods_per_year = detect_periods_per_year(factor_data.dates)
    if factor_periods_per_year != periods_per_year:
        raise FrequencyMismatchError(
            f"Detected periods per year mismatch: securities file={periods_per_year}, "
            f"factor file={factor_periods_per_year}."
        )
    factor_date_ranges = compute_factor_date_ranges(factor_data.values)
    logger.debug(
        "Loaded factor data: columns=%s periodsPerYear=%d",
        list(factor_data.values.columns),
        factor_periods_per_year,
    )

    window_plan = build_window_plan(
        n_perf_rows, config.perfWndw, config.perfWndwShft, config.doBcktest, config.bckTestWndw
    )
    logger.debug(
        "Window plan: %d window(s), nBacktests=%s nPerfWindows=%s isAnchored=%s",
        len(window_plan.windows),
        window_plan.n_backtests,
        window_plan.n_perf_windows,
        window_plan.is_anchored,
    )

    portfolio_types: list[tuple[str, tuple[float, float]]] = [(LONG_PORTFOLIO_TYPE, (0.0, 1.0))]
    if config.shortPortfolio:
        portfolio_types.append((SHORT_PORTFOLIO_TYPE, (config.shortLimit, 1.0)))
    model_names = [SIMP_MODEL_NAME, *config.factorModels]
    logger.debug("Models: %s, portfolio types: %s", model_names, [pt for pt, _ in portfolio_types])

    records: list[OptimizationRecord] = []
    for window in window_plan.windows:
        window_dates = resolve_window_dates(window, securities_data.dates)
        perf_dates = window_row_dates(securities_data.dates, window.perf_start_row, window.perf_end_row)
        combined_end_row = window.bcktest_end_row if window.has_bcktest else window.perf_end_row
        combined_dates = window_row_dates(securities_data.dates, window.perf_start_row, combined_end_row)
        logger.debug(
            "Window %s: perf=[%s..%s] bcktest=[%s..%s] hasBcktest=%s isAnchored=%s",
            window.perf_wndw_num,
            window_dates.perf_start.date(),
            window_dates.perf_end.date(),
            window_dates.bcktest_start.date() if window_dates.bcktest_start else None,
            window_dates.bcktest_end.date() if window_dates.bcktest_end else None,
            window.has_bcktest,
            window.is_anchored,
        )

        coverage = check_window_coverage(securities_data.returns, combined_dates)
        covered_tickers = [t for t, ok in coverage.items() if ok]
        for ticker, ok in coverage.items():
            if not ok:
                summary.note(
                    f"Window {window.perf_wndw_num}: ticker '{ticker}' excluded from optimization "
                    f"(missing data somewhere in [{combined_dates[0].date()}, {combined_dates[-1].date()}])."
                )
        if not covered_tickers:
            raise FatalPipelineError(
                f"Window {window.perf_wndw_num}: no tickers have full data coverage over "
                f"[{combined_dates[0].date()}, {combined_dates[-1].date()}]; optimizable universe is empty."
            )

        logger.debug(
            "Window %s: covered tickers=%s excluded=%s",
            window.perf_wndw_num,
            covered_tickers,
            [t for t, ok in coverage.items() if not ok],
        )

        perf_returns = securities_data.returns.loc[perf_dates, covered_tickers]
        cov = compute_covariance_matrix(perf_returns)

        aligned_factors = align_factor_window(factor_data.values, perf_dates, periods_per_year)
        rf = latest_value_in_window(aligned_factors["RF"], window_dates.perf_start, window_dates.perf_end)
        long_run_cutoff = compute_long_run_cutoff(window_dates.perf_end, periods_per_year)
        logger.debug("Window %s: rf=%.8f longRunCutoff=%s", window.perf_wndw_num, rf, long_run_cutoff.date())

        market_realized = None
        if window.has_bcktest and securities_data.market_proxy_returns is not None:
            bck_dates = window_row_dates(
                securities_data.dates, window.bcktest_start_row, window.bcktest_end_row
            )
            market_bck_returns = securities_data.market_proxy_returns.loc[bck_dates]
            market_realized = compute_realized_performance(market_bck_returns.to_numpy(), periods_per_year)

        for model_name in model_names:
            if model_name == SIMP_MODEL_NAME:
                mu = compute_simp_mu(perf_returns)
            else:
                mu, _betas = compute_factor_model_mu(
                    model_name,
                    perf_returns,
                    aligned_factors,
                    factor_data.values,
                    rf,
                    long_run_cutoff,
                    periods_per_year,
                )
            logger.debug("Window %s [%s]: mu=%s", window.perf_wndw_num, model_name, mu.to_dict())

            for portfolio_type, bounds in portfolio_types:
                outcome = optimize_portfolio(mu, cov, rf, bounds)
                logger.debug(
                    "Window %s [%s/%s]: bounds=%s success=%s weights=%s",
                    window.perf_wndw_num,
                    model_name,
                    portfolio_type,
                    bounds,
                    outcome.success,
                    outcome.weights,
                )

                if not outcome.success:
                    message = (
                        f"Window {window.perf_wndw_num} [{model_name}/{portfolio_type}]: "
                        f"optimization failed: {outcome.failure_reason}"
                    )
                    summary.flag(message)
                    logger.warning(message)
                    records.append(
                        _nan_record(
                            window,
                            window_dates.perf_start.date(),
                            window_dates.perf_end.date(),
                            window_dates.bcktest_start.date() if window_dates.bcktest_start else None,
                            window_dates.bcktest_end.date() if window_dates.bcktest_end else None,
                            model_name,
                            portfolio_type,
                        )
                    )
                    continue

                adjusted_weights = apply_weight_epsilon(outcome.weights, config.wtsEpsilon)
                if not any(w != 0.0 for w in adjusted_weights.values()):
                    message = (
                        f"Window {window.perf_wndw_num} [{model_name}/{portfolio_type}]: "
                        f"wtsEpsilon={config.wtsEpsilon} zeroed every weight; "
                        f"treating as a failed optimization."
                    )
                    summary.flag(message)
                    logger.warning(message)
                    records.append(
                        _nan_record(
                            window,
                            window_dates.perf_start.date(),
                            window_dates.perf_end.date(),
                            window_dates.bcktest_start.date() if window_dates.bcktest_start else None,
                            window_dates.bcktest_end.date() if window_dates.bcktest_end else None,
                            model_name,
                            portfolio_type,
                        )
                    )
                    continue

                mu_p, sigma_p = compute_portfolio_performance(adjusted_weights, mu, cov)

                forecast = compute_forecast_annualization(
                    mu_p, sigma_p, rf, periods_per_year, config.bckTestWndw
                )

                if window.has_bcktest:
                    bck_dates = window_row_dates(
                        securities_data.dates, window.bcktest_start_row, window.bcktest_end_row
                    )
                    portfolio_bck_returns = _portfolio_bcktest_returns(
                        securities_data.returns, bck_dates, adjusted_weights
                    )
                    realized = compute_realized_performance(
                        portfolio_bck_returns.to_numpy(), periods_per_year
                    )
                    aligned_factors_bck = align_factor_window(factor_data.values, bck_dates, periods_per_year)
                    yr_real_rf = compute_realized_rf(aligned_factors_bck["RF"].to_numpy(), periods_per_year)
                    yr_real_shrp = compute_realized_sharpe(realized.YrRealR, yr_real_rf, realized.YrRealVol)
                else:
                    realized = NAN_REALIZED_PERFORMANCE
                    yr_real_rf = float("nan")
                    yr_real_shrp = float("nan")

                yr_del_exp_real = residual_del_exp_real(forecast.YrExpR, realized.YrRealR)
                bcktest_del_exp_real = residual_del_exp_real(forecast.bkTstExpR, realized.bkTstRealR)

                if window.has_bcktest and market_realized is not None:
                    yr_prem_real_mkt = residual_prem_real_mkt(realized.YrRealR, market_realized.YrRealR)
                    bcktest_prem_real_mkt = residual_prem_real_mkt(
                        realized.bkTstRealR, market_realized.bkTstRealR
                    )
                else:
                    yr_prem_real_mkt = float("nan")
                    bcktest_prem_real_mkt = float("nan")

                records.append(
                    OptimizationRecord(
                        perfWndwNum=window.perf_wndw_num,
                        PerfStart=window_dates.perf_start.date(),
                        PerfEnd=window_dates.perf_end.date(),
                        bkTstStart=(
                            window_dates.bcktest_start.date() if window_dates.bcktest_start else None
                        ),
                        bkTstEnd=window_dates.bcktest_end.date() if window_dates.bcktest_end else None,
                        Model=model_name,
                        PortfolioType=portfolio_type,
                        YrExpR=forecast.YrExpR,
                        bkTstExpR=forecast.bkTstExpR,
                        YrExpVol=forecast.YrExpVol,
                        bkTstExpVol=forecast.bkTstExpVol,
                        YrExpRF=forecast.YrExpRF,
                        YrExpShrp=forecast.YrExpShrp,
                        YrRealR=realized.YrRealR,
                        bkTstRealR=realized.bkTstRealR,
                        YrRealVol=realized.YrRealVol,
                        bkTstRealVol=realized.bkTstRealVol,
                        YrRealRF=yr_real_rf,
                        YrRealShrp=yr_real_shrp,
                        YrDelExpReal=yr_del_exp_real,
                        bkTstDelExpReal=bcktest_del_exp_real,
                        YrPremRealMkt=yr_prem_real_mkt,
                        bkTstPremRealMkt=bcktest_prem_real_mkt,
                        weights=adjusted_weights,
                    )
                )

    logger.debug("Finished optimization loop: %d record(s) built. Writing output files.", len(records))
    catalog_order = tuple(FACTOR_MODEL_CATALOG)
    output_dir.mkdir(parents=True, exist_ok=True)

    lng_pmod_pval, lng_pval_pmod = build_results_tables(records, LONG_PORTFOLIO_TYPE, catalog_order)
    lng_pmod_pval_path = output_dir / "OptResultsLngPmodPval.csv"
    lng_pval_pmod_path = output_dir / "OptResultsLngPvalPmod.csv"
    write_output_csv(lng_pmod_pval_path, lng_pmod_pval)
    write_output_csv(lng_pval_pmod_path, lng_pval_pmod)

    shrt_pmod_pval_path: Path | None = None
    shrt_pval_pmod_path: Path | None = None
    if config.shortPortfolio:
        shrt_pmod_pval, shrt_pval_pmod = build_results_tables(records, SHORT_PORTFOLIO_TYPE, catalog_order)
        shrt_pmod_pval_path = output_dir / "OptResultsShrtPmodPval.csv"
        shrt_pval_pmod_path = output_dir / "OptResultsShrtPvalPmod.csv"
        write_output_csv(shrt_pmod_pval_path, shrt_pmod_pval)
        write_output_csv(shrt_pval_pmod_path, shrt_pval_pmod)

    overview_lng = build_overview_table(records, LONG_PORTFOLIO_TYPE, catalog_order)
    overview_lng_path = output_dir / "OptResultsOverviewLng.csv"
    write_output_csv(overview_lng_path, overview_lng)

    overview_shrt_path: Path | None = None
    if config.shortPortfolio:
        overview_shrt = build_overview_table(records, SHORT_PORTFOLIO_TYPE, catalog_order)
        overview_shrt_path = output_dir / "OptResultsOverviewShrt.csv"
        write_output_csv(overview_shrt_path, overview_shrt)

    lng_weights_tables = build_weights_tables(
        records, LONG_PORTFOLIO_TYPE, tuple(securities_data.returns.columns), catalog_order
    )
    weights_lng_paths: dict[str, Path] = {}
    for model_name, weights_df in lng_weights_tables.items():
        weights_path = output_dir / f"WtsLng{model_name}.csv"
        write_output_csv(weights_path, weights_df)
        weights_lng_paths[model_name] = weights_path

    weights_shrt_paths: dict[str, Path] = {}
    if config.shortPortfolio:
        shrt_weights_tables = build_weights_tables(
            records, SHORT_PORTFOLIO_TYPE, tuple(securities_data.returns.columns), catalog_order
        )
        for model_name, weights_df in shrt_weights_tables.items():
            weights_path = output_dir / f"WtsShrt{model_name}.csv"
            write_output_csv(weights_path, weights_df)
            weights_shrt_paths[model_name] = weights_path

    sharpe_history = build_sharpe_history_table(records, catalog_order)
    sharpe_history_path = output_dir / "SharpeHistory.csv"
    write_output_csv(sharpe_history_path, sharpe_history)

    rendered_summary = summary.render(
        config,
        config_path,
        window_plan,
        securities_path,
        periods_per_year,
        ticker_count,
        securities_date_range,
        n_perf_rows,
        factors_path,
        factor_periods_per_year,
        factor_date_ranges,
        records,
        catalog_order,
    )
    run_summary_path = output_dir / f"run_summary_{timestamp_str}.log"
    write_run_summary(run_summary_path, rendered_summary)
    logger.debug("Wrote run summary to %s", run_summary_path)

    return RunResult(
        output_dir=output_dir,
        records=records,
        window_plan=window_plan,
        summary_text=rendered_summary,
        run_summary_path=run_summary_path,
        sharpe_history_path=sharpe_history_path,
        opt_results_lng_pmod_pval_path=lng_pmod_pval_path,
        opt_results_lng_pval_pmod_path=lng_pval_pmod_path,
        opt_results_shrt_pmod_pval_path=shrt_pmod_pval_path,
        opt_results_shrt_pval_pmod_path=shrt_pval_pmod_path,
        opt_results_overview_lng_path=overview_lng_path,
        opt_results_overview_shrt_path=overview_shrt_path,
        weights_lng_paths=weights_lng_paths,
        weights_shrt_paths=weights_shrt_paths,
    )
