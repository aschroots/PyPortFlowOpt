# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@code-style.md

## What this is

PyPortFlowOpt (package `pyportflowopt`, console script `pyportflowopt`): a rolling,
multi-model portfolio optimization and backtesting tool. Given a securities price/return
history and a factor-returns file, it repeatedly optimizes a portfolio (`pyportfolioopt`'s
max-Sharpe/tangent-portfolio solver) over a shifting window of history, then backtests each
optimized portfolio against the real returns that followed — for every expected-return
model selected: `Simp` (simple historical average) plus any of PyPortFlow2's Fama-French-style
factor models (CAPM, FF3, FFC4, FF5, FF5Mod, FFC6). Everything is driven by a YAML config, so
the same code re-runs across different window sizes, backtest horizons, and factor-model
combinations with no code changes.

This repo normally sits alongside two sibling projects: **PyPortFlow1** (package `pyportflow`,
retrieves/shapes the price CSVs this project can consume) and **PyPortFlow2** (package
`pyportflowstats`, whose factor-regression/covariance/frequency-detection logic this project
ports and adapts — see "Interop" below). If you're working in a checkout where those siblings
aren't present, the interop notes are still useful background but nothing here imports from
either at runtime — PyPortFlowOpt has no installed dependency on PyPortFlow2; shared logic was
ported by copy-and-adapt, not imported, since PyPortFlow2's annualization behavior needed to
change for this project's needs.

## Commands

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -e ".[dev]"     # installs numpy, pandas, PyYAML, pyportfolioopt, cvxpy + pytest, pytest-cov

pytest                                               # full suite
pytest tests/test_windowing.py                       # single file
pytest tests/test_windowing.py::test_name -v         # single test

pyportflowopt \
  --config config.yaml \
  --securities prices.csv \
  --factors factors.csv \
  --output-dir output/ \
  --log-level INFO
```

`priceInputKind` (`prices` | `returns`) lives in the config file, not as a CLI flag —
deliberately, unlike PyPortFlow2's `--price-input-kind` (see `README.md`'s Usage section for
the reasoning). `--factors` is always required, even for a `Simp`-only run — `max_sharpe()`
needs a risk-free rate from the factor file's `RF` column regardless of which expected-return
model is in use.

## Architecture

`cli.py:main` is a thin wrapper: build the run's own `runOpt_<timestamp>/` subfolder under
`--output-dir`, configure logging into it, `config.py:load_config`, then hand off to
`pipeline.py:run`, the single orchestration point. Read `pipeline.py` first when tracing
behavior — it calls into every other module in sequence.

1. **`config.py`** — loads/validates the YAML into a frozen `PyPortFlowOptConfig`,
   hand-validated the same way PyPortFlow2's `config.py` is (no schema library).
   `FACTOR_MODEL_CATALOG`'s insertion order (ported verbatim from PyPortFlow2) is
   load-bearing — canonical model ordering for every output. `marketProxyTicker`/
   `marketProxyName` are genuinely optional (`None` when unset, not conditionally-required
   like `bckTestWndw`/`shortLimit`); `market_proxy_display_name()` is the single place that
   resolves which of the two to show.
2. **`data_loading.py`** — loads the securities CSV (splitting out the market-proxy column
   when configured) and the factor CSV (`RF` always required). Sorts the securities data
   ascending by date regardless of the input file's own row order, and the internal 1-based
   window row-index is purely positional over that order — never derived from a "Row" column
   value.
3. **`returns.py`** — `prices_to_simple_returns`, `detect_periods_per_year` (verbatim port of
   PyPortFlow2's median day-gap banding). The factor file's own detected periods-per-year
   must equal the securities file's or the run fails fatally (`FrequencyMismatchError`) at
   load time.
4. **`windowing.py`** — generates the rolling `perfWndw`/`bckTestWndw` backtest windows
   (`generate_backtest_windows`), shifting by `perfWndwShft`, with one final window anchored
   to the most recent data when `(nPerfRows - perfWndw - bckTestWndw) % perfWndwShft != 0`;
   `generate_shift_only_windows` handles `doBcktest: false`; `generate_final_window` always
   produces one additional live-forecast row (`perfWndwNum = "Final"`), independent of
   `doBcktest`. Also hosts `check_window_coverage` (ported from PyPortFlow2's `stats.py`) — a
   ticker with a gap anywhere in a window's date span (perf + bcktest combined, when a
   backtest slice exists) is excluded from just that window, not the whole run
   (`CoverageExhaustedError` only if that leaves a window's universe empty). See
   `tests/test_windowing.py` for the worked numeric example (`nPerfRows=40, perfWndw=10,
   bckTestWndw=6, perfWndwShft=10` → 4 backtests, the 4th anchored) if the row-index
   arithmetic isn't obvious from the code.
5. **`factor_models.py`** — `factors.py`-equivalent logic ported from PyPortFlow2, with every
   annualization multiplication stripped (alpha, long-run factor expected returns, RF) so
   everything stays period-native — the optimizer needs `mu`/covariance in matching units.
   `latest_value_in_window` replaces PyPortFlow2's `get_annualized_value_as_of`: bounded
   strictly to a window's own date range (not an unbounded look-back), a deliberate
   behavioral deviation. `align_factor_window` is called once per window (dozens of times per
   run), unlike PyPortFlow2's single global alignment.
6. **`optimization.py`** — `compute_simp_mu`/`compute_factor_model_mu` build the
   expected-return vector; `optimize_portfolio` calls `pypfopt.EfficientFrontier(mu, cov,
   weight_bounds=...).max_sharpe(risk_free_rate=rf)` directly — this *is* the
   tangent-portfolio computation. **Never call `pypfopt.expected_returns.*`/`risk_models.*`**
   — `mu`/`cov` are always precomputed and passed in, both to avoid pypfopt's default
   geometric-compounding annualization (confined entirely to those unused convenience
   modules) and because they aren't multi-factor-model aware. A failed optimization (no asset
   beats `rf`, or the solver doesn't converge) is a routine `OptimizationOutcome(success=False,
   ...)`, never an exception.
7. **`annualization.py`** — the report-layer scaling, deliberately asymmetric: forecast
   (`*Exp*`) figures scale **arithmetically** (`mu_p * periods_per_year`) to preserve MVO's
   requirement that portfolio expected return is a linear combination of asset expected
   returns; realized (`*Real*`) figures **compound geometrically** then CAGR-annualize, since
   they measure what actually happened. Don't "fix" this asymmetry — it's the whole point.
8. **`results.py`** — builds one internal tidy `OptimizationRecord` per (window, model,
   portfolio-type); `build_results_tables` computes one wide table and derives both
   `PmodPval`/`PvalPmod` column orderings as pure reorders of it, never two independent
   pivots.
9. **`output_writing.py`** — thin `.to_csv` wrappers only; all the shape logic lives in
   `results.py`.
10. **`run_summary.py`** — `RunSummary` (ported from PyPortFlow2's `logging_utils.py`)
    accumulates `.note()`/`.flag()` calls through the run and renders the timestamped
    `run_summary_<timestamp>.log` in one pass; every `.flag()` call is paired with a
    `logger.warning()` call at the same call site so the event lands in both `run.log` and
    the summary.
11. **`logging_setup.py`** — root-logger, dual-`Formatter` (console: no timestamp, file:
    timestamped), idempotent handler setup, ported from PyPortFlow1's `logging_setup.py` —
    configuring the root logger (not just this package's own logger) means `cvxpy`/`pypfopt`
    solver warnings land in `run.log` too.

`errors.py`'s `FatalPipelineError` subclasses (`ConfigError`, `DataLoadError`,
`DuplicateTickerError`, `MarketProxyNotFoundError`, `FrequencyMismatchError`,
`InsufficientHistoryError`, `FactorAlignmentError`, `CoverageExhaustedError`) halt the run
before any output is written; `cli.py` exits 2 for `ConfigError`, 1 for any other
`FatalPipelineError`.

## Interop with PyPortFlow1 / PyPortFlow2

- PyPortFlow1's `mgoodhist.csv`/`mgoodperf.csv` (`Row, Date`, then one column per ticker) are
  valid `--securities` input — PyPortFlowOpt sorts ascending by date itself, so PyPortFlow1's
  most-recent-first row order is fine as-is.
- PyPortFlowOpt's factor CSV follows the exact same conventions as PyPortFlow2's `--factors`
  input (same column whitelist, same Ken French percent-to-decimal normalization, same
  monthly `(year, month)` date-alignment rule for `periods_per_year == 12`) — see PyPortFlow2's
  own `CLAUDE.md` for the reasoning behind the date-alignment nuance if it comes up.
- PyPortFlow2 is **not** an installed dependency — it's a sibling checkout, not published
  anywhere installable, and its `factors.py` needed its annualization behavior changed for
  this project anyway. Shared logic was ported by copy-and-adapt into this project's own
  modules (`returns.py`, `covariance.py`, `factor_models.py`, `data_loading.py`'s loader
  primitives, `windowing.py`'s `check_window_coverage`), not imported — check both copies
  stay in sync (or deliberately diverge, and note why) if you fix a bug in the ported logic
  here or in PyPortFlow2 itself.
- All three projects render their summary log in the same broad convention (title +
  `=`-underline, flush-left section headers, `Input File: <filename>` per section) — this
  project's version additionally timestamps the filename itself
  (`run_summary_<timestamp>.log`, inside its own `runOpt_<timestamp>/` subfolder), matching
  PyPortFlow1's `summary_<timestamp>.log` naming rather than PyPortFlow2's fixed
  `run_summary.log`.
