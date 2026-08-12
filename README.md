# PyPortFlowOpt

A rolling, multi-model portfolio optimization and backtesting tool. Given a securities
price/return history and a factor-returns file, it repeatedly optimizes a portfolio (via
`pyportfolioopt`'s max-Sharpe/tangent-portfolio solver) over a shifting window of history,
then backtests each optimized portfolio against the real returns that followed — for every
expected-return model you select: `Simp` (simple historical average) plus any of PyPortFlow2's
Fama-French-style factor models (CAPM, FF3, FFC4, FF5, FF5Mod, FFC6).

Everything is driven by a YAML config file, so the same code can be re-run across different
window sizes, backtest horizons, and factor-model combinations without code changes.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux
pip install -e ".[dev]"
```

This installs the `pyportflowopt` console command along with its runtime dependencies
(`pandas`, `numpy`, `PyYAML`, `pyportfolioopt`, `cvxpy`) and dev/test dependencies
(`pytest`, `pytest-cov`).

## Usage

```bash
pyportflowopt \
  --config config.yaml \
  --securities prices.csv \
  --factors factors.csv \
  --output-dir output/ \
  --log-level INFO
```

- `--config`, `--securities`, `--factors`, `--output-dir` are all required.
- `--log-level`: `DEBUG`, `INFO` (default), `WARNING`, or `ERROR`.
- Whether `--securities` holds prices or already-computed returns is **not** a CLI flag —
  it's the `priceInputKind` config field (see below).

Each run creates its own `runOpt_YYYYMMDD_HHMMSS/` subfolder under `--output-dir` (mirroring
PyPortFlow1's `run_<timestamp>/` and PyPortFlow2's `runStats_<timestamp>/` convention), so
repeated runs into the same `--output-dir` never clobber a prior run's outputs.

## Input files

**Securities price/return CSV** — column 1 is a row number, column 2 is the date, and column
3 onward is one ticker per column (ticker symbols as the header row) — the same shape as
PyPortFlow1's `mgoodhist.csv`/`mgoodperf.csv` and PyPortFlow2's `--prices` input. Values can
be raw prices or already-computed simple returns; tell the tool which via the config's
`priceInputKind`. Dates can be in any order in the file — they're re-sorted chronologically on
load, and window row-numbering (`perfWndwNum`, `PerfStart`/`PerfEnd`, etc.) always follows that
chronological order, never the file's own `Row` column values.

**Factor/risk-free-rate CSV** — same conventions as PyPortFlow2's factor file: column 1 header
is `Date`, remaining columns matched by header name (not position) from
`Mkt-RF, SMB, SMB5Fac, HML, RMW, CMA, PR1YR, RF`, values in percent (Ken French convention),
normalized to decimal internally. `RF` is always required — even a `Simp`-only run needs it,
since the tangent-portfolio optimization needs a risk-free rate. The factor file's own detected
periods-per-year must match the securities file's; a mismatch is a fatal error.

**Config YAML** — see fields below.

## Config fields

| Field | Type | Description |
|---|---|---|
| `marketProxyTicker` | string, optional | Ticker of the market-proxy benchmark column in the securities CSV. Leave blank/omit to run with no market proxy at all — `YrPremRealMkt`/`bckTestPremRealMkt` are then `NaN` throughout and omitted from the summary. If set, must be found as a column in the securities CSV. |
| `marketProxyName` | string, optional | Display name used instead of the ticker in all logs/tables when `marketProxyTicker` is set. Ignored if `marketProxyTicker` is blank. |
| `priceInputKind` | `prices` \| `returns` | Whether the securities CSV holds raw prices (converted to simple returns internally) or already-computed returns. Always explicit, never auto-detected. |
| `perfWndw` | positive int | Number of rows of price-performance history in each optimization window. |
| `perfWndwShft` | positive int | Number of rows the window shifts forward between backtests. |
| `doBcktest` | bool | Whether to run the rolling backtest series. Must be exactly `TRUE`/`FALSE` — anything else is a fatal config error. |
| `bckTestWndw` | positive int | Number of rows in each backtest window. Required iff `doBcktest: true`. |
| `shortPortfolio` | bool | Whether to also compute a short-allowed portfolio alongside the default long-only one. Must be exactly `TRUE`/`FALSE`. |
| `shortLimit` | number in `(-1, 0]` | Maximum short position allowed on any one security. Required iff `shortPortfolio: true`. |
| `factorModels` | list of strings | Named factor models to run in addition to the always-included `Simp` model, e.g. `[CAPM, FF3]`. `[]` runs `Simp` only. Names are case-insensitive and must come from the catalog below; duplicates are rejected. |

### Factor model catalog

Each selected model runs as its own independent regression per ticker. In every output file,
models always appear in this table's fixed order (`Simp` first, then this order), regardless
of the order they're listed in `factorModels`.

| Name | Factors |
|---|---|
| `CAPM` | `Mkt-RF` |
| `FF3` | `Mkt-RF, SMB, HML` |
| `FFC4` | `Mkt-RF, SMB, HML, PR1YR` |
| `FF5` | `Mkt-RF, SMB5Fac, HML, RMW, CMA` |
| `FF5Mod` | `Mkt-RF, SMB, HML, RMW, CMA` |
| `FFC6` | `Mkt-RF, SMB5Fac, HML, RMW, CMA, PR1YR` |

Betas are estimated over each window's own `perfWndw` history; each factor's *expected*
return is instead a long-run average — `FF5Mod` averages each of its factors independently
over its own available history, every other model averages all of its factors over one
shared common-start date (the latest, most restrictive, of each factor's own oldest available
date). See `factor_models.py` for the exact mechanics.

## Windowing and backtesting

`perfWndw` rows are used to optimize a portfolio (compute expected returns and a covariance
matrix, then find the max-Sharpe/tangent portfolio for the risk-free rate in effect at that
point in history). When `doBcktest: true`, the following `bckTestWndw` rows are used to
measure how that portfolio actually performed, and the window then shifts forward by
`perfWndwShft` rows and repeats. If the history doesn't divide evenly into shift-sized steps,
one final backtest window is anchored to the most recent data instead of following
`perfWndwShft` — noted in the run summary. When `doBcktest: false`, the same shifting
`perfWndw` windows are still produced (for their `Exp*` forecast columns), just without any
backtest measurement. Either way, one additional window — labeled `Final` in `perfWndwNum` —
always uses the most recent `perfWndw` rows to produce a live forecast with no backtest.

A ticker with a gap anywhere in a given window's date range is excluded from just that
window's optimization (noted in the run summary), not the whole run.

### Annualization

Every reported figure comes in an annualized (`Yr*`) and a `bckTestWndw`-scaled (`bckTest*`)
form. The two sides of the comparison are deliberately **not** both compounded the same way:

- **Forecast (`*Exp*`) figures scale arithmetically** — annualized expected return is the
  window's per-period expected return times periods-per-year, not geometrically compounded.
  This preserves mean-variance optimization's requirement that a portfolio's expected return
  is a linear combination of its assets' expected returns.
- **Realized (`*Real*`) figures compound geometrically** — the backtest window's actual
  per-period returns are compounded into a total return, then CAGR-annualized. This reflects
  what actually happened, not a modeling convenience.

`YrDelForeReal`/`bckTestDelForeReal` are the resulting forecast-minus-realized residual;
`YrPremRealMkt`/`bckTestPremRealMkt` are the portfolio's realized return minus the market
proxy's realized return (both `NaN` when no `marketProxyTicker` is configured).

## Outputs

Written into `--output-dir/runOpt_YYYYMMDD_HHMMSS/` (see the note above):

- `OptResultsLngPmodPval.csv` / `OptResultsLngPvalPmod.csv` — one row per window
  (`perfWndwNum, PerfStart, PerfEnd, bckTestStart, bckTestEnd`), then 13 value columns per
  model (`<model>YrExpR`, `<model>bckTestExpR`, `<model>YrExpVol`, `<model>bckTestExpVol`,
  `<model>YrExpShrp`, `<model>YrRealR`, `<model>bckTestRealR`, `<model>YrRealVol`,
  `<model>bckTestRealVol`, `<model>YrDelForeReal`, `<model>bckTestDelForeReal`,
  `<model>YrPremRealMkt`, `<model>bckTestPremRealMkt`). Same data in both files — `PmodPval`
  groups all 13 columns per model together; `PvalPmod` groups all models' values of the same
  metric together.
- `OptResultsShrtPmodPval.csv` / `OptResultsShrtPvalPmod.csv` — same shape, for the
  short-allowed portfolio. Only written when `shortPortfolio: true`.
- `SharpeHistory.csv` — the same id columns, then `<model>LngShrp` per model (and
  `<model>ShrtShrp` per model if `shortPortfolio: true`) — each window's annualized
  `YrExpShrp`.
- `run.log` — full DEBUG-level trace of the run (root logger, so `cvxpy`/`pyportfolioopt`
  solver warnings are captured too).
- `run_summary_YYYYMMDD_HHMMSS.log` — a `PyPortFlowOpt Run Summary` title followed by
  `Configuration`, `Input Securities Summary`, `Input Factors Summary`, and (when
  `doBcktest: true`) `Output Summary` sections (per-model max Sharpe ratio and its window,
  plus mean/stdev of `YrDelForeReal` and, if a market proxy is configured, `YrPremRealMkt`,
  across all backtest windows), plus any `Notes`/`Flags` raised during the run (e.g. a ticker
  excluded from a window's coverage, or an optimization that failed because no asset beat the
  risk-free rate).

## Tests

```bash
pytest
```
