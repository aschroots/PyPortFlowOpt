# PyPortFlowOpt code style

No code exists in this project yet. This carries forward the conventions from the
sibling PyPortFlow2 project (which this project builds on) as a starting point — revise
it once PyPortFlowOpt's actual architecture takes shape.

Unlike PyPortFlow1/PyPortFlow2, the mechanical/syntactic side of style here (quotes,
import sorting/grouping, line length, unused imports, pathlib-over-`os.path`, modern
`X | None` syntax) is **enforced by `ruff.toml`**, not documented — run
`ruff check --fix .` and `ruff format .` instead of relying on manual discipline. What
follows is only the architectural/semantic conventions no linter can check.

## Types and data structures

- Structured return values should be dataclasses, not tuples or dicts.
- Default to `@dataclass(frozen=True)` (immutable) unless a value specifically needs to
  be mutated after construction. This is PyPortFlow2's convention — PyPortFlow1's
  dataclasses are mutable by default; don't carry that habit over here.
- Prefer `typing.Protocol` over an ABC base class when what's needed is a duck-typed
  interface rather than shared implementation (see PyPortFlow2's
  `data_loading.py:TabularSeriesLoader` for the pattern).

## Errors

- One exception class per failure mode, all rooted in a single project-wide base
  exception, living in one `errors.py`. Each subclass body should just be a one-line
  docstring — no custom `__init__`/message-formatting logic.
- Keep a "fatal, halts before any output is written" subclass separate from
  recoverable/expected conditions, mirroring PyPortFlow2's `FatalPipelineError`.

## Logging and run reporting

- Prefer PyPortFlow2's pattern over ad hoc logging: accumulate non-fatal notes/warnings
  into a single run-summary object (see `logging_utils.py:RunSummary` in PyPortFlow2)
  and render them into one summary log at the end of a run, rather than scattering
  `logger.warning()` calls across modules. PyPortFlow2 also instantiates only one
  logger, in `cli.py` — worth keeping to if this project ends up with a similar CLI
  shape.

## Docstrings and comments

- Docstrings are reserved for non-obvious behavior — a hidden constraint, an invariant,
  a workaround, something that would surprise a reader. Straightforward functions get
  none. Ruff's docstring rules (`D`) are deliberately not enabled in `ruff.toml`, since
  they'd force a docstring onto every public function/class, which contradicts this.
- Inline `#` comments follow the same rule: explain *why*, not what the next line
  already says.

## Interop with PyPortFlow1/PyPortFlow2

- If this project produces or consumes CSVs shaped like PyPortFlow1's
  `mgoodhist.csv`/`mgoodperf.csv` or PyPortFlow2's price/return input, match those
  column shapes exactly (`Row, Date`, then one column per ticker). The three projects
  interoperate by convention, not a shared schema module — there's nothing that will
  catch a mismatch except a downstream failure.
