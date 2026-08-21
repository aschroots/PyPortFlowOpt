from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .errors import ConfigError

FACTOR_MODEL_CATALOG: dict[str, tuple[str, ...]] = {
    "CAPM": ("Mkt-RF",),
    "FF3": ("Mkt-RF", "SMB", "HML"),
    "FFC4": ("Mkt-RF", "SMB", "HML", "PR1YR"),
    "FF5": ("Mkt-RF", "SMB5Fac", "HML", "RMW", "CMA"),
    "FF5Mod": ("Mkt-RF", "SMB", "HML", "RMW", "CMA"),
    "FFC6": ("Mkt-RF", "SMB5Fac", "HML", "RMW", "CMA", "PR1YR"),
}
"""Dict insertion order is also the canonical model ordering used downstream for output
column/row sequencing -- 'Simp' always precedes this catalog's own order, independent of
the order a config's factorModels lists them in."""

_REQUIRED_BOOL_FIELDS = ("doBcktest", "shortPortfolio")
_REQUIRED_INT_FIELDS = ("perfWndw", "perfWndwShft")
_PRICE_INPUT_KINDS = ("prices", "returns")


@dataclass(frozen=True)
class PyPortFlowOptConfig:
    marketProxyTicker: str | None
    marketProxyName: str | None
    priceInputKind: str
    perfWndw: int
    perfWndwShft: int
    doBcktest: bool
    bckTestWndw: int | None
    shortPortfolio: bool
    shortLimit: float | None
    factorModels: tuple[str, ...]
    wtsEpsilon: float


def market_proxy_display_name(config: PyPortFlowOptConfig) -> str | None:
    """None when no market proxy is configured; otherwise marketProxyName if given, else
    the raw ticker -- the single source of truth every log/table/output should consult."""
    if config.marketProxyTicker is None:
        return None
    return config.marketProxyName or config.marketProxyTicker


def load_config(path: Path) -> PyPortFlowOptConfig:
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ConfigError(f"Config file {path} did not parse to a mapping of settings.")
    return _validate(raw)


def _validate_factor_models(raw: dict, errors: list[str]) -> tuple[str, ...]:
    if "factorModels" not in raw:
        errors.append("Missing required config field: 'factorModels'")
        return ()

    raw_factor_models = raw["factorModels"]
    if not isinstance(raw_factor_models, list):
        errors.append(f"'factorModels' must be a list of model name strings, got {raw_factor_models!r}")
        return ()

    # Case-insensitive lookup keyed by uppercase, but the VALUE preserves the catalog's
    # own defined casing (e.g. "FF5Mod") as the canonical form.
    catalog_lookup = {name.upper(): name for name in FACTOR_MODEL_CATALOG}

    canonical_names: list[str] = []
    invalid_names: list[object] = []
    duplicate_names: list[object] = []
    seen: set[str] = set()
    for entry in raw_factor_models:
        if not isinstance(entry, str) or entry.upper() not in catalog_lookup:
            invalid_names.append(entry)
            continue
        canonical = catalog_lookup[entry.upper()]
        if canonical in seen:
            duplicate_names.append(entry)
            continue
        seen.add(canonical)
        canonical_names.append(canonical)

    if invalid_names:
        errors.append(
            f"'factorModels' contains unrecognized model name(s): {invalid_names}. "
            f"Valid names: {sorted(FACTOR_MODEL_CATALOG)}"
        )
    if duplicate_names:
        errors.append(
            f"'factorModels' contains duplicate model name(s) (case-insensitive): {duplicate_names}"
        )

    return tuple(canonical_names)


def _validate_wts_epsilon(raw: dict, errors: list[str]) -> float:
    raw_value = raw.get("wtsEpsilon")
    if raw_value in (None, ""):
        return 0.0
    if not isinstance(raw_value, int | float) or isinstance(raw_value, bool) or raw_value < 0:
        errors.append(f"'wtsEpsilon' must be a non-negative number when provided, got {raw_value!r}")
        return 0.0
    return float(raw_value)


def _validate_optional_str(raw: dict, field_name: str, errors: list[str]) -> str | None:
    value = raw.get(field_name)
    if isinstance(value, str) and value.strip() == "":
        value = None
    if value is not None and not isinstance(value, str):
        errors.append(f"'{field_name}' must be a string when provided, got {value!r}")
        return None
    return value


def _validate(raw: dict) -> PyPortFlowOptConfig:
    errors: list[str] = []

    for field_name in (*_REQUIRED_BOOL_FIELDS, *_REQUIRED_INT_FIELDS, "priceInputKind"):
        if field_name not in raw:
            errors.append(f"Missing required config field: '{field_name}'")

    bools: dict[str, bool] = {}
    for field_name in _REQUIRED_BOOL_FIELDS:
        if field_name in raw:
            value = raw[field_name]
            if not isinstance(value, bool):
                errors.append(f"'{field_name}' must be a boolean (TRUE/FALSE), got {value!r}")
            else:
                bools[field_name] = value

    ints: dict[str, int] = {}
    for field_name in _REQUIRED_INT_FIELDS:
        if field_name in raw:
            value = raw[field_name]
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                errors.append(f"'{field_name}' must be a positive integer, got {value!r}")
            else:
                ints[field_name] = value

    price_input_kind: str | None = None
    if "priceInputKind" in raw:
        value = raw["priceInputKind"]
        if value not in _PRICE_INPUT_KINDS:
            errors.append(f"'priceInputKind' must be one of {_PRICE_INPUT_KINDS}, got {value!r}")
        else:
            price_input_kind = value

    do_bcktest = bools.get("doBcktest")
    bck_test_wndw: int | None = None
    raw_bck_test_wndw = raw.get("bckTestWndw")
    bck_test_wndw_required = do_bcktest is True
    if raw_bck_test_wndw in (None, ""):
        if bck_test_wndw_required:
            errors.append("'bckTestWndw' must be a positive integer when 'doBcktest' is TRUE")
    elif (
        not isinstance(raw_bck_test_wndw, int)
        or isinstance(raw_bck_test_wndw, bool)
        or raw_bck_test_wndw <= 0
    ):
        errors.append(f"'bckTestWndw' must be a positive integer when provided, got {raw_bck_test_wndw!r}")
    else:
        bck_test_wndw = raw_bck_test_wndw

    short_portfolio = bools.get("shortPortfolio")
    short_limit: float | None = None
    raw_short_limit = raw.get("shortLimit")
    short_limit_required = short_portfolio is True
    if raw_short_limit in (None, ""):
        if short_limit_required:
            errors.append("'shortLimit' must be a number in (-1, 0] when 'shortPortfolio' is TRUE")
    elif (
        not isinstance(raw_short_limit, int | float)
        or isinstance(raw_short_limit, bool)
        or not (-1 < raw_short_limit <= 0)
    ):
        errors.append(f"'shortLimit' must be a number in (-1, 0] when provided, got {raw_short_limit!r}")
    else:
        short_limit = float(raw_short_limit)

    factor_models = _validate_factor_models(raw, errors)

    market_proxy_ticker = _validate_optional_str(raw, "marketProxyTicker", errors)
    market_proxy_name = _validate_optional_str(raw, "marketProxyName", errors)

    wts_epsilon = _validate_wts_epsilon(raw, errors)

    if errors:
        raise ConfigError("Invalid configuration:\n" + "\n".join(f"  - {e}" for e in errors))

    return PyPortFlowOptConfig(
        marketProxyTicker=market_proxy_ticker,
        marketProxyName=market_proxy_name,
        priceInputKind=price_input_kind,
        perfWndw=ints["perfWndw"],
        perfWndwShft=ints["perfWndwShft"],
        doBcktest=bools["doBcktest"],
        bckTestWndw=bck_test_wndw,
        shortPortfolio=bools["shortPortfolio"],
        shortLimit=short_limit,
        factorModels=factor_models,
        wtsEpsilon=wts_epsilon,
    )
