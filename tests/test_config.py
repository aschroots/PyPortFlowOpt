from __future__ import annotations

import pytest
import yaml

from pyportflowopt.config import FACTOR_MODEL_CATALOG, load_config, market_proxy_display_name
from pyportflowopt.errors import ConfigError

_BASE_CONFIG = {
    "marketProxyTicker": "SPY",
    "marketProxyName": "S&P 500",
    "priceInputKind": "prices",
    "perfWndw": 10,
    "perfWndwShft": 5,
    "doBcktest": True,
    "bckTestWndw": 6,
    "shortPortfolio": True,
    "shortLimit": -0.3,
    "factorModels": ["FF3", "FFC4"],
}


def _write_config(tmp_path, overrides=None, remove=None):
    data = dict(_BASE_CONFIG)
    if remove:
        for key in remove:
            data.pop(key, None)
    if overrides:
        data.update(overrides)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_valid_config_round_trips(tmp_path):
    path = _write_config(tmp_path)
    config = load_config(path)
    assert config.marketProxyTicker == "SPY"
    assert config.marketProxyName == "S&P 500"
    assert config.priceInputKind == "prices"
    assert config.perfWndw == 10
    assert config.perfWndwShft == 5
    assert config.doBcktest is True
    assert config.bckTestWndw == 6
    assert config.shortPortfolio is True
    assert config.shortLimit == -0.3
    assert config.factorModels == ("FF3", "FFC4")


@pytest.mark.parametrize(
    "field", ["doBcktest", "shortPortfolio", "perfWndw", "perfWndwShft", "priceInputKind"]
)
def test_missing_required_field_is_fatal(tmp_path, field):
    path = _write_config(tmp_path, remove=[field])
    with pytest.raises(ConfigError):
        load_config(path)


@pytest.mark.parametrize("field", ["perfWndw", "perfWndwShft"])
def test_non_positive_int_fields_are_fatal(tmp_path, field):
    path = _write_config(tmp_path, overrides={field: 0})
    with pytest.raises(ConfigError):
        load_config(path)


def test_bool_fields_reject_non_bool(tmp_path):
    path = _write_config(tmp_path, overrides={"doBcktest": "yes"})
    with pytest.raises(ConfigError):
        load_config(path)


def test_price_input_kind_must_be_prices_or_returns(tmp_path):
    path = _write_config(tmp_path, overrides={"priceInputKind": "weird"})
    with pytest.raises(ConfigError):
        load_config(path)


def test_price_input_kind_returns_is_valid(tmp_path):
    path = _write_config(tmp_path, overrides={"priceInputKind": "returns"})
    config = load_config(path)
    assert config.priceInputKind == "returns"


def test_bcktest_wndw_required_when_do_bcktest_true(tmp_path):
    path = _write_config(tmp_path, remove=["bckTestWndw"])
    with pytest.raises(ConfigError):
        load_config(path)


def test_bcktest_wndw_omitted_when_do_bcktest_false_is_fine(tmp_path):
    path = _write_config(tmp_path, overrides={"doBcktest": False}, remove=["bckTestWndw"])
    config = load_config(path)
    assert config.bckTestWndw is None


def test_bcktest_wndw_validated_even_when_do_bcktest_false(tmp_path):
    # "validated whenever present" -- a bad value is still rejected even though it's optional.
    path = _write_config(tmp_path, overrides={"doBcktest": False, "bckTestWndw": -1})
    with pytest.raises(ConfigError):
        load_config(path)


def test_short_limit_required_when_short_portfolio_true(tmp_path):
    path = _write_config(tmp_path, remove=["shortLimit"])
    with pytest.raises(ConfigError):
        load_config(path)


def test_short_limit_omitted_when_short_portfolio_false_is_fine(tmp_path):
    path = _write_config(tmp_path, overrides={"shortPortfolio": False}, remove=["shortLimit"])
    config = load_config(path)
    assert config.shortLimit is None


@pytest.mark.parametrize("bad_value", [0.0, -1.0, -1.5, 0.5, 1.0])
def test_short_limit_out_of_range_is_fatal(tmp_path, bad_value):
    path = _write_config(tmp_path, overrides={"shortLimit": bad_value})
    if bad_value == 0.0:
        # 0.0 IS in the valid (-1, 0] range -- only assert the genuinely-out-of-range ones.
        config = load_config(path)
        assert config.shortLimit == 0.0
        return
    with pytest.raises(ConfigError):
        load_config(path)


def test_short_limit_minus_one_exclusive_is_fatal(tmp_path):
    path = _write_config(tmp_path, overrides={"shortLimit": -1.0})
    with pytest.raises(ConfigError):
        load_config(path)


def test_market_proxy_ticker_optional_defaults_to_none(tmp_path):
    path = _write_config(tmp_path, remove=["marketProxyTicker", "marketProxyName"])
    config = load_config(path)
    assert config.marketProxyTicker is None
    assert config.marketProxyName is None


def test_market_proxy_ticker_blank_string_becomes_none(tmp_path):
    path = _write_config(tmp_path, overrides={"marketProxyTicker": "  "})
    config = load_config(path)
    assert config.marketProxyTicker is None


def test_market_proxy_name_ignored_without_error_when_ticker_blank(tmp_path):
    path = _write_config(tmp_path, overrides={"marketProxyTicker": "", "marketProxyName": "S&P 500"})
    config = load_config(path)
    assert config.marketProxyTicker is None
    # marketProxyName is stored as given regardless -- only *consulted* when a ticker is set.
    assert config.marketProxyName == "S&P 500"


def test_market_proxy_display_name_none_when_no_ticker(tmp_path):
    path = _write_config(tmp_path, remove=["marketProxyTicker", "marketProxyName"])
    config = load_config(path)
    assert market_proxy_display_name(config) is None


def test_market_proxy_display_name_prefers_name_over_ticker(tmp_path):
    path = _write_config(tmp_path)
    config = load_config(path)
    assert market_proxy_display_name(config) == "S&P 500"


def test_market_proxy_display_name_falls_back_to_ticker(tmp_path):
    path = _write_config(tmp_path, remove=["marketProxyName"])
    config = load_config(path)
    assert market_proxy_display_name(config) == "SPY"


def test_factor_models_empty_list_is_valid(tmp_path):
    path = _write_config(tmp_path, overrides={"factorModels": []})
    config = load_config(path)
    assert config.factorModels == ()


def test_factor_models_missing_field_is_fatal(tmp_path):
    path = _write_config(tmp_path, remove=["factorModels"])
    with pytest.raises(ConfigError):
        load_config(path)


def test_factor_models_unrecognized_name_is_fatal(tmp_path):
    path = _write_config(tmp_path, overrides={"factorModels": ["FF3", "NOTAMODEL"]})
    with pytest.raises(ConfigError) as exc_info:
        load_config(path)
    assert "NOTAMODEL" in str(exc_info.value)


def test_factor_models_case_insensitive_matching_preserves_catalog_casing(tmp_path):
    path = _write_config(tmp_path, overrides={"factorModels": ["ff5mod", "capm"]})
    config = load_config(path)
    assert config.factorModels == ("FF5Mod", "CAPM")


def test_factor_models_case_insensitive_duplicate_is_fatal(tmp_path):
    path = _write_config(tmp_path, overrides={"factorModels": ["FF3", "ff3"]})
    with pytest.raises(ConfigError) as exc_info:
        load_config(path)
    assert "duplicate" in str(exc_info.value).lower()


def test_factor_models_not_a_list_is_fatal(tmp_path):
    path = _write_config(tmp_path, overrides={"factorModels": "FF3"})
    with pytest.raises(ConfigError):
        load_config(path)


def test_all_six_catalog_models_accepted(tmp_path):
    path = _write_config(tmp_path, overrides={"factorModels": list(FACTOR_MODEL_CATALOG)})
    config = load_config(path)
    assert set(config.factorModels) == set(FACTOR_MODEL_CATALOG)


def test_multiple_bad_fields_combined_into_one_error(tmp_path):
    path = _write_config(tmp_path, overrides={"perfWndw": -1, "perfWndwShft": -1})
    with pytest.raises(ConfigError) as exc_info:
        load_config(path)
    message = str(exc_info.value)
    assert "perfWndw" in message
    assert "perfWndwShft" in message


def test_config_file_not_a_mapping_is_fatal(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_config_valid_full_fixture_loads(fixtures_dir):
    config = load_config(fixtures_dir / "config_valid_full.yaml")
    assert config.doBcktest is True
    assert config.shortPortfolio is True
    assert config.marketProxyTicker == "SPY"
    assert config.factorModels == ("CAPM", "FF3", "FFC4", "FF5", "FF5Mod", "FFC6")


def test_config_valid_no_bcktest_fixture_loads(fixtures_dir):
    config = load_config(fixtures_dir / "config_valid_no_bcktest.yaml")
    assert config.doBcktest is False
    assert config.bckTestWndw is None
    assert config.marketProxyTicker is None


def test_config_invalid_missing_bcktest_wndw_fixture_is_fatal(fixtures_dir):
    with pytest.raises(ConfigError):
        load_config(fixtures_dir / "config_invalid_missing_bcktest_wndw.yaml")


def test_config_invalid_unknown_model_fixture_is_fatal(fixtures_dir):
    with pytest.raises(ConfigError) as exc_info:
        load_config(fixtures_dir / "config_invalid_unknown_model.yaml")
    assert "NOTAMODEL" in str(exc_info.value)
