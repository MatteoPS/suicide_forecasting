"""Tests for the ensemble-combination layer.

`src/ensemble/` does not exist yet and config/ensemble.yaml is empty
scaffolding. The config tests run today; the combination tests skip until the
module lands.
"""
import importlib.util

import pytest

from src.utils.config import load_config

needs_ensemble = pytest.mark.skipif(
    importlib.util.find_spec("src.ensemble") is None,
    reason="src/ensemble does not exist yet; see config/ensemble.yaml",
)


# ==========================================================================
# configuration scaffolding - runs today
# ==========================================================================

def test_config_parses():
    assert isinstance(load_config("ensemble.yaml"), dict)


@pytest.mark.parametrize(
    "section", ["models_to_combine", "weighting", "scoring", "output"]
)
def test_config_declares_expected_sections(section):
    assert section in load_config("ensemble.yaml")


@pytest.mark.parametrize("key", ["strategy", "metric", "window_size"])
def test_weighting_keys(key):
    assert key in load_config("ensemble.yaml")["weighting"]


@pytest.mark.parametrize("key", ["rules", "horizons_months"])
def test_scoring_keys(key):
    assert key in load_config("ensemble.yaml")["scoring"]


@pytest.mark.parametrize("key", ["quantiles", "point_estimate"])
def test_output_keys(key):
    assert key in load_config("ensemble.yaml")["output"]


# ==========================================================================
# combination - skipped until src/ensemble/ exists
# ==========================================================================

@pytest.mark.notimplemented
@needs_ensemble
def test_ensemble_runs_without_error():
    """Combine a dict of {model name: predictions} per config/ensemble.yaml."""


@pytest.mark.notimplemented
@needs_ensemble
def test_output_shape_matches_input():
    """One combined value per horizon per location."""


@pytest.mark.notimplemented
@needs_ensemble
def test_no_nan_in_combined_forecast():
    """One member dropping out must not NaN the whole combination."""


@pytest.mark.notimplemented
@needs_ensemble
def test_weighted_average_sums_to_one():
    """Weights that do not sum to 1 silently rescale the forecast."""


@pytest.mark.notimplemented
@needs_ensemble
def test_combined_forecast_within_model_range():
    """A linear pool cannot fall outside the envelope of its members."""


@pytest.mark.notimplemented
@needs_ensemble
def test_scoring_rules_computable():
    """WIS and MAE, per config/ensemble.yaml scoring.rules."""


@pytest.mark.notimplemented
@needs_ensemble
def test_quantile_order_preserved():
    """Non-monotone quantiles are the classic quantile-crossing bug."""


@pytest.mark.notimplemented
@needs_ensemble
def test_point_estimate_matches_config():
    """Mean or median, per config/ensemble.yaml output.point_estimate."""


@pytest.mark.notimplemented
@needs_ensemble
def test_all_models_contributed():
    """Every model in models_to_combine has a non-zero weight, so a typo in a
    model name does not silently drop a member."""
