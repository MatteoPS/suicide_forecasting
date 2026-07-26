"""Tests for the dynamical (contagion + EAKF) layer.

`src/dynamical/` does not exist yet; config/dynamical.yaml is empty
scaffolding, and notebooks/3.1.1 and 3.1.2 are MATLAB replications of the
published figures rather than Python code.

The config tests run today. The model tests skip until src/dynamical/ lands.
"""
import importlib.util

import pytest

from src.utils.config import load_config

needs_model = pytest.mark.skipif(
    importlib.util.find_spec("src.dynamical") is None,
    reason="src/dynamical does not exist yet; see config/dynamical.yaml",
)


# ==========================================================================
# configuration scaffolding - runs today
# ==========================================================================

def test_config_parses():
    assert isinstance(load_config("dynamical.yaml"), dict)


@pytest.mark.parametrize(
    "section", ["contagion_model", "eakf", "assimilation", "forecast"]
)
def test_config_declares_expected_sections(section):
    assert section in load_config("dynamical.yaml")


@pytest.mark.parametrize(
    "key", ["compartments", "beta_init", "gamma_init", "population_scale"]
)
def test_contagion_model_keys(key):
    assert key in load_config("dynamical.yaml")["contagion_model"]


@pytest.mark.parametrize(
    "key", ["ensemble_size", "inflation_factor", "localization_radius"]
)
def test_eakf_keys(key):
    """Inflation and localization are what keep an EAKF from collapsing; both
    have to be configurable, not hard-coded."""
    assert key in load_config("dynamical.yaml")["eakf"]


@pytest.mark.parametrize(
    "key", ["horizon_months", "n_ensemble_members", "output_quantiles"]
)
def test_forecast_keys(key):
    assert key in load_config("dynamical.yaml")["forecast"]


# ==========================================================================
# the model itself - skipped until src/dynamical/ exists
# ==========================================================================

@pytest.mark.notimplemented
@needs_model
def test_model_initializes_without_error():
    """Build the compartmental model from config/dynamical.yaml."""


@pytest.mark.notimplemented
@needs_model
def test_eakf_runs_without_error():
    """One assimilation cycle against a short synthetic observation series."""


@pytest.mark.notimplemented
@needs_model
def test_state_estimates_finite():
    """No NaN or inf in the posterior state - the usual sign of filter
    divergence."""


@pytest.mark.notimplemented
@needs_model
def test_no_negative_compartments():
    """Compartments are population counts and cannot go below zero."""


@pytest.mark.notimplemented
@needs_model
def test_ensemble_spread_nonzero():
    """A collapsed ensemble reports false confidence."""


@pytest.mark.notimplemented
@needs_model
def test_forecast_shape():
    """(n_ensemble_members, horizon_months), per config/dynamical.yaml."""


@pytest.mark.notimplemented
@needs_model
def test_forecast_values_finite():
    """No NaN or inf in the projected trajectories."""


@pytest.mark.notimplemented
@needs_model
def test_parameter_estimates_in_bounds():
    """beta and gamma stay inside an epidemiologically plausible range."""


@pytest.mark.notimplemented
@needs_model
def test_assimilation_reduces_spread():
    """Observations must shrink the posterior relative to the prior; if they do
    not, the filter is not assimilating anything."""
