"""Tests for the statistical layer.

`src/statistical/baselines.py` exists and is covered in test_baselines.py.
The Bayesian spatial model described by config/statistical.yaml does not exist
yet, so its tests are declared here and skip until it lands.

They are real skipped tests rather than commented-out code, so the intended
checks show up in `pytest -ra`, stay importable, and cannot silently rot.
"""
import importlib.util

import pytest

from src.utils.config import load_config

MODEL_MODULE = "src.statistical.spatial"

needs_model = pytest.mark.skipif(
    importlib.util.find_spec("src.statistical") is None
    or importlib.util.find_spec(MODEL_MODULE) is None,
    reason=f"{MODEL_MODULE} does not exist yet; see config/statistical.yaml",
)


# ==========================================================================
# configuration scaffolding - runs today
# ==========================================================================

def test_config_parses():
    assert isinstance(load_config("statistical.yaml"), dict)


@pytest.mark.parametrize("section", ["model", "priors", "sampling", "spatial"])
def test_config_declares_expected_sections(section):
    """Pins the section names the model code will read, so a rename shows up
    here rather than as a KeyError halfway through sampling."""
    assert section in load_config("statistical.yaml")


@pytest.mark.parametrize("prior", ["intercept", "slope", "spatial_smoothing"])
def test_config_declares_expected_priors(prior):
    assert prior in load_config("statistical.yaml")["priors"]


@pytest.mark.parametrize("key", ["draws", "tune", "chains", "target_accept"])
def test_config_declares_expected_sampling_keys(key):
    assert key in load_config("statistical.yaml")["sampling"]


# ==========================================================================
# the model itself - skipped until src/statistical/spatial.py exists
# ==========================================================================

@pytest.mark.notimplemented
@needs_model
def test_model_builds_without_error():
    """Instantiate from config/statistical.yaml on a small synthetic ZCTA
    panel and assert it compiles."""


@pytest.mark.notimplemented
@needs_model
def test_posterior_shape():
    """(chains, draws, n_params), matching config/statistical.yaml."""


@pytest.mark.notimplemented
@needs_model
def test_no_divergences():
    """Zero divergent transitions at the configured target_accept."""


@pytest.mark.notimplemented
@needs_model
def test_rhat_convergence():
    """All R-hat values < 1.01."""


@pytest.mark.notimplemented
@needs_model
def test_posterior_predictive_shape():
    """One predictive draw per observed ZCTA-month."""


@pytest.mark.notimplemented
@needs_model
def test_spatial_smoothing_applied():
    """Neighbouring ZCTAs shrink toward each other more than distant ones do."""


@pytest.mark.notimplemented
@needs_model
def test_predictions_non_negative():
    """These are counts, so the predictive distribution has no mass below zero."""
