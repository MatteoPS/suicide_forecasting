"""Tests for the machine-learning layer.

`src/ml/` does not exist yet. config/ml.yaml is scaffolding for it, and the
LightGBM baseline currently lives in `src/statistical/baselines.py` (tested in
test_baselines.py) because notebooks/0.1 compares it against ETS and ARIMA.

The config tests below run today. The model tests skip until src/ml/ lands.
"""
import importlib.util

import pytest

from src.utils.config import load_config

needs_ml = pytest.mark.skipif(
    importlib.util.find_spec("src.ml") is None,
    reason="src/ml does not exist yet; see config/ml.yaml and docs/final_repo_structure_dummy.md",
)


# ==========================================================================
# configuration scaffolding - runs today
# ==========================================================================

def test_config_parses():
    assert isinstance(load_config("ml.yaml"), dict)


@pytest.mark.parametrize(
    "section",
    ["cross_validation", "lightgbm", "xgboost", "catboost", "random_forest",
     "neural_network"],
)
def test_config_declares_expected_sections(section):
    assert section in load_config("ml.yaml")


@pytest.mark.parametrize("key", ["folds", "strategy", "metric"])
def test_cross_validation_keys(key):
    """Time-series data needs an explicit CV strategy; a missing key here is
    how a random split - and leakage - sneaks in."""
    assert key in load_config("ml.yaml")["cross_validation"]


@pytest.mark.parametrize(
    "key", ["n_estimators", "learning_rate", "max_depth", "num_leaves"]
)
def test_lightgbm_keys(key):
    assert key in load_config("ml.yaml")["lightgbm"]


def test_every_tree_model_declares_a_learning_rate():
    conf = load_config("ml.yaml")
    for model in ("lightgbm", "xgboost", "catboost"):
        assert "learning_rate" in conf[model], model


# ==========================================================================
# the models themselves - skipped until src/ml/ exists
# ==========================================================================

@pytest.mark.notimplemented
@needs_ml
def test_lightgbm_trains_without_error():
    """Fit on synthetic features/labels using config/ml.yaml."""


@pytest.mark.notimplemented
@needs_ml
def test_xgboost_trains_without_error():
    """As above, for the XGBoost configuration."""


@pytest.mark.notimplemented
@needs_ml
def test_rf_trains_without_error():
    """As above, for the random-forest configuration."""


@pytest.mark.notimplemented
@needs_ml
def test_predictions_are_probabilities():
    """Every prediction in [0, 1] for the classification framing."""


@pytest.mark.notimplemented
@needs_ml
def test_prediction_shape():
    """One prediction per input row."""


@pytest.mark.notimplemented
@needs_ml
def test_no_nan_predictions():
    """NaN features must not propagate into predictions."""


@pytest.mark.notimplemented
@needs_ml
def test_cv_folds_respected():
    """The number of fits matches config/ml.yaml cross_validation.folds, and no
    fold's validation window precedes its training window."""


@pytest.mark.notimplemented
@needs_ml
def test_shap_values_computable():
    """SHAP values are the interpretability story for a surveillance model."""


@pytest.mark.notimplemented
@needs_ml
def test_feature_importance_non_empty():
    """A model that assigns zero importance to every feature has not learned."""
