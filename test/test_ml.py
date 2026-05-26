# import pytest
# import numpy as np
# import pandas as pd


# @pytest.fixture
# def ml_config():
#     # load from config/ml.yaml
#     pass

# @pytest.fixture
# def synthetic_features():
#     pass

# @pytest.fixture
# def synthetic_labels():
#     pass

# @pytest.fixture
# def trained_lightgbm(synthetic_features, synthetic_labels, ml_config):
#     # train minimal LightGBM from src/ml/
#     pass

# @pytest.fixture
# def trained_xgboost(synthetic_features, synthetic_labels, ml_config):
#     pass

# @pytest.fixture
# def trained_rf(synthetic_features, synthetic_labels, ml_config):
#     pass


# def test_lightgbm_trains_without_error(trained_lightgbm):
#     pass

# def test_xgboost_trains_without_error(trained_xgboost):
#     pass

# def test_rf_trains_without_error(trained_rf):
#     pass

# def test_predictions_are_probabilities(trained_lightgbm, synthetic_features):
#     pass

# def test_prediction_shape(trained_lightgbm, synthetic_features):
#     pass

# def test_no_nan_predictions(trained_lightgbm, synthetic_features):
#     pass

# def test_cv_folds_respected(synthetic_features, synthetic_labels, ml_config):
#     pass

# def test_shap_values_computable(trained_lightgbm, synthetic_features):
#     pass

# def test_feature_importance_non_empty(trained_lightgbm):
#     pass
