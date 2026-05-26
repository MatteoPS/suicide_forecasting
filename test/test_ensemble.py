import pytest
import numpy as np


@pytest.fixture
def ensemble_config():
    # load from config/ensemble.yaml
    pass

@pytest.fixture
def mock_model_forecasts():
    # dict of {model_name: array of predictions}
    pass

@pytest.fixture
def combined_forecast(mock_model_forecasts, ensemble_config):
    # run ensemble combination from src/ensemble/
    pass


def test_ensemble_runs_without_error(combined_forecast):
    pass

def test_output_shape_matches_input(combined_forecast, mock_model_forecasts):
    pass

def test_no_nan_in_combined_forecast(combined_forecast):
    pass

def test_weighted_average_sums_to_one(ensemble_config):
    pass

def test_combined_forecast_within_model_range(combined_forecast, mock_model_forecasts):
    pass

def test_scoring_rules_computable(combined_forecast):
    pass

def test_quantile_order_preserved(combined_forecast):
    pass

def test_point_estimate_matches_config(combined_forecast, ensemble_config):
    pass

def test_all_models_contributed(combined_forecast, ensemble_config):
    pass
