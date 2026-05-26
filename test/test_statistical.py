import pytest
import numpy as np


@pytest.fixture
def model_config():
    # load from config/statistical.yaml
    pass

@pytest.fixture
def small_data():
    # minimal synthetic dataset to run model fast
    pass

@pytest.fixture
def fitted_model(small_data, model_config):
    # instantiate and fit model from src/statistical/
    pass


def test_model_builds_without_error(small_data, model_config):
    pass

def test_posterior_shape(fitted_model):
    pass

def test_no_divergences(fitted_model):
    pass

def test_rhat_convergence(fitted_model):
    # all R-hat values < 1.01
    pass

def test_posterior_predictive_shape(fitted_model, small_data):
    pass

def test_spatial_smoothing_applied(fitted_model):
    pass

def test_predictions_non_negative(fitted_model):
    pass
