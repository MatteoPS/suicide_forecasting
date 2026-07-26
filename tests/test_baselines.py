"""Tests for `src.statistical.baselines`.

The models themselves belong to darts and are not retested here. What is worth
guarding is the plumbing the notebook used to do inline: the on-disk cache
(which will happily hand back a stale prediction) and the shape of the result
dict that `plot_forecast_grid` consumes.

Most tests stub the two fitting functions, so they neither import darts nor
fit anything. One `slow` test at the end fits a real model end to end.
"""
import pickle

import numpy as np
import pandas as pd
import pytest

from src.statistical import baselines
from src.statistical.baselines import (
    fit_predict_multivariate_lgbm,
    run_baseline_comparison,
)


class FakeForecast:
    """Stands in for a darts TimeSeries prediction; supports [component]."""

    def __init__(self, name):
        self.name = name

    def __getitem__(self, component):
        return FakeForecast(f"{self.name}[{component}]")

    def __eq__(self, other):
        return isinstance(other, FakeForecast) and other.name == self.name

    def __repr__(self):
        return f"FakeForecast({self.name!r})"


class FakeTimeSeries:
    def __init__(self, components):
        self.components = list(components)
        self.sliced_to = None

    def __getitem__(self, item):
        if isinstance(item, slice):
            self.sliced_to = item
            return self
        return FakeTimeSeries([item])


# ==========================================================================
# run_baseline_comparison
# ==========================================================================

@pytest.fixture
def stub_fitters(monkeypatch):
    """Replaces both fitting functions; records what they were asked for."""
    calls = {"univariate": [], "multivariate": []}

    def fake_univariate(ts, val_len, lags, models=None):
        calls["univariate"].append((ts.components, val_len, lags))
        return {
            "ExponentialSmoothing": FakeForecast(f"es-{ts.components[0]}"),
            "AutoARIMA": FakeForecast(f"arima-{ts.components[0]}"),
            "LightGBM": FakeForecast(f"lgbm-uni-{ts.components[0]}"),
        }

    def fake_multivariate(ts, val_len, lags, cache_dir=None, cache_key=None):
        calls["multivariate"].append((val_len, lags, cache_dir, cache_key))
        return FakeForecast("lgbm-multi")

    monkeypatch.setattr(baselines, "fit_predict_univariate", fake_univariate)
    monkeypatch.setattr(baselines, "fit_predict_multivariate_lgbm", fake_multivariate)
    return calls


def test_comparison_returns_a_dict_per_column(stub_fitters):
    result = run_baseline_comparison(
        FakeTimeSeries(["New York", "Utah"]), ["New York", "Utah"], 12, 24
    )
    assert set(result) == {"New York", "Utah"}


def test_comparison_includes_every_model(stub_fitters):
    result = run_baseline_comparison(FakeTimeSeries(["Utah"]), ["Utah"], 12, 24)
    assert set(result["Utah"]) == {
        "ExponentialSmoothing", "AutoARIMA", "LightGBM (Uni)", "LightGBM (Multi)"
    }


def test_comparison_distinguishes_the_two_lightgbm_fits(stub_fitters):
    """One model sees a single state, the other sees the whole panel. Labelling
    them both 'LightGBM' would make the plot unreadable."""
    result = run_baseline_comparison(FakeTimeSeries(["Utah"]), ["Utah"], 12, 24)
    assert result["Utah"]["LightGBM (Uni)"] == FakeForecast("lgbm-uni-Utah")
    assert result["Utah"]["LightGBM (Multi)"] == FakeForecast("lgbm-multi[Utah]")


def test_comparison_fits_the_multivariate_model_only_once(stub_fitters):
    """It is the expensive one; refitting per state would be pure waste."""
    run_baseline_comparison(
        FakeTimeSeries(["a", "b", "c"]), ["a", "b", "c"], 12, 24
    )
    assert len(stub_fitters["multivariate"]) == 1
    assert len(stub_fitters["univariate"]) == 3


def test_comparison_passes_the_cache_settings_through(stub_fitters):
    run_baseline_comparison(FakeTimeSeries(["Utah"]), ["Utah"], 12, 24,
                            cache_dir="/tmp/artifacts", cache_key="Monthly_NVDRS")
    assert stub_fitters["multivariate"][0] == (12, 24, "/tmp/artifacts", "Monthly_NVDRS")


def test_comparison_output_feeds_plot_forecast_grid(stub_fitters):
    """Contract check: the result shape is {column: {model: forecast}}."""
    result = run_baseline_comparison(FakeTimeSeries(["Utah"]), ["Utah"], 12, 24)
    assert all(isinstance(v, dict) for v in result.values())


# ==========================================================================
# the prediction cache
# ==========================================================================

def test_cache_hit_skips_fitting_entirely(tmp_path, monkeypatch):
    """A cached prediction must short-circuit before any model is touched."""
    cached = {"marker": "from disk"}
    with open(tmp_path / "lgbm_pred_Monthly.pkl", "wb") as fh:
        pickle.dump(cached, fh)

    def explode(*args, **kwargs):
        raise AssertionError("cache miss: should not have fitted")

    monkeypatch.setattr(baselines, "fit_predict_univariate", explode)

    result = fit_predict_multivariate_lgbm(
        _explosive_series(), val_len=12, lags=24,
        cache_dir=tmp_path, cache_key="Monthly",
    )
    assert result == cached


def test_cache_is_keyed_only_by_cache_key(tmp_path):
    """Documented trap: `val_len` and `lags` are not part of the cache key, so
    changing either returns the prediction made under the old settings. The
    notebook says to clear ../artifacts when tuning."""
    with open(tmp_path / "lgbm_pred_Monthly.pkl", "wb") as fh:
        pickle.dump("stale", fh)

    for val_len, lags in [(12, 24), (30, 60)]:
        result = fit_predict_multivariate_lgbm(
            _explosive_series(), val_len=val_len, lags=lags,
            cache_dir=tmp_path, cache_key="Monthly",
        )
        assert result == "stale"


def test_different_cache_keys_do_not_collide(tmp_path):
    for key in ("Monthly", "Weekly"):
        with open(tmp_path / f"lgbm_pred_{key}.pkl", "wb") as fh:
            pickle.dump(key, fh)

    assert fit_predict_multivariate_lgbm(
        _explosive_series(), 12, 24, cache_dir=tmp_path, cache_key="Weekly"
    ) == "Weekly"


def test_cache_directory_is_created(tmp_path, monkeypatch):
    target = tmp_path / "artifacts" / "nested"
    fitted = FakeForecast("fresh")

    monkeypatch.setattr(baselines, "fit_predict_multivariate_lgbm",
                        baselines.fit_predict_multivariate_lgbm)
    monkeypatch.setattr(baselines, "_build_lgbm", None, raising=False)

    class StubModel:
        def __init__(self, lags=None):
            pass

        def fit(self, train):
            return self

        def save(self, path):
            open(path, "wb").close()

        def predict(self, n):
            return fitted

    _patch_lightgbm(monkeypatch, StubModel)

    result = fit_predict_multivariate_lgbm(
        FakeTimeSeries(["a"]), 12, 24, cache_dir=target, cache_key="Monthly"
    )
    assert target.is_dir()
    assert result == fitted
    assert (target / "lgbm_pred_Monthly.pkl").exists()


def test_second_call_reads_the_cache_written_by_the_first(tmp_path, monkeypatch):
    fitted = FakeForecast("fresh")
    fit_count = {"n": 0}

    class StubModel:
        def __init__(self, lags=None):
            pass

        def fit(self, train):
            fit_count["n"] += 1
            return self

        def save(self, path):
            open(path, "wb").close()

        def predict(self, n):
            return fitted

    _patch_lightgbm(monkeypatch, StubModel)

    for _ in range(2):
        fit_predict_multivariate_lgbm(
            FakeTimeSeries(["a"]), 12, 24, cache_dir=tmp_path, cache_key="Monthly"
        )
    assert fit_count["n"] == 1


def test_without_a_cache_dir_nothing_is_written(tmp_path, monkeypatch):
    class StubModel:
        def __init__(self, lags=None):
            pass

        def fit(self, train):
            return self

        def predict(self, n):
            return FakeForecast("fresh")

    _patch_lightgbm(monkeypatch, StubModel)

    fit_predict_multivariate_lgbm(FakeTimeSeries(["a"]), 12, 24)
    assert list(tmp_path.iterdir()) == []


# ==========================================================================
# end to end, with a real model
# ==========================================================================

@pytest.mark.slow
def test_exponential_smoothing_fits_a_real_series():
    """Smoke test that the darts wiring holds: a seasonal series in, a
    forecast of the requested length out."""
    darts = pytest.importorskip("darts")

    index = pd.date_range("2015-01-31", periods=60, freq="ME")
    values = 100 + 10 * np.sin(np.arange(60) * 2 * np.pi / 12)
    ts = darts.TimeSeries.from_dataframe(
        pd.DataFrame({"date": index, "count": values}), time_col="date"
    )

    forecasts = baselines.fit_predict_univariate(
        ts, val_len=12, lags=24, models=["ExponentialSmoothing"]
    )
    assert len(forecasts["ExponentialSmoothing"]) == 12
    assert np.isfinite(forecasts["ExponentialSmoothing"].values()).all()


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _explosive_series():
    """A series that raises if anyone tries to slice it for training."""

    class Explosive:
        def __getitem__(self, item):
            raise AssertionError("cache miss: should not have prepared training data")

    return Explosive()


def _patch_lightgbm(monkeypatch, stub):
    """Swap darts' LightGBMModel for a stub, so no gradient boosting happens."""
    import darts.models

    monkeypatch.setattr(darts.models, "LightGBMModel", stub)
