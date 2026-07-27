"""Baseline forecasters for NVDRS incident counts.

These are first passes with no held-ou evaluation, 
early results were noise dominated.

The LightGBM baseline lives here alongside the statistical models because the
notebook compares them head to head. It belongs in `src/ml/` once that package
exists.

`darts` is imported inside the functions so that importing this module - to
read it, or to collect tests - does not require the forecasting stack.
"""
import pickle
from pathlib import Path


def fit_predict_univariate(ts, val_len: int, lags: int, models: list = None) -> dict:
    """Fit each baseline on `ts` minus the last `val_len` points and forecast.

    `models` selects which of 'ExponentialSmoothing', 'AutoARIMA' and
    'LightGBM' to run; the default runs all three.

    Returns {model name: forecast TimeSeries}.
    """
    from darts.models import AutoARIMA, ExponentialSmoothing, LightGBMModel

    builders = {
        "ExponentialSmoothing": lambda: ExponentialSmoothing(),
        "AutoARIMA": lambda: AutoARIMA(),
        "LightGBM": lambda: LightGBMModel(lags=lags),
    }
    if models is None:
        models = list(builders)

    train = ts[:-val_len]
    return {name: builders[name]().fit(train).predict(val_len) for name in models}


def fit_predict_multivariate_lgbm(ts, val_len: int, lags: int, cache_dir=None, cache_key: str = None):
    """Fit one LightGBM across every component of a multivariate series.

    Fitting the national panel takes long enough that the notebook cached both
    the model and its predictions to disk. That caching is preserved here and
    is opt-in: pass `cache_dir` to enable it.

    Note the cache is keyed only by `cache_key`, not by `val_len` or `lags`.
    Change either of those and you must clear the cache or pass a new key,
    otherwise you get back a prediction made under the old settings.
    """
    from darts.models import LightGBMModel

    if cache_dir is None:
        model = LightGBMModel(lags=lags)
        model.fit(ts[:-val_len])
        return model.predict(val_len)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_key or "default"
    model_path = cache_dir / f"lgbm_model_{key}.pkl"
    pred_path = cache_dir / f"lgbm_pred_{key}.pkl"

    # Check the cache before preparing anything: a hit should do no work.
    if pred_path.exists():
        with open(pred_path, "rb") as fh:
            return pickle.load(fh)

    if model_path.exists():
        model = LightGBMModel.load(str(model_path))
    else:
        model = LightGBMModel(lags=lags)
        model.fit(ts[:-val_len])
        model.save(str(model_path))

    prediction = model.predict(val_len)
    with open(pred_path, "wb") as fh:
        pickle.dump(prediction, fh)
    return prediction


def run_baseline_comparison(ts, columns: list, val_len: int, lags: int,
                            cache_dir=None, cache_key: str = None) -> dict:
    """Forecast every column in `columns` with every baseline.

    Combines a single multivariate LightGBM fit - which sees all components at
    once - with per-column univariate fits, so the two can be compared on the
    same validation window.

    Returns {column: {model name: forecast TimeSeries}}, shaped for
    `src.utils.viz.plot_forecast_grid`.
    """
    multi_pred = fit_predict_multivariate_lgbm(
        ts, val_len, lags, cache_dir=cache_dir, cache_key=cache_key
    )

    results = {}
    for column in columns:
        forecasts = fit_predict_univariate(ts[column], val_len, lags)
        forecasts = {f"{name} (Uni)" if name == "LightGBM" else name: pred
                     for name, pred in forecasts.items()}
        forecasts["LightGBM (Multi)"] = multi_pred[column]
        results[column] = forecasts

    return results
