"""Turning NVDRS incident rows into gap-free, resampled time series.

Extracted from notebooks/0.1 and notebooks/0.5, which both repeated the same
three steps: aggregate to daily counts, reindex onto a complete daily calendar
so absent days read as zero rather than as gaps, then resample.

`darts` is imported inside `to_darts_timeseries` rather than at module scope so
that the aggregation helpers stay usable in an environment without it.
"""
from typing import Literal

import pandas as pd

from src.etl.transform import aggregate_nvdrs_daily, aggregate_nvdrs_daily_injury

# pandas frequency aliases, keyed by the name used throughout the notebooks.
DEFAULT_FREQS = {"daily": "D", "weekly": "W", "monthly": "ME"}


def resample_counts(
    daily_df: pd.DataFrame,
    date_col: str = "DeathDate",
    freqs: dict | None = None,
) -> dict:
    """Reindex daily counts onto a complete calendar, then resample.

    `aggregate_nvdrs_*` only emits rows for dates that actually occur, so a day
    with no deaths is simply absent. Resampling straight off that frame would
    treat the gap as missing rather than as a zero and bias every rolling
    statistic upwards. The `resample('D').sum()` here is what closes the gaps.

    The date index is coerced to datetime first. NVDRS is read with dtype=str
    and only `DeathDate` is converted by `filter_nvdrs_suicides`, so an
    InjuryDate-based frame still arrives as strings and would otherwise fail
    at `.resample()`.

    Returns a dict keyed like `freqs` ('daily', 'weekly', 'monthly' by
    default), each value a DataFrame indexed by `date_col`.
    """
    freqs = DEFAULT_FREQS if freqs is None else freqs

    indexed = daily_df.set_index(date_col) if date_col in daily_df.columns else daily_df
    if not isinstance(indexed.index, pd.DatetimeIndex):
        indexed = indexed.set_axis(pd.to_datetime(indexed.index), axis=0)

    complete = indexed.resample("D").sum()

    return {name: complete.resample(alias).sum() for name, alias in freqs.items()}


def build_nvdrs_frames(
    df: pd.DataFrame,
    geo_level: Literal["county", "state"] | None = None,
    geo_col: str = "DeathFIPS",
    date_basis: Literal["death", "injury"] = "death",
    freqs: dict | None = None,
) -> dict:
    """Aggregate incidents and return daily / weekly / monthly frames.

    `geo_level=None` gives a single national count column. Otherwise the result
    is one column per geography.

    Note that for `date_basis='death'` the geography is chosen by `geo_col`,
    not by `geo_level`; `geo_level` only decides whether to pivot at all. That
    is inherited from `aggregate_nvdrs_daily` - pass `geo_col='DeathState'`
    explicitly when you want state columns.
    """
    if date_basis == "injury":
        daily = aggregate_nvdrs_daily_injury(df, geo_level=geo_level)
        date_col = "InjuryDate"
    else:
        daily = aggregate_nvdrs_daily(df, geo_level=geo_level, geo_col=geo_col)
        date_col = "DeathDate"

    return resample_counts(daily, date_col=date_col, freqs=freqs)


def to_darts_timeseries(frame: pd.DataFrame, time_col: str | None = None, freq: str | None = None):
    """Convert one resampled frame into a darts `TimeSeries`.

    Missing dates are filled with 0 rather than interpolated: an absent day is
    a day with no deaths, not an unobserved one.
    """
    from darts import TimeSeries

    if time_col is None:
        frame = frame.reset_index()
        time_col = frame.columns[0]

    return TimeSeries.from_dataframe(
        frame,
        time_col=time_col,
        fill_missing_dates=True,
        freq=freq,
        fillna_value=0,
    )


def to_darts_dict(frames: dict, freqs: dict | None = None) -> dict:
    """Apply `to_darts_timeseries` across the output of `build_nvdrs_frames`."""
    freqs = DEFAULT_FREQS if freqs is None else freqs
    return {name: to_darts_timeseries(frame, freq=freqs.get(name))
            for name, frame in frames.items()}
