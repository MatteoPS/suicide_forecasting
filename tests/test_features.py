"""Tests for `src.etl.features`.

The one behaviour worth guarding hard is gap filling. The aggregators only
emit rows for dates that occur, so a day with no deaths is simply absent from
the frame. Resampling that directly treats the gap as missing rather than as a
zero, which biases every monthly total and every rolling statistic upward.
"""
import pandas as pd
import pytest

from src.etl.features import (
    DEFAULT_FREQS,
    build_nvdrs_frames,
    resample_counts,
    to_darts_dict,
    to_darts_timeseries,
)


# ==========================================================================
# resample_counts
# ==========================================================================

@pytest.fixture
def sparse_daily():
    """Three observed days spanning a week; the other four are absent."""
    return pd.DataFrame(
        {
            "DeathDate": pd.to_datetime(["2015-01-01", "2015-01-04", "2015-01-07"]),
            "incident_count": [2, 3, 1],
        }
    )


def test_resample_fills_absent_days_with_zero(sparse_daily):
    frames = resample_counts(sparse_daily)
    daily = frames["daily"]
    assert len(daily) == 7, "1 Jan through 7 Jan inclusive"
    assert daily.loc["2015-01-02", "incident_count"] == 0


def test_resample_preserves_the_total(sparse_daily):
    frames = resample_counts(sparse_daily)
    for name, frame in frames.items():
        assert frame["incident_count"].sum() == 6, name


def test_resample_returns_all_default_frequencies(sparse_daily):
    assert set(resample_counts(sparse_daily)) == set(DEFAULT_FREQS)


def test_resample_monthly_uses_month_end(sparse_daily):
    monthly = resample_counts(sparse_daily)["monthly"]
    assert monthly.index[0] == pd.Timestamp("2015-01-31")


def test_resample_accepts_a_custom_frequency_map(sparse_daily):
    frames = resample_counts(sparse_daily, freqs={"quarterly": "QE"})
    assert set(frames) == {"quarterly"}
    assert frames["quarterly"]["incident_count"].sum() == 6


def test_resample_accepts_an_already_indexed_frame(sparse_daily):
    indexed = sparse_daily.set_index("DeathDate")
    frames = resample_counts(indexed)
    assert len(frames["daily"]) == 7


# ==========================================================================
# build_nvdrs_frames
# ==========================================================================

def test_build_national_frames(suicides_frame):
    frames = build_nvdrs_frames(suicides_frame)
    assert list(frames["daily"].columns) == ["incident_count"]
    assert frames["daily"]["incident_count"].sum() == len(suicides_frame)


def test_build_state_frames_when_geo_col_is_given(suicides_frame):
    """`geo_level` alone is not enough; `aggregate_nvdrs_daily` reads the
    column name from `geo_col`. Passing it explicitly is the documented fix."""
    frames = build_nvdrs_frames(suicides_frame, geo_level="state", geo_col="DeathState")
    assert "New York" in frames["monthly"].columns


def test_build_county_frames(suicides_frame):
    frames = build_nvdrs_frames(suicides_frame, geo_level="county", geo_col="DeathFIPS")
    assert "36061" in frames["daily"].columns


def test_build_frames_close_the_calendar_gaps(suicides_frame):
    """The fixture spans March 2015 to January 2017 with three observed days."""
    frames = build_nvdrs_frames(suicides_frame)
    daily = frames["daily"]
    assert (daily.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()


def test_build_frames_on_injury_dates(suicides_frame):
    frames = build_nvdrs_frames(suicides_frame, geo_level="state", date_basis="injury")
    assert "New York" in frames["monthly"].columns
    assert frames["daily"].index.name == "InjuryDate"


def test_injury_and_death_bases_disagree_when_dates_differ(suicides_frame):
    """Incident 1 was injured on 28 Feb and died on 1 Mar. The two bases must
    put it in different months, which is the whole reason both exist."""
    by_death = build_nvdrs_frames(suicides_frame)["monthly"]
    by_injury = build_nvdrs_frames(suicides_frame, date_basis="injury")["monthly"]
    assert by_death.loc["2015-03-31", "incident_count"] == 2
    assert by_injury.loc["2015-02-28", "incident_count"] == 1


# ==========================================================================
# darts conversion
# ==========================================================================

def test_to_darts_timeseries_fills_missing_dates_with_zero(sparse_daily):
    pytest.importorskip("darts")
    ts = to_darts_timeseries(sparse_daily, time_col="DeathDate", freq="D")
    assert len(ts) == 7
    assert ts.values().sum() == 6


def test_to_darts_timeseries_infers_the_time_column_from_the_index(sparse_daily):
    pytest.importorskip("darts")
    frames = resample_counts(sparse_daily)
    ts = to_darts_timeseries(frames["daily"], freq="D")
    assert len(ts) == 7


def test_to_darts_dict_covers_every_frequency(suicides_frame):
    pytest.importorskip("darts")
    frames = build_nvdrs_frames(suicides_frame, geo_level="state", geo_col="DeathState")
    series = to_darts_dict(frames)
    assert set(series) == set(DEFAULT_FREQS)
    assert "New York" in series["monthly"].components
