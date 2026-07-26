"""Tests for `src.geospatial.satscan_io`.

These readers exist because pandas turns 01001 into the integer 1001 on every
read, and a ZIP that lost its leading zero joins to nothing. Most of the file
is therefore about ZIP normalisation.
"""
import pandas as pd
import pytest

from src.geospatial.satscan_io import (
    clean_zip,
    daily_case_totals,
    read_satscan_cases,
    read_satscan_geo,
    read_satscan_pop,
)


# ==========================================================================
# clean_zip
# ==========================================================================

@pytest.mark.parametrize(
    "raw,expected",
    [
        (1001, "01001"),            # integer that lost its leading zero
        ("1001", "01001"),          # same, as text
        (1001.0, "01001"),          # float from a column with NaN in it
        ("01001.0", "01001"),       # float that kept its zero
        ("  10001 ", "10001"),      # whitespace from a fixed-width export
        ("10001", "10001"),         # already correct
    ],
)
def test_clean_zip_normalises_every_shape(raw, expected):
    assert clean_zip(pd.Series([raw])).iloc[0] == expected


def test_clean_zip_only_strips_a_trailing_dot_zero():
    """'.0' in the middle of a value must survive; the anchor is what does it."""
    assert clean_zip(pd.Series(["1.05"])).iloc[0] == "01.05"


def test_clean_zip_returns_strings():
    assert clean_zip(pd.Series([1001, 10001])).map(type).eq(str).all()


# ==========================================================================
# readers
# ==========================================================================

def test_read_cases(write_satscan_files):
    files = write_satscan_files(
        [(1001, 2, "2015-01-01"), ("10002", 1, "2015-01-02")],
        [("01001", 42.07, -72.62)],
    )
    cases = read_satscan_cases(files["cas"])
    assert list(cases.columns) == ["zip", "n_case", "date"]
    assert list(cases["zip"]) == ["01001", "10002"]
    assert pd.api.types.is_datetime64_any_dtype(cases["date"])
    assert cases["n_case"].sum() == 3


def test_read_geo(write_satscan_files):
    files = write_satscan_files(
        [("10001", 1, "2015-01-01")],
        [(1001, 42.07, -72.62)],
    )
    geo = read_satscan_geo(files["geo"])
    assert list(geo.columns) == ["zip", "lat", "lon"]
    assert geo.loc[0, "zip"] == "01001"
    assert geo.loc[0, "lat"] == pytest.approx(42.07)


def test_read_pop(write_satscan_files):
    files = write_satscan_files(
        [("10001", 1, "2015-01-01")],
        [("10001", 40.75, -73.99)],
        [(1001, 2015, 12345)],
    )
    pop = read_satscan_pop(files["pop"])
    assert list(pop.columns) == ["zip", "year", "population"]
    assert pop.loc[0, "zip"] == "01001"
    assert pop.loc[0, "population"] == 12345


def test_readers_agree_on_zip_format(write_satscan_files):
    """The whole point: a .cas with bare integers still joins to a .geo with
    zero-padded strings."""
    files = write_satscan_files(
        [(1001, 1, "2015-01-01")],
        [("01001", 42.07, -72.62)],
        [(1001, 2015, 1000)],
    )
    cases = read_satscan_cases(files["cas"])
    geo = read_satscan_geo(files["geo"])
    pop = read_satscan_pop(files["pop"])
    assert len(cases.merge(geo, on="zip").merge(pop, on="zip")) == 1


def test_readers_handle_multiple_spaces(tmp_path):
    """SaTScan files are whitespace-delimited, not single-space delimited."""
    path = tmp_path / "padded.cas"
    path.write_text("10001    2   2015-01-01\n10002\t1\t2015-01-02\n")
    cases = read_satscan_cases(path)
    assert len(cases) == 2
    assert cases["n_case"].sum() == 3


# ==========================================================================
# daily_case_totals
# ==========================================================================

def test_daily_case_totals_sums_across_zips(write_satscan_files):
    files = write_satscan_files(
        [("10001", 2, "2015-01-01"), ("10002", 3, "2015-01-01"),
         ("10001", 1, "2015-01-02")],
        [("10001", 40.75, -73.99)],
    )
    totals = daily_case_totals(read_satscan_cases(files["cas"]))
    assert totals[pd.Timestamp("2015-01-01")] == 5
    assert totals[pd.Timestamp("2015-01-02")] == 1
