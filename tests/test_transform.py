"""Tests for `src.etl.transform`.

Everything here runs on a handful of synthetic rows. The real NVDRS extract is
multi-GB and restricted; none of it is read.

Two tests are marked `xfail` on purpose. They describe behaviour the rest of
the module implies but does not deliver, so they document a known defect
instead of asserting it is correct. See the reason strings.
"""
import numpy as np
import pandas as pd
import pytest

from src.etl import transform
from src.etl.transform import (
    aggregate_nvdrs_daily,
    aggregate_nvdrs_daily_injury,
    aggregate_nvdrs_monthly,
    calc_pct_change,
    enrich_fips_data,
    enrich_zip_data,
    filter_nvdrs_suicides,
    harmonize_zcta_boundaries,
)


# ==========================================================================
# filter_nvdrs_suicides
# ==========================================================================

def test_filter_drops_non_suicide_incidents(suicides_frame):
    assert "4" not in set(suicides_frame["IncidentID"]), "single homicide must be dropped"
    assert not suicides_frame["IncidentCategory_c"].str.contains("suicide", case=False).eq(False).any()


def test_filter_keeps_single_suicides(suicides_frame):
    singles = suicides_frame[suicides_frame["IncidentCategory_c"] == "Single suicide"]
    assert len(singles) == 2


def test_filter_keeps_only_the_perpetrator_in_homicide_suicide(suicides_frame):
    """A murder-suicide contributes exactly one suicide, not two deaths."""
    inc3 = suicides_frame[suicides_frame["IncidentID"] == "3"]
    assert len(inc3) == 1
    assert inc3.iloc[0]["PersonType"] == "Both victim and suspect"


def test_filter_converts_deathdate_to_datetime(suicides_frame):
    assert pd.api.types.is_datetime64_any_dtype(suicides_frame["DeathDate"])


def test_filter_matches_category_case_insensitively(nvdrs_frame):
    shouty = nvdrs_frame.copy()
    shouty["IncidentCategory_c"] = shouty["IncidentCategory_c"].str.upper()
    # 'SINGLE SUICIDE' no longer equals 'Single suicide', so those rows fall into
    # the multi-person branch, but they must still survive the substring filter.
    assert len(filter_nvdrs_suicides(shouty)) > 0


def test_filter_tolerates_missing_category(nvdrs_frame):
    with_nan = nvdrs_frame.copy()
    with_nan.loc[0, "IncidentCategory_c"] = np.nan
    result = filter_nvdrs_suicides(with_nan)
    assert "1" not in set(result["IncidentID"]), "NaN category is not a suicide"


def test_filter_does_not_mutate_input(nvdrs_frame):
    before = nvdrs_frame.copy()
    filter_nvdrs_suicides(nvdrs_frame)
    pd.testing.assert_frame_equal(nvdrs_frame, before)


@pytest.mark.xfail(
    reason=(
        "Known defect. The multi-person branch keeps exactly one row per "
        "IncidentID via idxmax() on 'PersonType == Both victim and suspect'. "
        "When no row carries that label - a 'Multiple suicides' incident, "
        "where every decedent died by suicide - idxmax() falls back to the "
        "first row, silently dropping the other victims."
    ),
    strict=True,
)
def test_filter_keeps_every_victim_of_a_multiple_suicide(suicides_frame):
    inc5 = suicides_frame[suicides_frame["IncidentID"] == "5"]
    assert len(inc5) == 2


def test_filter_multiple_suicide_current_behaviour(suicides_frame):
    """Pins today's output so the xfail above cannot flip unnoticed."""
    assert len(suicides_frame[suicides_frame["IncidentID"] == "5"]) == 1


# ==========================================================================
# aggregators
# ==========================================================================

def test_aggregate_daily_national_counts(suicides_frame):
    daily = aggregate_nvdrs_daily(suicides_frame)
    assert list(daily.columns) == ["DeathDate", "incident_count"]
    row = daily[daily["DeathDate"] == pd.Timestamp("2015-03-01")]
    assert row["incident_count"].iloc[0] == 2


def test_aggregate_daily_counts_sum_to_row_count(suicides_frame):
    daily = aggregate_nvdrs_daily(suicides_frame)
    assert daily["incident_count"].sum() == len(suicides_frame)


def test_aggregate_daily_geo_pivot_fills_absent_pairs_with_zero(suicides_frame):
    pivot = aggregate_nvdrs_daily(suicides_frame, geo_level="county", geo_col="DeathFIPS")
    assert pivot.isna().sum().sum() == 0
    # 36061 had no deaths on 2016-07-04, so that cell must be 0, not NaN.
    row = pivot[pivot["DeathDate"] == pd.Timestamp("2016-07-04")]
    assert row["36061"].iloc[0] == 0


def test_aggregate_daily_pivot_does_not_insert_absent_dates(suicides_frame):
    """Only dates that appear in the data become rows.

    Callers who need a gap-free index have to `.resample('D').sum()` afterwards,
    which is exactly what the notebooks do.
    """
    pivot = aggregate_nvdrs_daily(suicides_frame, geo_level="county")
    assert len(pivot) == suicides_frame["DeathDate"].nunique()
    assert pd.Timestamp("2015-03-02") not in set(pivot["DeathDate"])


def test_aggregate_daily_geo_level_only_toggles_the_pivot(suicides_frame):
    """`geo_level` is a truthiness flag here; `geo_col` picks the column.

    This is inconsistent with `aggregate_nvdrs_monthly` and
    `aggregate_nvdrs_daily_injury`, which both branch on `geo_level` itself.
    See the xfail below.
    """
    by_flag = aggregate_nvdrs_daily(suicides_frame, geo_level="state")
    by_col = aggregate_nvdrs_daily(suicides_frame, geo_level="county", geo_col="DeathFIPS")
    pd.testing.assert_frame_equal(by_flag, by_col)


@pytest.mark.xfail(
    reason=(
        "Known defect. aggregate_nvdrs_daily(df, geo_level='state') pivots on "
        "geo_col, which defaults to 'DeathFIPS', so it returns county columns. "
        "Its siblings aggregate_nvdrs_monthly and aggregate_nvdrs_daily_injury "
        "both return DeathState/InjuryState for geo_level='state'. Callers that "
        "select state columns by name get a KeyError, or - as in "
        "notebooks/0.5, which guards with `if state in df.columns` - silently "
        "plot nothing."
    ),
    strict=True,
)
def test_aggregate_daily_state_level_returns_state_columns(suicides_frame):
    pivot = aggregate_nvdrs_daily(suicides_frame, geo_level="state")
    assert "New York" in pivot.columns


def test_aggregate_monthly_state_returns_state_columns(suicides_frame):
    pivot = aggregate_nvdrs_monthly(suicides_frame, geo_level="state")
    assert "New York" in pivot.columns
    assert "Utah" in pivot.columns


def test_aggregate_monthly_county_returns_fips_columns(suicides_frame):
    pivot = aggregate_nvdrs_monthly(suicides_frame, geo_level="county")
    assert "36061" in pivot.columns


def test_aggregate_monthly_national(suicides_frame):
    monthly = aggregate_nvdrs_monthly(suicides_frame)
    assert list(monthly.columns) == ["DeathDate_myr", "incident_count"]
    assert monthly["incident_count"].sum() == len(suicides_frame)


def test_aggregate_injury_uses_injury_columns(suicides_frame):
    pivot = aggregate_nvdrs_daily_injury(suicides_frame, geo_level="state")
    assert "InjuryDate" in pivot.columns
    assert "New York" in pivot.columns


def test_aggregate_injury_national(suicides_frame):
    daily = aggregate_nvdrs_daily_injury(suicides_frame)
    assert list(daily.columns) == ["InjuryDate", "incident_count"]


def test_aggregate_injury_county_uses_injury_fips(suicides_frame):
    pivot = aggregate_nvdrs_daily_injury(suicides_frame, geo_level="county")
    assert "19153" in pivot.columns


# ==========================================================================
# harmonize_zcta_boundaries
# ==========================================================================

@pytest.fixture
def crosswalk_file(tmp_path, monkeypatch):
    """Writes a pipe-delimited stand-in for the Census 2010->2020 ZCTA file.

    Layout, chosen to cover every branch:
      10001 -> 20001            whole ZCTA, AF = 1.0
      10002 -> 20002 / 20003    split 60/40
      10003 -> 20003            merges into 20003 alongside part of 10002
      10004 -> 20004            AREALAND_ZCTA5_10 = 0, so AF must be 0
      10009                     absent from the crosswalk entirely
    """
    rows = [
        ("10001", "20001", 1000, 1000),
        ("10002", "20002", 1000, 600),
        ("10002", "20003", 1000, 400),
        ("10003", "20003", 500, 500),
        ("10004", "20004", 0, 0),
    ]
    path = tmp_path / "zcta_crosswalk.txt"
    header = "GEOID_ZCTA5_10|GEOID_ZCTA5_20|AREALAND_ZCTA5_10|AREALAND_PART"
    path.write_text(
        header + "\n" + "\n".join("|".join(str(v) for v in r) for r in rows) + "\n"
    )
    monkeypatch.setattr(transform, "get_data_path", lambda *a, **k: path)
    return path


@pytest.fixture
def pop_frame():
    return pd.DataFrame(
        {
            "ZIP": ["10001", "10002", "10003", "10004", "10009", "20001"],
            "Year": [2015, 2015, 2015, 2015, 2015, 2022],
            "Population": [1000.0, 1000.0, 500.0, 700.0, 900.0, 1100.0],
        }
    )


def test_harmonize_maps_whole_zcta_unchanged(crosswalk_file, pop_frame):
    out = harmonize_zcta_boundaries(pop_frame)
    row = out[(out["ZIP"] == "20001") & (out["Year"] == 2015)]
    assert row["Population"].iloc[0] == 1000


def test_harmonize_apportions_a_split_by_land_area(crosswalk_file, pop_frame):
    out = harmonize_zcta_boundaries(pop_frame)
    # 10002 (pop 1000) splits 60/40 into 20002 and 20003.
    assert out.loc[(out["ZIP"] == "20002") & (out["Year"] == 2015), "Population"].iloc[0] == 600


def test_harmonize_sums_contributions_into_a_merged_zcta(crosswalk_file, pop_frame):
    out = harmonize_zcta_boundaries(pop_frame)
    # 20003 receives 40% of 10002 (400) plus all of 10003 (500).
    assert out.loc[(out["ZIP"] == "20003") & (out["Year"] == 2015), "Population"].iloc[0] == 900


def test_harmonize_conserves_total_population_across_a_split(crosswalk_file, pop_frame):
    """Area-based apportionment must not create or destroy people."""
    two_zctas = pop_frame[pop_frame["ZIP"].isin(["10002", "10003"])]
    out = harmonize_zcta_boundaries(two_zctas)
    assert out["Population"].sum() == 1500


def test_harmonize_zero_land_area_gives_zero_not_division_error(crosswalk_file, pop_frame):
    out = harmonize_zcta_boundaries(pop_frame)
    assert out.loc[(out["ZIP"] == "20004") & (out["Year"] == 2015), "Population"].iloc[0] == 0


def test_harmonize_passes_2021_and_later_through_untouched(crosswalk_file, pop_frame):
    out = harmonize_zcta_boundaries(pop_frame)
    row = out[(out["ZIP"] == "20001") & (out["Year"] == 2022)]
    assert row["Population"].iloc[0] == 1100


def test_harmonize_treats_2020_as_pre_boundary_change(crosswalk_file):
    """The API switched boundaries in 2021, so 2020 still needs harmonizing."""
    df = pd.DataFrame({"ZIP": ["10002"], "Year": [2020], "Population": [1000.0]})
    out = harmonize_zcta_boundaries(df)
    assert set(out["ZIP"]) == {"20002", "20003"}

def test_harmonize_reports_zips_absent_from_the_crosswalk(crosswalk_file, pop_frame, capsys):
    """ZIP 10009 has 900 people and no crosswalk row. It's dropped, but now
    with a printed warning naming it and how much population it carried."""
    out = harmonize_zcta_boundaries(pop_frame)
    assert "10009" not in set(out["ZIP"])
    assert out[out["Year"] == 2015]["Population"].sum() == 2500  # 900 people lost
    captured = capsys.readouterr()
    assert "10009" in captured.out
    assert "Warning" in captured.out

def test_harmonize_carries_state_and_county_when_asked(crosswalk_file):
    df = pd.DataFrame(
        {
            "ZIP": ["10001"], "Year": [2015], "Population": [1000.0],
            "state": ["36"], "county": ["061"],
        }
    )
    out = harmonize_zcta_boundaries(df, state_col="state", county_col="county")
    assert out.loc[0, "state"] == "36"
    assert out.loc[0, "county"] == "061"


def test_harmonize_output_is_sorted_by_zip_then_year(crosswalk_file, pop_frame):
    out = harmonize_zcta_boundaries(pop_frame)
    assert out[["ZIP", "Year"]].equals(
        out[["ZIP", "Year"]].sort_values(["ZIP", "Year"]).reset_index(drop=True)
    )


def test_harmonize_rounds_population_to_whole_people(crosswalk_file):
    df = pd.DataFrame({"ZIP": ["10002"], "Year": [2015], "Population": [1001.0]})
    out = harmonize_zcta_boundaries(df)
    assert (out["Population"] == out["Population"].round(0)).all()


# ==========================================================================
# enrich_fips_data
# ==========================================================================

@pytest.fixture
def fips_crosswalk_file(tmp_path, monkeypatch):
    """Headerless national_county.txt: State_Abbr, State_FIPS, County_FIPS, Name, Class."""
    path = tmp_path / "national_county.txt"
    path.write_text(
        "NY,36,061,New York County,H1\n"
        "UT,49,035,Salt Lake County,H1\n"
    )
    monkeypatch.setattr(transform, "get_data_path", lambda *a, **k: path)
    return path


def test_enrich_fips_missing_column_returns_nan_columns(fips_crosswalk_file):
    df = pd.DataFrame({"KEY": ["k1"]})
    out = enrich_fips_data(df)
    assert out["Hospital_State"].isna().all()
    assert out["Hospital_County"].isna().all()


def test_enrich_fips_joins_state_and_county(fips_crosswalk_file):
    df = pd.DataFrame({"HFIPSSTCO": ["36061", "49035"]})
    out = enrich_fips_data(df)
    assert list(out["Hospital_State"]) == ["NY", "UT"]
    assert list(out["Hospital_County"]) == ["New York County", "Salt Lake County"]


def test_enrich_fips_zero_pads_and_strips_float_suffix(fips_crosswalk_file):
    """HCUP FIPS often arrive as floats (36061.0) or short ints (4013)."""
    df = pd.DataFrame({"HFIPSSTCO": [36061.0]})
    out = enrich_fips_data(df)
    assert out.loc[0, "HFIPSSTCO"] == "36061"
    assert out.loc[0, "Hospital_State"] == "NY"


def test_enrich_fips_unmatched_code_yields_nan(fips_crosswalk_file):
    df = pd.DataFrame({"HFIPSSTCO": ["99999"]})
    out = enrich_fips_data(df)
    assert pd.isna(out.loc[0, "Hospital_State"])


def test_enrich_fips_preserves_nan_input(fips_crosswalk_file):
    df = pd.DataFrame({"HFIPSSTCO": [np.nan, "36061"]})
    out = enrich_fips_data(df)
    assert pd.isna(out.loc[0, "HFIPSSTCO"])
    assert out.loc[1, "Hospital_State"] == "NY"


def test_enrich_fips_drops_the_join_key_column(fips_crosswalk_file):
    df = pd.DataFrame({"HFIPSSTCO": ["36061"]})
    assert "FIPS" not in enrich_fips_data(df).columns


# ==========================================================================
# enrich_zip_data
# ==========================================================================

def test_enrich_zip_adds_city_county_state(monkeypatch, fake_search_engine):
    from tests.conftest import FakeZipResult

    engine = fake_search_engine(
        {"01001": FakeZipResult(major_city="Agawam", county="Hampden County", state="MA")}
    )
    monkeypatch.setattr(transform, "SearchEngine", engine)

    df = pd.DataFrame({"ZIP": [1001.0]})
    out = enrich_zip_data(df)
    assert out.loc[0, "ZIP_c"] == "01001", "float ZIPs must lose '.0' and zero-pad"
    assert out.loc[0, "City"] == "Agawam"
    assert out.loc[0, "County"] == "Hampden County"
    assert out.loc[0, "State"] == "MA"


def test_enrich_zip_unknown_zip_yields_none(monkeypatch, fake_search_engine):
    monkeypatch.setattr(transform, "SearchEngine", fake_search_engine({}))
    out = enrich_zip_data(pd.DataFrame({"ZIP": ["99999"]}))
    assert out.loc[0, "City"] is None


def test_enrich_zip_honours_custom_column_names(monkeypatch, fake_search_engine):
    from tests.conftest import FakeZipResult

    engine = fake_search_engine({"10001": FakeZipResult(major_city="New York")})
    monkeypatch.setattr(transform, "SearchEngine", engine)
    out = enrich_zip_data(pd.DataFrame({"DerivedZip": ["10001"]}),
                          zip_col="DerivedZip", zip_col_c="clean")
    assert out.loc[0, "clean"] == "10001"
    assert out.loc[0, "City"] == "New York"


# ==========================================================================
# calc_pct_change
# ==========================================================================

@pytest.mark.parametrize(
    "old,new,expected",
    [
        (100.0, 150.0, 50.0),
        (100.0, 50.0, -50.0),
        (100.0, 100.0, 0.0),
        (200.0, 50.0, -75.0),
    ],
)
def test_calc_pct_change_values(old, new, expected):
    df = pd.DataFrame({"a": [old], "b": [new]})
    assert calc_pct_change(df, "a", "b")[0] == pytest.approx(expected)


def test_calc_pct_change_zero_denominator_is_nan():
    df = pd.DataFrame({"a": [0.0], "b": [10.0]})
    assert np.isnan(calc_pct_change(df, "a", "b")[0])


def test_calc_pct_change_negative_denominator_is_nan():
    """Populations are never negative; guarding on `> 0` treats them as invalid."""
    df = pd.DataFrame({"a": [-10.0], "b": [10.0]})
    assert np.isnan(calc_pct_change(df, "a", "b")[0])
