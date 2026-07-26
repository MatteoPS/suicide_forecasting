"""End-to-end tests for `src.geospatial.prep_satscan.prep_satscan_gui`.

This is the component the README marks "Working - real data", so it gets an
integration test rather than a set of unit tests: the whole function runs, with
every external boundary replaced by a stub.

  load_nvdrs        -> six synthetic rows instead of the restricted extract
  fetch_census      -> a canned ACS response instead of the live API
  SearchEngine      -> an in-memory dict instead of the uszipcode SQLite DB
  get_data_path     -> tmp_path fixtures for the Gazetteer and ZCTA crosswalk
  PROJECT_ROOT      -> tmp_path, so outputs land in the test's own directory

The assertions target the parts that quietly change the numerator or the
denominator: sentinel ZIPs, the residence fallback, the 2010 backfill, the
unmappable-ZIP drop, and the zero-population drop.
"""
import pandas as pd
import pytest

from src.geospatial import prep_satscan
from src.geospatial.prep_satscan import prep_satscan_gui

from tests.conftest import FakeZipResult

REQUIRED = ["IncidentID", "DeathDate", "InjuryZip", "ResidenceZip",
            "IncidentCategory_c", "PersonType"]


def _read(out_dir, name):
    return pd.read_csv(out_dir / name, dtype={"ZIP": str})


# ==========================================================================
# input validation
# ==========================================================================

def test_rejects_column_list_missing_required_fields():
    with pytest.raises(ValueError, match="missing required columns"):
        prep_satscan_gui(nvdrs_cols=["IncidentID", "DeathDate"])


def test_error_names_the_missing_columns():
    with pytest.raises(ValueError) as exc:
        prep_satscan_gui(nvdrs_cols=["IncidentID", "DeathDate", "InjuryZip",
                                     "ResidenceZip", "IncidentCategory_c"])
    assert "PersonType" in str(exc.value)


def test_accepts_a_superset_of_required_columns(prep_env):
    prep_satscan_gui(nvdrs_cols=REQUIRED + ["Sex", "AgeYears_c"], nickname="ok")
    assert (prep_env["out_dir"] / "ok_full_cas.csv").exists()


# ==========================================================================
# artifacts
# ==========================================================================

def test_writes_the_four_satscan_artifacts(prep_env):
    prep_satscan_gui(nickname="test")
    out = prep_env["out_dir"]
    for suffix in ("_full_cas.csv", "_full_pop.csv", "_full_geo.csv",
                   "_nvdrs_analytical.csv"):
        assert (out / f"test{suffix}").exists(), suffix


def test_case_file_has_the_satscan_column_order(prep_env):
    prep_satscan_gui(nickname="test")
    cases = _read(prep_env["out_dir"], "test_full_cas.csv")
    assert list(cases.columns) == ["ZIP", "Cases", "DeathDate"]


def test_population_file_has_the_satscan_column_order(prep_env):
    prep_satscan_gui(nickname="test")
    pop = _read(prep_env["out_dir"], "test_full_pop.csv")
    assert list(pop.columns) == ["ZIP", "Year", "Population"]


def test_coordinate_file_has_the_satscan_column_order(prep_env):
    prep_satscan_gui(nickname="test")
    geo = _read(prep_env["out_dir"], "test_full_geo.csv")
    assert list(geo.columns) == ["ZIP", "Latitude", "Longitude"]


def test_nickname_defaults_to_the_literal_string_none(prep_env):
    """Documented wart: `nickname` has no default, so calling
    prep_satscan_gui() straight from `__main__` writes 'None_full_cas.csv'.
    Harmless but confusing; worth a real default or a required argument.
    """
    prep_satscan_gui()
    assert (prep_env["out_dir"] / "None_full_cas.csv").exists()


# ==========================================================================
# case selection
# ==========================================================================

def test_keeps_only_suicides(prep_env):
    prep_satscan_gui(nickname="test")
    cases = _read(prep_env["out_dir"], "test_full_cas.csv")
    assert "2015-03-05" not in set(cases["DeathDate"]), "the homicide leaked through"


def test_drops_deaths_before_2010(prep_env):
    prep_satscan_gui(nickname="test")
    cases = _read(prep_env["out_dir"], "test_full_cas.csv")
    assert not cases["DeathDate"].str.startswith("2009").any()


def test_falls_back_to_residence_zip_by_default(prep_env):
    """Incident 2 has a blank injury ZIP and incident 3 a 99999 sentinel;
    both should be located by residence."""
    prep_satscan_gui(nickname="test")
    cases = _read(prep_env["out_dir"], "test_full_cas.csv")
    assert {"10002", "10003"} <= set(cases["ZIP"])


def test_use_res_zip_false_drops_cases_without_an_injury_zip(prep_env):
    prep_satscan_gui(nickname="test", use_res_zip=False)
    cases = _read(prep_env["out_dir"], "test_full_cas.csv")
    assert "10002" not in set(cases["ZIP"])
    assert "10003" not in set(cases["ZIP"])
    assert "10001" in set(cases["ZIP"])


def test_aggregates_to_zip_and_date(prep_env, nvdrs_rows, monkeypatch):
    """Two deaths in one ZIP on one day must become a single row with Cases=2,
    which is the format SaTScan's Poisson model expects."""
    doubled = pd.concat([nvdrs_rows, nvdrs_rows.iloc[[0]]], ignore_index=True)
    doubled.loc[len(doubled) - 1, "IncidentID"] = "7"
    monkeypatch.setattr(prep_satscan, "load_nvdrs", lambda **kw: doubled.copy())

    prep_satscan_gui(nickname="test")
    cases = _read(prep_env["out_dir"], "test_full_cas.csv")
    row = cases[(cases["ZIP"] == "10001") & (cases["DeathDate"] == "2015-03-01")]
    assert len(row) == 1
    assert row["Cases"].iloc[0] == 2


# ==========================================================================
# unmappable ZIPs
# ==========================================================================

def test_unmappable_zip_is_dropped_from_the_case_file(prep_env):
    """SaTScan aborts on a location with no coordinate, so 00501 must go."""
    prep_satscan_gui(nickname="test")
    cases = _read(prep_env["out_dir"], "test_full_cas.csv")
    assert "00501" not in set(cases["ZIP"])


def test_unmappable_zip_is_written_to_the_missingness_file(prep_env):
    """The drop is documented, not silent - that is the point of the file."""
    prep_satscan_gui(nickname="test")
    dropped = _read(prep_env["out_dir"], "test_dropped_unmappable_cases.csv")
    assert "00501" in set(dropped["ZIP"])
    assert dropped.loc[dropped["ZIP"] == "00501", "Cases"].iloc[0] == 1


def test_missingness_file_records_unknown_metadata_when_lookup_fails(prep_env):
    prep_satscan_gui(nickname="test")
    dropped = _read(prep_env["out_dir"], "test_dropped_unmappable_cases.csv")
    assert dropped.loc[dropped["ZIP"] == "00501", "Type"].iloc[0] == "Unknown"


def test_coordinate_healer_recovers_zips_absent_from_the_gazetteer(
    prep_env, monkeypatch, fake_search_engine
):
    """The Gazetteer drops populated ZCTAs; uszipcode patches them back in."""
    engine = fake_search_engine(
        {"00501": FakeZipResult(lat=40.8, lng=-73.0, major_city="Holtsville",
                                county="Suffolk County", state="NY")}
    )
    monkeypatch.setattr(prep_satscan, "SearchEngine", engine)

    prep_satscan_gui(nickname="test")
    geo = _read(prep_env["out_dir"], "test_full_geo.csv")
    assert "00501" in set(geo["ZIP"])
    # ...and once it has a coordinate it is no longer dropped as unmappable.
    assert not (prep_env["out_dir"] / "test_dropped_unmappable_cases.csv").exists()


def test_healed_zip_without_population_still_loses_its_cases(prep_env, monkeypatch,
                                                             fake_search_engine):
    """Healing supplies a coordinate but not a denominator. A ZIP-year with no
    population would make the Poisson rate undefined, so its cases go too."""
    engine = fake_search_engine({"00501": FakeZipResult(lat=40.8, lng=-73.0)})
    monkeypatch.setattr(prep_satscan, "SearchEngine", engine)

    prep_satscan_gui(nickname="test")
    cases = _read(prep_env["out_dir"], "test_full_cas.csv")
    assert "00501" not in set(cases["ZIP"])


def test_zero_population_zip_year_loses_its_cases(prep_env, monkeypatch):
    prep_env["state"]["populations"]["10001"] = 0
    prep_satscan_gui(nickname="test")
    cases = _read(prep_env["out_dir"], "test_full_cas.csv")
    assert "10001" not in set(cases["ZIP"])


# ==========================================================================
# population handling
# ==========================================================================

def test_backfills_2010_from_the_2011_acs(prep_env):
    """ACS 5-year starts at 2011. Using the 2010 Decennial instead would put a
    step change in the denominator, so 2011 is copied back to 2010.
    """
    prep_satscan_gui(nickname="test")
    pop = _read(prep_env["out_dir"], "test_full_pop.csv")
    assert 2010 in set(pop["Year"])
    p2010 = pop[(pop["Year"] == 2010) & (pop["ZIP"] == "10001")]["Population"].iloc[0]
    p2011 = pop[(pop["Year"] == 2011) & (pop["ZIP"] == "10001")]["Population"].iloc[0]
    assert p2010 == p2011


def test_population_covers_the_requested_years(prep_env):
    prep_satscan_gui(nickname="test", pop_years=range(2011, 2014))
    pop = _read(prep_env["out_dir"], "test_full_pop.csv")
    assert sorted(pop["Year"].unique()) == [2010, 2011, 2012, 2013]


def test_population_file_excludes_zips_without_coordinates(prep_env):
    prep_satscan_gui(nickname="test")
    pop = _read(prep_env["out_dir"], "test_full_pop.csv")
    geo = _read(prep_env["out_dir"], "test_full_geo.csv")
    assert set(pop["ZIP"]) <= set(geo["ZIP"])


def test_zips_are_five_digit_strings_everywhere(prep_env):
    """Leading zeros survive the whole pipeline, in every artifact."""
    prep_satscan_gui(nickname="test")
    for name in ("test_full_cas.csv", "test_full_pop.csv", "test_full_geo.csv"):
        zips = _read(prep_env["out_dir"], name)["ZIP"]
        assert zips.str.fullmatch(r"\d{5}").all(), name


# ==========================================================================
# analytical baseline
# ==========================================================================

def test_analytical_file_keeps_person_level_rows(prep_env):
    """`filter_satscan` builds regional enriched files from this one, so it has
    to stay person-level rather than aggregated."""
    prep_satscan_gui(nickname="test")
    analytical = pd.read_csv(prep_env["out_dir"] / "test_nvdrs_analytical.csv",
                             dtype={"DerivedZip": str})
    assert "DerivedZip" in analytical.columns
    assert "IncidentID" in analytical.columns
    assert len(analytical) == 3, "three mappable suicides on or after 2010"
