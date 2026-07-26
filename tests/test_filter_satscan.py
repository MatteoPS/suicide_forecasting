"""Tests for `src.geospatial.filter_satscan.create_regional_satscan_files`.

The function slices the four global SaTScan artifacts down to a region. The
interesting behaviour is which ZIPs it refuses to carry over: a ZIP needs a
coordinate *and* a non-zero population, otherwise SaTScan either aborts or
computes an undefined Poisson rate.

Fixtures write miniature global artifacts into tmp_path; no real output is read.
"""
import pandas as pd
import pytest

from src.geospatial import filter_satscan
from src.geospatial.filter_satscan import create_regional_satscan_files

NICKNAME = "global"


@pytest.fixture
def region_env(tmp_path, monkeypatch):
    """Five NY ZIPs and one NJ ZIP, each present for a different reason.

      10001  New York County, NY   coords + population + 2 cases
      10002  New York County, NY   coords + population + no cases
      10003  Kings County, NY      coords + population + 1 case
      10004  New York County, NY   NO coordinate          -> must be dropped
      10005  New York County, NY   population 0           -> must be dropped
      07001  Middlesex County, NJ  a different state entirely
    """
    base = tmp_path / "data" / "processed" / "satscan"
    base.mkdir(parents=True)
    monkeypatch.setattr(filter_satscan, "PROJECT_ROOT", tmp_path)

    mapping = pd.DataFrame(
        {
            "ZIP": ["10001", "10002", "10003", "10004", "10005", "07001"],
            "state": ["NY", "NY", "NY", "NY", "NY", "NJ"],
            "county": ["New York County", "New York County", "Kings County",
                       "New York County", "New York County", "Middlesex County"],
        }
    )
    map_path = tmp_path / "uszipcode_static.csv"
    mapping.to_csv(map_path, index=False)
    monkeypatch.setattr(filter_satscan, "get_data_path", lambda *a, **k: map_path)

    # 10004 deliberately absent: no coordinate.
    pd.DataFrame(
        {
            "ZIP": ["10001", "10002", "10003", "10005", "07001"],
            "Latitude": [40.75, 40.751, 40.68, 40.706, 40.58],
            "Longitude": [-73.99, -73.99, -73.99, -74.01, -74.28],
        }
    ).to_csv(base / f"{NICKNAME}_full_geo.csv", index=False)

    pop_rows = []
    for zip_code, pop in [("10001", 1000), ("10002", 2000), ("10003", 3000),
                          ("10004", 100), ("10005", 0), ("07001", 500)]:
        for year in (2015, 2016):
            pop_rows.append({"ZIP": zip_code, "Year": year, "Population": pop})
    pd.DataFrame(pop_rows).to_csv(base / f"{NICKNAME}_full_pop.csv", index=False)

    pd.DataFrame(
        {
            "ZIP": ["10001", "10001", "10003", "10005", "07001"],
            "Cases": [1, 1, 1, 5, 3],
            "DeathDate": ["2015-03-01", "2015-04-01", "2015-05-01",
                          "2015-06-01", "2015-07-01"],
        }
    ).to_csv(base / f"{NICKNAME}_full_cas.csv", index=False)

    pd.DataFrame(
        {
            "IncidentID": ["1", "2", "3", "4"],
            "DerivedZip": ["10001", "10001", "10003", "07001"],
            "DeathDate": ["2015-03-01", "2015-04-01", "2015-05-01", "2015-07-01"],
            "Sex": ["Male", "Female", "Male", "Female"],
        }
    ).to_csv(base / f"{NICKNAME}_nvdrs_analytical.csv", index=False)

    return {"base": base, "tmp": tmp_path}


def _out(region_env, regional, suffix):
    return pd.read_csv(
        region_env["base"] / regional / f"{regional}_{suffix}.csv", dtype={"ZIP": str}
    )


# ==========================================================================
# region selection
# ==========================================================================

def test_requires_a_region(region_env):
    with pytest.raises(ValueError, match="Provide either"):
        create_regional_satscan_files("nowhere", NICKNAME)


def test_state_filter_excludes_other_states(region_env):
    summary, _ = create_regional_satscan_files("ny", NICKNAME, state_names=["NY"])
    assert "07001" not in set(summary["ZIP"])


def test_state_filter_keeps_every_usable_zip_in_the_state(region_env):
    summary, _ = create_regional_satscan_files("ny", NICKNAME, state_names=["NY"])
    assert set(summary["ZIP"]) == {"10001", "10002", "10003"}


def test_county_filter_narrows_within_a_state(region_env):
    summary, _ = create_regional_satscan_files(
        "manhattan", NICKNAME, state_names=["NY"], county_names=["New York County"]
    )
    assert set(summary["ZIP"]) == {"10001", "10002"}


def test_county_filter_without_a_state_matches_the_name_everywhere(region_env):
    """County names repeat across states, so this is a wide net by design."""
    summary, _ = create_regional_satscan_files(
        "middlesex", NICKNAME, county_names=["Middlesex County"]
    )
    assert set(summary["ZIP"]) == {"07001"}


def test_multiple_states(region_env):
    summary, _ = create_regional_satscan_files(
        "tristate", NICKNAME, state_names=["NY", "NJ"]
    )
    assert set(summary["ZIP"]) == {"10001", "10002", "10003", "07001"}


# ==========================================================================
# ZIP eligibility
# ==========================================================================

def test_drops_zips_without_coordinates(region_env):
    _, dropped = create_regional_satscan_files("ny", NICKNAME, state_names=["NY"])
    assert "10004" in dropped


def test_drops_zips_with_zero_population(region_env):
    """10005 has five cases but no denominator; keeping it would make the
    Poisson rate undefined."""
    _, dropped = create_regional_satscan_files("ny", NICKNAME, state_names=["NY"])
    assert "10005" in dropped


def test_dropped_zip_cases_do_not_reach_the_case_file(region_env):
    create_regional_satscan_files("ny", NICKNAME, state_names=["NY"])
    cases = _out(region_env, "ny", "cas")
    assert "10005" not in set(cases["ZIP"])
    assert cases["Cases"].sum() == 3, "2 in 10001 + 1 in 10003"


def test_dropped_list_contains_nothing_else(region_env):
    _, dropped = create_regional_satscan_files("ny", NICKNAME, state_names=["NY"])
    assert set(dropped) == {"10004", "10005"}


# ==========================================================================
# output files
# ==========================================================================

def test_writes_four_regional_files(region_env):
    create_regional_satscan_files("ny", NICKNAME, state_names=["NY"])
    out_dir = region_env["base"] / "ny"
    for suffix in ("geo", "pop", "cas", "enriched_cas"):
        assert (out_dir / f"ny_{suffix}.csv").exists(), suffix


def test_regional_files_agree_on_their_zip_set(region_env):
    """SaTScan cross-references the three files by location ID; a ZIP present
    in one and absent from another is a hard error at run time."""
    create_regional_satscan_files("ny", NICKNAME, state_names=["NY"])
    geo = set(_out(region_env, "ny", "geo")["ZIP"])
    pop = set(_out(region_env, "ny", "pop")["ZIP"])
    cas = set(_out(region_env, "ny", "cas")["ZIP"])
    assert geo == pop == {"10001", "10002", "10003"}
    assert cas <= geo


def test_enriched_file_renames_derivedzip_to_zip(region_env):
    create_regional_satscan_files("ny", NICKNAME, state_names=["NY"])
    enriched = _out(region_env, "ny", "enriched_cas")
    assert "ZIP" in enriched.columns
    assert "DerivedZip" not in enriched.columns


def test_enriched_file_stays_person_level(region_env):
    create_regional_satscan_files("ny", NICKNAME, state_names=["NY"])
    enriched = _out(region_env, "ny", "enriched_cas")
    assert len(enriched) == 3
    assert set(enriched["IncidentID"]) == {1, 2, 3}


def test_output_directory_is_named_after_the_region(region_env):
    create_regional_satscan_files("upstate", NICKNAME, state_names=["NY"])
    assert (region_env["base"] / "upstate").is_dir()


def test_zips_keep_their_leading_zeros(region_env):
    create_regional_satscan_files("nj", NICKNAME, state_names=["NJ"])
    assert set(_out(region_env, "nj", "geo")["ZIP"]) == {"07001"}


# ==========================================================================
# summary
# ==========================================================================

def test_summary_includes_zips_with_no_cases(region_env):
    """A ZCTA with population and zero deaths is a real observation, not a
    missing row; the Poisson baseline needs it."""
    summary, _ = create_regional_satscan_files("ny", NICKNAME, state_names=["NY"])
    row = summary[summary["ZIP"] == "10002"]
    assert len(row) == 1
    assert row["Total_Cases"].iloc[0] == 0


def test_summary_sums_cases_across_dates(region_env):
    summary, _ = create_regional_satscan_files("ny", NICKNAME, state_names=["NY"])
    assert summary.loc[summary["ZIP"] == "10001", "Total_Cases"].iloc[0] == 2


def test_summary_sums_population_across_years(region_env):
    summary, _ = create_regional_satscan_files("ny", NICKNAME, state_names=["NY"])
    assert summary.loc[summary["ZIP"] == "10001", "Total_Pop_14_Years"].iloc[0] == 2000


def test_summary_is_sorted_by_case_count(region_env):
    summary, _ = create_regional_satscan_files("ny", NICKNAME, state_names=["NY"])
    assert summary["Total_Cases"].is_monotonic_decreasing


def test_summary_columns(region_env):
    summary, _ = create_regional_satscan_files("ny", NICKNAME, state_names=["NY"])
    assert set(summary.columns) == {"ZIP", "Total_Cases", "Total_Pop_14_Years"}


def test_summary_excludes_dropped_zips(region_env):
    summary, _ = create_regional_satscan_files("ny", NICKNAME, state_names=["NY"])
    assert "10005" not in set(summary["ZIP"])


# ==========================================================================
# whitespace tolerance
# ==========================================================================

def test_tolerates_padded_state_and_county_names(region_env, tmp_path, monkeypatch):
    """The static uszipcode dump has trailing spaces in places; the function
    strips them before matching."""
    padded = pd.DataFrame(
        {"ZIP": ["10001"], "state": ["  NY "], "county": [" New York County  "]}
    )
    path = tmp_path / "padded.csv"
    padded.to_csv(path, index=False)
    monkeypatch.setattr(filter_satscan, "get_data_path", lambda *a, **k: path)

    summary, _ = create_regional_satscan_files(
        "ny", NICKNAME, state_names=["NY"], county_names=["New York County"]
    )
    assert set(summary["ZIP"]) == {"10001"}
