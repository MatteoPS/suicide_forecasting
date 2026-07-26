"""Cross-stage contract tests for the detection pipeline.

test_prep_satscan.py and test_filter_satscan.py test each stage's branches.
This file tests the *properties of the artifacts* the next stage relies on,
end to end:

    prep_satscan_gui  ->  create_regional_satscan_files  ->  run_small_st_dbscan

These are the invariants that, when violated, either abort SaTScan outright or
- worse - produce a plausible-looking relative risk from a broken denominator.
Everything runs on the synthetic fixtures in conftest; no real data is read.
"""
import pandas as pd
import pytest

from src.geospatial import filter_satscan
from src.geospatial.filter_satscan import create_regional_satscan_files
from src.geospatial.prep_satscan import prep_satscan_gui
from src.geospatial.satscan_io import read_satscan_cases, read_satscan_geo
from src.geospatial.st_dbscan import run_small_st_dbscan

NICKNAME = "pipeline"


@pytest.fixture
def global_artifacts(prep_env):
    """Stage 1: the four global SaTScan files."""
    prep_satscan_gui(nickname=NICKNAME)
    out = prep_env["out_dir"]
    return {
        "dir": out,
        "root": prep_env["root"],
        "cases": pd.read_csv(out / f"{NICKNAME}_full_cas.csv", dtype={"ZIP": str}),
        "pop": pd.read_csv(out / f"{NICKNAME}_full_pop.csv", dtype={"ZIP": str}),
        "geo": pd.read_csv(out / f"{NICKNAME}_full_geo.csv", dtype={"ZIP": str}),
    }


@pytest.fixture
def regional_artifacts(global_artifacts, tmp_path, monkeypatch):
    """Stage 2: the same four files, sliced to New York."""
    mapping = pd.DataFrame(
        {
            "ZIP": ["10001", "10002", "10003"],
            "state": ["NY", "NY", "NY"],
            "county": ["New York County", "New York County", "Kings County"],
        }
    )
    map_path = tmp_path / "zip_to_state_county.csv"
    mapping.to_csv(map_path, index=False)

    monkeypatch.setattr(filter_satscan, "PROJECT_ROOT", global_artifacts["root"])
    monkeypatch.setattr(filter_satscan, "get_data_path", lambda *a, **k: map_path)

    summary, dropped = create_regional_satscan_files("nyc", NICKNAME, state_names=["NY"])
    region_dir = global_artifacts["dir"] / "nyc"
    return {
        "dir": region_dir,
        "summary": summary,
        "dropped": dropped,
        "cases": pd.read_csv(region_dir / "nyc_cas.csv", dtype={"ZIP": str}),
        "pop": pd.read_csv(region_dir / "nyc_pop.csv", dtype={"ZIP": str}),
        "geo": pd.read_csv(region_dir / "nyc_geo.csv", dtype={"ZIP": str}),
    }


# ==========================================================================
# schema
# ==========================================================================

def test_expected_columns(global_artifacts):
    """SaTScan reads these files positionally; column order is the contract."""
    assert list(global_artifacts["cases"].columns) == ["ZIP", "Cases", "DeathDate"]
    assert list(global_artifacts["pop"].columns) == ["ZIP", "Year", "Population"]
    assert list(global_artifacts["geo"].columns) == ["ZIP", "Latitude", "Longitude"]


def test_column_types(global_artifacts):
    assert pd.api.types.is_integer_dtype(global_artifacts["cases"]["Cases"])
    assert pd.api.types.is_numeric_dtype(global_artifacts["pop"]["Population"])
    assert pd.api.types.is_numeric_dtype(global_artifacts["geo"]["Latitude"])
    assert pd.api.types.is_numeric_dtype(global_artifacts["geo"]["Longitude"])


def test_no_missing_values_anywhere(global_artifacts):
    """SaTScan has no missing-value encoding; a blank field aborts the run."""
    for name in ("cases", "pop", "geo"):
        assert not global_artifacts[name].isna().any().any(), name


# ==========================================================================
# value ranges
# ==========================================================================

def test_no_negative_counts(global_artifacts):
    assert (global_artifacts["cases"]["Cases"] > 0).all()


def test_no_negative_population(global_artifacts):
    assert (global_artifacts["pop"]["Population"] >= 0).all()


def test_coordinates_are_plausible(global_artifacts):
    lat = global_artifacts["geo"]["Latitude"]
    lon = global_artifacts["geo"]["Longitude"]
    assert lat.between(-90, 90).all()
    assert lon.between(-180, 180).all()


def test_date_range_starts_at_the_study_window(global_artifacts):
    """The README scopes the work to 2010-2023."""
    dates = pd.to_datetime(global_artifacts["cases"]["DeathDate"])
    assert dates.min() >= pd.Timestamp("2010-01-01")


def test_geographic_ids_valid(global_artifacts):
    for name in ("cases", "pop", "geo"):
        zips = global_artifacts[name]["ZIP"]
        assert zips.str.fullmatch(r"\d{5}").all(), name


# ==========================================================================
# referential integrity between the three files
# ==========================================================================

def test_every_case_zip_has_a_coordinate(global_artifacts):
    """A location ID in the case file with no row in the coordinate file is a
    hard SaTScan error."""
    assert set(global_artifacts["cases"]["ZIP"]) <= set(global_artifacts["geo"]["ZIP"])


def test_every_population_zip_has_a_coordinate(global_artifacts):
    assert set(global_artifacts["pop"]["ZIP"]) <= set(global_artifacts["geo"]["ZIP"])


def test_every_case_zip_year_has_a_population(global_artifacts):
    """Without a denominator the Poisson expectation is undefined, and SaTScan
    would silently treat the ZIP-year as zero-population."""
    cases = global_artifacts["cases"].copy()
    cases["Year"] = pd.to_datetime(cases["DeathDate"]).dt.year
    merged = cases.merge(global_artifacts["pop"], on=["ZIP", "Year"], how="left")
    assert merged["Population"].notna().all()
    assert (merged["Population"] > 0).all()


def test_no_duplicate_zip_year_in_the_population_file(global_artifacts):
    """A duplicated ZIP-year would double the denominator for that cell."""
    pop = global_artifacts["pop"]
    assert not pop.duplicated(subset=["ZIP", "Year"]).any()


def test_no_duplicate_zip_in_the_coordinate_file(global_artifacts):
    assert not global_artifacts["geo"]["ZIP"].duplicated().any()


def test_no_duplicate_zip_date_in_the_case_file(global_artifacts):
    """Cases are pre-aggregated; a repeat row would be counted twice."""
    cases = global_artifacts["cases"]
    assert not cases.duplicated(subset=["ZIP", "DeathDate"]).any()


def test_population_covers_every_year_in_the_case_file(global_artifacts):
    case_years = set(pd.to_datetime(global_artifacts["cases"]["DeathDate"]).dt.year)
    assert case_years <= set(global_artifacts["pop"]["Year"])


# ==========================================================================
# geographic scale linkage: global -> regional
# ==========================================================================

def test_regional_files_are_a_subset_of_the_global_ones(global_artifacts, regional_artifacts):
    for name in ("cases", "pop", "geo"):
        assert set(regional_artifacts[name]["ZIP"]) <= set(global_artifacts[name]["ZIP"]), name


def test_regional_subsetting_does_not_change_case_counts(global_artifacts, regional_artifacts):
    """Slicing to a region must filter rows, never re-aggregate them."""
    merged = regional_artifacts["cases"].merge(
        global_artifacts["cases"], on=["ZIP", "DeathDate"], suffixes=("_reg", "_glob")
    )
    assert len(merged) == len(regional_artifacts["cases"])
    assert (merged["Cases_reg"] == merged["Cases_glob"]).all()


def test_regional_subsetting_does_not_change_population(global_artifacts, regional_artifacts):
    merged = regional_artifacts["pop"].merge(
        global_artifacts["pop"], on=["ZIP", "Year"], suffixes=("_reg", "_glob")
    )
    assert (merged["Population_reg"] == merged["Population_glob"]).all()


def test_regional_files_preserve_referential_integrity(regional_artifacts):
    assert set(regional_artifacts["cases"]["ZIP"]) <= set(regional_artifacts["geo"]["ZIP"])
    assert set(regional_artifacts["pop"]["ZIP"]) == set(regional_artifacts["geo"]["ZIP"])


def test_regional_summary_accounts_for_every_usable_zip(regional_artifacts):
    """Every ZCTA carried into the analysis appears in the summary, including
    the ones with zero cases - they are the Poisson baseline."""
    assert set(regional_artifacts["summary"]["ZIP"]) == set(regional_artifacts["geo"]["ZIP"])


def test_regional_summary_case_totals_match_the_case_file(regional_artifacts):
    expected = regional_artifacts["cases"].groupby("ZIP")["Cases"].sum()
    actual = regional_artifacts["summary"].set_index("ZIP")["Total_Cases"]
    for zip_code, total in expected.items():
        assert actual[zip_code] == total


# ==========================================================================
# stage 3: the artifacts feed ST-DBSCAN
# ==========================================================================

def _to_satscan_format(csv_path, out_path):
    """Strip the header and re-emit whitespace-delimited.

    This conversion is not in the repo. `create_regional_satscan_files` writes
    headed CSVs; `run_small_st_dbscan` reads headerless whitespace files. Today
    that gap is bridged by hand (SaTScan's own export). Worth automating.
    """
    frame = pd.read_csv(csv_path, dtype=str)
    out_path.write_text(
        "\n".join(" ".join(row) for row in frame.itertuples(index=False)) + "\n"
    )
    return out_path


def test_regional_csv_is_not_directly_readable_by_st_dbscan(regional_artifacts):
    """Documents the format gap between stage 2 and stage 3.

    Reading the CSV as if it were a .cas file yields a single mangled column,
    because the file is comma-delimited and carries a header row.
    """
    raw = pd.read_csv(regional_artifacts["dir"] / "nyc_cas.csv",
                      header=None, names=["zip", "n_case", "date"], sep=r"\s+")
    assert raw["n_case"].isna().all(), "everything landed in the first column"


def test_converted_artifacts_round_trip_through_st_dbscan(regional_artifacts, tmp_path):
    cas = _to_satscan_format(regional_artifacts["dir"] / "nyc_cas.csv", tmp_path / "nyc.cas")
    geo = _to_satscan_format(regional_artifacts["dir"] / "nyc_geo.csv", tmp_path / "nyc.geo")

    cases = read_satscan_cases(cas)
    coords = read_satscan_geo(geo)
    assert set(cases["zip"]) <= set(coords["zip"])

    clusters = run_small_st_dbscan(cas, geo, eps1_km=1.5, eps2_days=14, min_threshold=1)
    assert "cluster" in clusters.columns


def test_clustering_invents_no_records(regional_artifacts, tmp_path):
    """No data leakage in the structural sense: the clusterer may drop noise
    points, but it must never emit a ZIP or a date that was not in the input."""
    cas = _to_satscan_format(regional_artifacts["dir"] / "nyc_cas.csv", tmp_path / "nyc.cas")
    geo = _to_satscan_format(regional_artifacts["dir"] / "nyc_geo.csv", tmp_path / "nyc.geo")

    cases = read_satscan_cases(cas)
    clusters = run_small_st_dbscan(cas, geo, eps1_km=1.5, eps2_days=14, min_threshold=1)

    assert set(clusters["zip"]) <= set(cases["zip"])
    assert set(clusters["date"]) <= set(cases["date"])
    assert len(clusters) <= len(cases)


def test_clustering_never_returns_noise_labels(regional_artifacts, tmp_path):
    cas = _to_satscan_format(regional_artifacts["dir"] / "nyc_cas.csv", tmp_path / "nyc.cas")
    geo = _to_satscan_format(regional_artifacts["dir"] / "nyc_geo.csv", tmp_path / "nyc.geo")
    clusters = run_small_st_dbscan(cas, geo, eps1_km=1.5, eps2_days=14, min_threshold=1)
    assert (clusters["cluster"] >= 0).all()


# ==========================================================================
# documented losses
# ==========================================================================

def test_dropped_cases_are_recorded_not_silent(global_artifacts):
    """Cases in unmappable ZIPs leave the analysis. The README promises they
    are written to a file rather than dropped silently, and the missingness
    file is what makes the numerator reconcilable."""
    dropped_path = global_artifacts["dir"] / f"{NICKNAME}_dropped_unmappable_cases.csv"
    assert dropped_path.exists()

    dropped = pd.read_csv(dropped_path, dtype={"ZIP": str})
    assert dropped["Cases"].sum() > 0
    assert set(dropped["ZIP"]).isdisjoint(set(global_artifacts["cases"]["ZIP"]))


def test_analytical_file_matches_the_case_file_zip_set(global_artifacts):
    """`filter_satscan` builds the enriched regional file from the analytical
    baseline, so the two have to agree on which ZIPs survived."""
    analytical = pd.read_csv(
        global_artifacts["dir"] / f"{NICKNAME}_nvdrs_analytical.csv",
        dtype={"DerivedZip": str},
    )
    assert set(global_artifacts["cases"]["ZIP"]) <= set(analytical["DerivedZip"])
