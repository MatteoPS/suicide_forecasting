"""Tests for `src.geospatial.st_dbscan`.

The synthetic geometry is built so that distances are readable by hand:
0.001 degrees of latitude is ~0.111 km, and 1.0 degree is ~111 km. Every
fixture below is a few points either well inside or well outside the epsilon
being tested, never near the boundary except where the boundary is the point.

The last section encodes the density-threshold problem called out in the
README, so that if `min_threshold` is ever recalibrated the tests say plainly
what changed.
"""
import pandas as pd
import pytest

from src.geospatial.st_dbscan import run_small_st_dbscan


# 0.001 deg latitude ~ 0.111 km, so these three sit inside a 0.5 km circle.
TIGHT_A = [("10001", 40.7500, -73.9900),
           ("10002", 40.7510, -73.9900),
           ("10003", 40.7520, -73.9900)]
# One degree north of A: ~111 km away, far outside any eps1_km used here.
TIGHT_B = [("20001", 41.7500, -73.9900),
           ("20002", 41.7510, -73.9900),
           ("20003", 41.7520, -73.9900)]


@pytest.fixture
def two_groups(write_satscan_files):
    """Two spatially separate, temporally tight groups plus one lone outlier."""
    cases = (
        [("10001", 1, "2015-01-01"), ("10002", 1, "2015-01-02"), ("10003", 1, "2015-01-03")]
        + [("20001", 1, "2015-01-01"), ("20002", 1, "2015-01-02"), ("20003", 1, "2015-01-03")]
        + [("30001", 1, "2016-06-01")]
    )
    geo = TIGHT_A + TIGHT_B + [("30001", 45.0000, -70.0000)]
    return write_satscan_files(cases, geo)


# ==========================================================================
# clustering behaviour
# ==========================================================================

def test_separates_two_spatially_distinct_groups(two_groups):
    out = run_small_st_dbscan(two_groups["cas"], two_groups["geo"],
                              eps1_km=1.5, eps2_days=14, min_threshold=2)
    assert out["cluster"].nunique() == 2


def test_drops_noise_points(two_groups):
    """The lone 2016 point in Maine belongs to nothing and must not come back."""
    out = run_small_st_dbscan(two_groups["cas"], two_groups["geo"],
                              eps1_km=1.5, eps2_days=14, min_threshold=2)
    assert "30001" not in set(out["zip"])
    assert (out["cluster"] != -1).all()


def test_each_group_keeps_all_its_members(two_groups):
    out = run_small_st_dbscan(two_groups["cas"], two_groups["geo"],
                              eps1_km=1.5, eps2_days=14, min_threshold=2)
    assert len(out) == 6
    assert out.groupby("cluster").size().tolist() == [3, 3]


def test_returns_expected_columns(two_groups):
    out = run_small_st_dbscan(two_groups["cas"], two_groups["geo"],
                              eps1_km=1.5, eps2_days=14, min_threshold=2)
    for col in ("zip", "n_case", "date", "days", "lat", "lon", "weight", "cluster","weight_type"):
        assert col in out.columns


def test_days_are_offsets_from_the_earliest_case(two_groups):
    out = run_small_st_dbscan(two_groups["cas"], two_groups["geo"],
                              eps1_km=1.5, eps2_days=14, min_threshold=2)
    assert out["days"].min() == 0
    assert out.loc[out["zip"] == "10003", "days"].iloc[0] == 2


# ==========================================================================
# the temporal epsilon
# ==========================================================================

def _one_zip_two_dates(write_satscan_files, date_a, date_b):
    """Same place, two dates. Only eps2_days can separate them."""
    cases = [("10001", 1, date_a), ("10002", 1, date_b)]
    geo = [("10001", 40.7500, -73.9900), ("10002", 40.7501, -73.9900)]
    return write_satscan_files(cases, geo)


def test_close_in_time_clusters_together(write_satscan_files):
    files = _one_zip_two_dates(write_satscan_files, "2015-01-01", "2015-01-05")
    out = run_small_st_dbscan(files["cas"], files["geo"],
                              eps1_km=1.5, eps2_days=14, min_threshold=1)
    assert out["cluster"].nunique() == 1


def test_far_apart_in_time_does_not_cluster(write_satscan_files):
    """Co-located but 100 days apart: contagion windows are the whole point."""
    files = _one_zip_two_dates(write_satscan_files, "2015-01-01", "2015-04-11")
    out = run_small_st_dbscan(files["cas"], files["geo"],
                              eps1_km=1.5, eps2_days=14, min_threshold=1)
    assert out["cluster"].nunique() == 2


def test_temporal_epsilon_is_inclusive(write_satscan_files):
    """`temporal_matrix <= eps2_days`, so exactly eps2_days apart still links."""
    files = _one_zip_two_dates(write_satscan_files, "2015-01-01", "2015-01-15")
    out = run_small_st_dbscan(files["cas"], files["geo"],
                              eps1_km=1.5, eps2_days=14, min_threshold=1)
    assert out["cluster"].nunique() == 1


# ==========================================================================
# the spatial epsilon
# ==========================================================================

def test_far_apart_in_space_does_not_cluster(write_satscan_files):
    cases = [("10001", 1, "2015-01-01"), ("20001", 1, "2015-01-02")]
    geo = [("10001", 40.7500, -73.9900), ("20001", 41.7500, -73.9900)]  # ~111 km
    files = write_satscan_files(cases, geo)
    out = run_small_st_dbscan(files["cas"], files["geo"],
                              eps1_km=1.5, eps2_days=14, min_threshold=1)
    assert out["cluster"].nunique() == 2


def test_distances_are_haversine_kilometres(write_satscan_files):
    """~5.6 km apart: outside eps1_km=1.5, inside eps1_km=10.

    If the earth radius or the radian conversion ever regresses, this flips.
    """
    cases = [("10001", 1, "2015-01-01"), ("10002", 1, "2015-01-02")]
    geo = [("10001", 40.7500, -73.9900), ("10002", 40.8000, -73.9900)]
    files = write_satscan_files(cases, geo)

    apart = run_small_st_dbscan(files["cas"], files["geo"],
                                eps1_km=1.5, eps2_days=14, min_threshold=1)
    together = run_small_st_dbscan(files["cas"], files["geo"],
                                   eps1_km=10.0, eps2_days=14, min_threshold=1)
    assert apart["cluster"].nunique() == 2
    assert together["cluster"].nunique() == 1


# ==========================================================================
# ZIP normalisation
# ==========================================================================

def test_zero_pads_zips_so_new_england_merges_survive(write_satscan_files):
    """ZIP 01001 arrives as the integer 1001 in one file and as '01001' in the
    other. Without the zfill the inner merge silently returns zero rows and
    the whole run reports no clusters."""
    cases = [(1001, 1, "2015-01-01"), (1002, 1, "2015-01-02")]
    geo = [("01001", 42.0700, -72.6200), ("01002", 42.0710, -72.6200)]
    files = write_satscan_files(cases, geo)
    out = run_small_st_dbscan(files["cas"], files["geo"],
                              eps1_km=1.5, eps2_days=14, min_threshold=1)
    assert len(out) == 2
    assert set(out["zip"]) == {"01001", "01002"}


def test_cases_without_coordinates_are_dropped(write_satscan_files):
    """The merge is an inner join, so an unmapped ZIP disappears rather than
    poisoning the distance matrix with NaN."""
    cases = [("10001", 1, "2015-01-01"), ("99999", 1, "2015-01-02")]
    geo = [("10001", 40.7500, -73.9900)]
    files = write_satscan_files(cases, geo)
    out = run_small_st_dbscan(files["cas"], files["geo"],
                              eps1_km=1.5, eps2_days=14, min_threshold=1)
    assert set(out["zip"]) == {"10001"}


# ==========================================================================
# weighting
# ==========================================================================

def test_without_population_weight_is_the_raw_count(write_satscan_files):
    cases = [("10001", 3, "2015-01-01"), ("10002", 2, "2015-01-02")]
    geo = [("10001", 40.7500, -73.9900), ("10002", 40.7501, -73.9900)]
    files = write_satscan_files(cases, geo)
    out = run_small_st_dbscan(files["cas"], files["geo"],
                              eps1_km=1.5, eps2_days=14, min_threshold=1)
    assert sorted(out["weight"]) == [2, 3]


def test_with_population_weight_is_incidence_per_100k(write_satscan_files):
    cases = [("10001", 1, "2015-01-01"), ("10002", 2, "2015-01-02")]
    geo = [("10001", 40.7500, -73.9900), ("10002", 40.7501, -73.9900)]
    pop = [("10001", 2015, 100_000), ("10002", 2015, 100_000)]
    files = write_satscan_files(cases, geo, pop)
    out = run_small_st_dbscan(files["cas"], files["geo"], files["pop"],
                              eps1_km=1.5, eps2_days=14, min_threshold=0.5)
    assert out.loc[out["zip"] == "10001", "weight"].iloc[0] == pytest.approx(1.0)
    assert out.loc[out["zip"] == "10002", "weight"].iloc[0] == pytest.approx(2.0)


def test_population_is_matched_on_the_case_year(write_satscan_files):
    """A ZIP that doubles in population must halve in incidence."""
    cases = [("10001", 1, "2015-01-01"), ("10001", 1, "2020-01-01")]
    geo = [("10001", 40.7500, -73.9900)]
    pop = [("10001", 2015, 100_000), ("10001", 2020, 200_000)]
    files = write_satscan_files(cases, geo, pop)
    out = run_small_st_dbscan(files["cas"], files["geo"], files["pop"],
                              eps1_km=1.5, eps2_days=14, min_threshold=0.5)
    by_year = out.set_index("year")["weight"]
    assert by_year[2015] == pytest.approx(1.0)
    assert by_year[2020] == pytest.approx(0.5)


def test_zero_population_yields_zero_weight_not_infinity(write_satscan_files):
    cases = [("10001", 1, "2015-01-01"), ("10002", 1, "2015-01-02")]
    geo = [("10001", 40.7500, -73.9900), ("10002", 40.7501, -73.9900)]
    pop = [("10001", 2015, 100_000), ("10002", 2015, 0)]
    files = write_satscan_files(cases, geo, pop)
    out = run_small_st_dbscan(files["cas"], files["geo"], files["pop"],
                              eps1_km=1.5, eps2_days=14, min_threshold=0.5)
    assert out.loc[out["zip"] == "10002", "weight"].iloc[0] == 0.0
    assert out["weight"].notna().all()


def test_missing_population_row_yields_zero_weight(write_satscan_files):
    """A ZIP-year absent from the .pop file merges to NaN, which must become 0
    rather than propagate into the sample weights."""
    cases = [("10001", 1, "2015-01-01"), ("10002", 1, "2015-01-02")]
    geo = [("10001", 40.7500, -73.9900), ("10002", 40.7501, -73.9900)]
    pop = [("10001", 2015, 100_000)]
    files = write_satscan_files(cases, geo, pop)
    out = run_small_st_dbscan(files["cas"], files["geo"], files["pop"],
                              eps1_km=1.5, eps2_days=14, min_threshold=0.5)
    assert out.loc[out["zip"] == "10002", "weight"].iloc[0] == 0.0


def test_rate_weighting_favours_small_zctas(write_satscan_files):
    """Documented and intended: one death in a 1,000-person ZCTA outweighs one
    death in a 100,000-person ZCTA by a factor of 100."""
    cases = [("10001", 1, "2015-01-01"), ("10002", 1, "2015-01-01")]
    geo = [("10001", 40.7500, -73.9900), ("10002", 40.7501, -73.9900)]
    pop = [("10001", 2015, 1_000), ("10002", 2015, 100_000)]
    files = write_satscan_files(cases, geo, pop)
    out = run_small_st_dbscan(files["cas"], files["geo"], files["pop"],
                              eps1_km=1.5, eps2_days=14, min_threshold=0.5)
    weights = out.set_index("zip")["weight"]
    assert weights["10001"] == pytest.approx(100 * weights["10002"])


# ==========================================================================
# the density threshold (README "Known issues")
# ==========================================================================

def test_threshold_binds_on_raw_counts(two_groups):
    """Three single cases per group: a threshold of 4 leaves nothing standing."""
    out = run_small_st_dbscan(two_groups["cas"], two_groups["geo"],
                              eps1_km=1.5, eps2_days=14, min_threshold=4)
    assert len(out) == 0


def test_low_threshold_makes_every_rate_weighted_point_a_core_point(write_satscan_files):
    """The bug recorded in README 'Known issues'.

    Sample weights are incidence per 100k. One case in a 100k-person ZCTA is a
    weight of 1.0, which already clears min_threshold=0.1 on its own, so an
    isolated record with no neighbours in space or time still comes back as a
    'cluster'. This is why the NYC run returned 981 clusters from 3,830
    records. If the threshold is ever raised by the two to three orders of
    magnitude the README calls for, this test should be updated, not deleted.
    """
    cases = [("10001", 1, "2015-01-01"), ("20001", 1, "2018-06-01")]
    geo = [("10001", 40.7500, -73.9900), ("20001", 45.0000, -70.0000)]
    pop = [("10001", 2015, 100_000), ("20001", 2018, 100_000)]
    files = write_satscan_files(cases, geo, pop)

    out = run_small_st_dbscan(files["cas"], files["geo"], files["pop"],
                              eps1_km=0.5, eps2_days=180, min_threshold=0.1)
    assert len(out) == 2
    assert out["cluster"].nunique() == 2, "two isolated points, two 'clusters'"


def test_raising_the_threshold_suppresses_isolated_points(write_satscan_files):
    """The same data with a threshold above a single point's weight."""
    cases = [("10001", 1, "2015-01-01"), ("20001", 1, "2018-06-01")]
    geo = [("10001", 40.7500, -73.9900), ("20001", 45.0000, -70.0000)]
    pop = [("10001", 2015, 100_000), ("20001", 2018, 100_000)]
    files = write_satscan_files(cases, geo, pop)

    out = run_small_st_dbscan(files["cas"], files["geo"], files["pop"],
                              eps1_km=0.5, eps2_days=180, min_threshold=1.5)
    assert len(out) == 0


def test_fractional_thresholds_survive_the_integer_cast(write_satscan_files):
    """DBSCAN needs an integer min_samples, so the implementation scales both
    the threshold and the weights by 1e6. Proportions must be preserved."""
    cases = [("10001", 1, "2015-01-01")]
    geo = [("10001", 40.7500, -73.9900)]
    pop = [("10001", 2015, 10_000_000)]  # weight = 0.01 per 100k
    files = write_satscan_files(cases, geo, pop)

    below = run_small_st_dbscan(files["cas"], files["geo"], files["pop"],
                                eps1_km=0.5, eps2_days=180, min_threshold=0.005)
    above = run_small_st_dbscan(files["cas"], files["geo"], files["pop"],
                                eps1_km=0.5, eps2_days=180, min_threshold=0.02)
    assert len(below) == 1
    assert len(above) == 0


# ==========================================================================
# result hygiene
# ==========================================================================

def test_result_is_a_copy_not_a_view(two_groups):
    """Callers append columns to the result; a view would raise
    SettingWithCopyWarning or silently fail."""
    out = run_small_st_dbscan(two_groups["cas"], two_groups["geo"],
                              eps1_km=1.5, eps2_days=14, min_threshold=2)
    out["annotation"] = "x"
    assert "annotation" in out.columns


def test_dates_are_parsed_to_timestamps(two_groups):
    out = run_small_st_dbscan(two_groups["cas"], two_groups["geo"],
                              eps1_km=1.5, eps2_days=14, min_threshold=2)
    assert pd.api.types.is_datetime64_any_dtype(out["date"])
