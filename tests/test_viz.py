"""Tests for `src.utils.viz`.

Plots are hard to assert on, so these tests check the things that actually go
wrong in practice: files getting written, columns getting silently skipped,
and top-N filtering picking the wrong clusters. conftest forces the Agg
backend, so nothing tries to open a window.
"""
import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from src.utils.viz import (
    plot_daily_case_totals,
    plot_forecast_grid,
    plot_grouped_series,
    plot_source_comparison,
    plot_zip_population_pivot,
    visualize_st_dbscan_clusters,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def monthly_counts():
    index = pd.date_range("2015-01-31", periods=12, freq="ME")
    return pd.DataFrame(
        {
            "Pima, AZ": np.arange(12),
            "Pinal, AZ": np.arange(12) * 2,
            "Polk, IA": np.arange(12) + 5,
        },
        index=index,
    )


# ==========================================================================
# plot_grouped_series
# ==========================================================================

def test_grouped_series_draws_one_line_per_column(monthly_counts):
    groups = {"Arizona": ("Pima, AZ", "Pinal, AZ"), "Iowa": ("Polk, IA",)}
    ax = plot_grouped_series(monthly_counts, groups, "Counties")
    assert len(ax.lines) == 3


def test_grouped_series_labels_one_entry_per_group(monthly_counts):
    """Twenty counties across seven states should give seven legend entries."""
    groups = {"Arizona": ("Pima, AZ", "Pinal, AZ"), "Iowa": ("Polk, IA",)}
    ax = plot_grouped_series(monthly_counts, groups, "Counties")
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["Arizona", "Iowa"]


def test_grouped_series_colours_a_group_consistently(monthly_counts):
    groups = {"Arizona": ("Pima, AZ", "Pinal, AZ"), "Iowa": ("Polk, IA",)}
    ax = plot_grouped_series(monthly_counts, groups, "Counties")
    assert ax.lines[0].get_color() == ax.lines[1].get_color()
    assert ax.lines[0].get_color() != ax.lines[2].get_color()


def test_grouped_series_reports_absent_columns(monthly_counts, capsys):
    """NVDRS county coverage varies by site and year, so a requested county
    may genuinely not exist. It should be named, not silently dropped."""
    groups = {"Arizona": ("Pima, AZ", "Maricopa, AZ")}
    ax = plot_grouped_series(monthly_counts, groups, "Counties")
    assert len(ax.lines) == 1
    assert "Maricopa, AZ" in capsys.readouterr().out


def test_grouped_series_still_labels_a_group_whose_first_column_is_absent(monthly_counts):
    groups = {"Arizona": ("Maricopa, AZ", "Pima, AZ")}
    ax = plot_grouped_series(monthly_counts, groups, "Counties")
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["Arizona"]


def test_grouped_series_accepts_an_existing_axes(monthly_counts):
    _, ax = plt.subplots()
    returned = plot_grouped_series(monthly_counts, {"Iowa": ("Polk, IA",)}, "t", ax=ax)
    assert returned is ax


# ==========================================================================
# plot_source_comparison
# ==========================================================================

@pytest.fixture
def two_sources():
    index = pd.date_range("2015-01-31", periods=24, freq="ME")
    wonder = pd.DataFrame({"New York": np.arange(24), "Utah": np.arange(24)}, index=index)
    nvdrs = pd.DataFrame({"New York": np.arange(24) * 0.9}, index=index)
    return {"WONDER": wonder, "NVDRS": nvdrs}


def test_source_comparison_makes_one_panel_per_column(two_sources):
    fig, axes = plot_source_comparison(two_sources, ["New York", "Utah"], "t")
    assert len(fig.axes) == 2


def test_source_comparison_overlays_every_available_source(two_sources):
    fig, axes = plot_source_comparison(two_sources, ["New York"], "t")
    assert len(fig.axes[0].lines) == 2


def test_source_comparison_reports_a_missing_series(two_sources, capsys):
    """The notebook's `if state in df.columns` guard made a missing NVDRS
    series look identical to a genuine run of zeros. Now it says so."""
    plot_source_comparison(two_sources, ["Utah"], "t")
    out = capsys.readouterr().out
    assert "NVDRS" in out and "Utah" in out


def test_source_comparison_removes_unused_panels(two_sources):
    """Three columns over two per row leaves one empty axes to delete."""
    fig, _ = plot_source_comparison(
        two_sources, ["New York", "Utah", "New York"], "t", ncols=2
    )
    assert len(fig.axes) == 3


def test_source_comparison_honours_the_date_window(two_sources):
    fig, _ = plot_source_comparison(
        two_sources, ["New York"], "t", start="2015-06-01", end="2015-12-31"
    )
    xdata = fig.axes[0].lines[0].get_xdata()
    assert len(xdata) == 7


def test_source_comparison_applies_requested_colours(two_sources):
    fig, _ = plot_source_comparison(
        two_sources, ["New York"], "t", colors={"WONDER": "black", "NVDRS": "orange"}
    )
    assert fig.axes[0].lines[0].get_color() == "black"


# ==========================================================================
# plot_daily_case_totals
# ==========================================================================

def test_daily_case_totals_plots_one_line(write_satscan_files):
    from src.geospatial.satscan_io import read_satscan_cases

    files = write_satscan_files(
        [("10001", 2, "2015-01-01"), ("10002", 3, "2015-01-01"),
         ("10001", 1, "2015-01-02")],
        [("10001", 40.75, -73.99)],
    )
    ax = plot_daily_case_totals(read_satscan_cases(files["cas"]))
    assert len(ax.lines) == 1
    assert list(ax.lines[0].get_ydata()) == [5, 1]


# ==========================================================================
# plot_forecast_grid
# ==========================================================================

class FakeSeries:
    """Duck-type for a darts TimeSeries: slicing plus `.plot(ax=, label=)`."""

    def __init__(self, values):
        self.values_ = list(values)

    def __getitem__(self, item):
        return FakeSeries(self.values_[item])

    def plot(self, ax=None, label=None, lw=None):
        ax.plot(range(len(self.values_)), self.values_, label=label)


def test_forecast_grid_draws_actual_plus_every_model():
    actual = {"New York": FakeSeries(range(20))}
    forecasts = {"New York": {"AutoARIMA": FakeSeries(range(5)),
                              "LightGBM (Multi)": FakeSeries(range(5))}}
    fig, axes = plot_forecast_grid(actual, forecasts, ["New York"], plot_len=10,
                                   title="Monthly Forecasts")
    assert len(axes[0].lines) == 3


def test_forecast_grid_plots_only_the_requested_tail():
    actual = {"New York": FakeSeries(range(100))}
    forecasts = {"New York": {}}
    fig, axes = plot_forecast_grid(actual, forecasts, ["New York"], plot_len=12, title="t")
    assert len(axes[0].lines[0].get_ydata()) == 12


def test_forecast_grid_hides_unused_panels():
    actual = {s: FakeSeries(range(20)) for s in ("a", "b", "c")}
    forecasts = {s: {} for s in actual}
    fig, axes = plot_forecast_grid(actual, forecasts, ["a", "b", "c"], 10, "t", ncols=2)
    assert axes[3].axison is False


def test_forecast_grid_titles_each_panel():
    actual = {"Utah": FakeSeries(range(20))}
    fig, axes = plot_forecast_grid(actual, {"Utah": {}}, ["Utah"], 10, "t")
    assert axes[0].get_title() == "Utah"


# ==========================================================================
# visualize_st_dbscan_clusters
# ==========================================================================

@pytest.fixture
def cluster_frame():
    """Three clusters of decreasing size, so top_n selection is checkable."""
    rows = []
    for cluster, size in [(0, 5), (1, 3), (2, 1)]:
        for i in range(size):
            rows.append({
                "zip": f"1000{cluster}", "n_case": 1,
                "date": pd.Timestamp("2015-01-01") + pd.Timedelta(days=i),
                "days": i, "lat": 40.75 + cluster * 0.01, "lon": -73.99,
                "cluster": cluster,
            })
    return pd.DataFrame(rows)


def test_visualize_writes_three_html_files(cluster_frame, tmp_path):
    visualize_st_dbscan_clusters(cluster_frame, output_dir=str(tmp_path), nickname="nyc")
    for name in ("nyc_1_cluster_map.html", "nyc_2_cluster_3d.html",
                 "nyc_3_cluster_timeline.html"):
        assert (tmp_path / name).exists(), name


def test_visualize_creates_the_output_directory(cluster_frame, tmp_path):
    target = tmp_path / "outputs" / "nested"
    visualize_st_dbscan_clusters(cluster_frame, output_dir=str(target))
    assert target.is_dir()


def test_visualize_omits_the_prefix_without_a_nickname(cluster_frame, tmp_path):
    visualize_st_dbscan_clusters(cluster_frame, output_dir=str(tmp_path))
    assert (tmp_path / "1_cluster_map.html").exists()


def test_visualize_top_n_keeps_the_largest_clusters(cluster_frame, tmp_path, capsys):
    visualize_st_dbscan_clusters(cluster_frame, output_dir=str(tmp_path), top_n=2)
    out = capsys.readouterr().out
    assert "Showing Top 2 of 3 Clusters" in out


def test_visualize_top_n_above_the_cluster_count_is_a_no_op(cluster_frame, tmp_path, capsys):
    visualize_st_dbscan_clusters(cluster_frame, output_dir=str(tmp_path), top_n=99)
    assert "Total Clusters: 3" in capsys.readouterr().out


def test_visualize_does_not_mutate_its_input(cluster_frame, tmp_path):
    before = cluster_frame.copy()
    visualize_st_dbscan_clusters(cluster_frame, output_dir=str(tmp_path), top_n=2)
    pd.testing.assert_frame_equal(cluster_frame, before)


# ==========================================================================
# plot_zip_population_pivot
# ==========================================================================

@pytest.fixture
def population_pivot():
    return pd.DataFrame(
        {
            "DerivedZip": ["01001", "10001"],
            "2015": [1000, 2000],
            "2016": [1100, 2100],
            "County": ["Hampden", "New York"],
            "State": ["MA", "NY"],
        }
    )


def test_zip_pivot_plots_one_line_per_zip(population_pivot):
    plot_zip_population_pivot(population_pivot, ["01001", "10001"])
    assert len(plt.gca().lines) == 2


def test_zip_pivot_accepts_an_unpadded_zip(population_pivot):
    """'01001' passed as the integer 1001 must still be found."""
    plot_zip_population_pivot(population_pivot, [10001])
    assert len(plt.gca().lines) == 1


def test_zip_pivot_warns_about_an_unknown_zip(population_pivot, capsys):
    plot_zip_population_pivot(population_pivot, ["99999"])
    assert "not found" in capsys.readouterr().out


def test_zip_pivot_ignores_non_year_columns(population_pivot):
    """County and State are text; plotting them would raise."""
    plot_zip_population_pivot(population_pivot, ["10001"])
    assert list(plt.gca().lines[0].get_ydata()) == [2000, 2100]


def test_zip_pivot_labels_with_county_and_state(population_pivot):
    plot_zip_population_pivot(population_pivot, ["10001"],
                                county_col="County", state_col="State")
    labels = [t.get_text() for t in plt.gca().get_legend().get_texts()]
    assert labels == ["ZIP 10001 (New York, NY)"]
