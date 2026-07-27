import numpy as np
import math
import os

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from scipy.spatial import ConvexHull

def plot_grouped_series(pivot_df, groups: dict, title: str, ylabel: str = "Incident Count",
                        xlabel: str = "Date", legend_title: str = "Group",
                        figsize=(15, 7), cmap: str = "tab10", ax=None):
    """Plots many columns, one colour per group, one legend entry per group.

    Moved out of notebooks/0.1. `groups` maps a group name to the columns
    belonging to it, e.g. {'Arizona': ('Pima, AZ', 'Pinal, AZ')}. Columns
    absent from `pivot_df` are skipped and reported, rather than raising -
    county coverage varies by NVDRS site and year.

    Returns the Axes so callers can adjust it further.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    colors = plt.get_cmap(cmap, len(groups))
    missing = []

    for group_idx, (group, columns) in enumerate(groups.items()):
        color = colors(group_idx)
        first_in_group = True
        for column in columns:
            if column not in pivot_df.columns:
                missing.append(column)
                continue
            # Only the first column of a group carries a legend label.
            ax.plot(pivot_df.index, pivot_df[column], color=color,
                    label=group if first_in_group else "_nolegend_")
            first_in_group = False

    if missing:
        print(f"Warning: {len(missing)} column(s) not in the data and not plotted: {missing}")

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.legend(title=legend_title)
    return ax


def plot_source_comparison(sources: dict, columns: list, title: str,
                            start=None, end=None, ncols: int = 2,
                            colors: dict = None, figsize_per_row: float = 2.0):
    """One small-multiple per column, overlaying several data sources.

    Compares NVDRS against CDC WONDER state by state. 
    `sources` maps a label to a DataFrame indexed 
    by date with one column per geography.

    Returns (fig, axes).
    """
    nrows = math.ceil(len(columns) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, figsize_per_row * nrows),
                            sharex=True, squeeze=False)
    axes = axes.flatten()

    styles = colors or {}
    missing = []

    for i, column in enumerate(columns):
        ax = axes[i]
        for label, frame in sources.items():
            if column not in frame.columns:
                missing.append((label, column))
                continue
            subset = frame.loc[start:end]
            ax.plot(subset.index, subset[column], lw=1, alpha=0.7,
                    color=styles.get(label),
                    label=label if i == 0 else "")
        ax.set_title(column, fontsize=9)
        ax.tick_params(axis="both", labelsize=8)

    for j in range(len(columns), len(axes)):
        fig.delaxes(axes[j])

    if missing:
        print(f"Warning: {len(missing)} series not plotted (source, column): {missing}")

    fig.suptitle(title, fontsize=12)
    fig.legend(loc="upper left")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, axes


def plot_daily_case_totals(cases: pd.DataFrame, title: str = "Overall Suicide Cases Over Time",
                           ylabel: str = "Total Cases", figsize=(12, 6), ax=None):
    """Plots the national daily total from a SaTScan case frame.

    Moved out of notebooks/1.2. Expects the output of
    `src.geospatial.satscan_io.read_satscan_cases`.
    """
    from src.geospatial.satscan_io import daily_case_totals

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    daily_case_totals(cases).plot(ax=ax, title=title, ylabel=ylabel)
    return ax


def plot_forecast_grid(actual: dict, forecasts: dict, columns: list, plot_len: int,
                       title: str, ncols: int = 2):
    """One panel per series, showing the observed tail plus each forecast.

    Moved out of notebooks/0.1. `actual` and `forecasts` are keyed by column
    name; `forecasts[column]` is itself a dict of {model name: prediction}.
    Values are darts TimeSeries, which supply their own `.plot(ax=...)`.

    Returns (fig, axes).
    """
    nrows = math.ceil(len(columns) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 4 * nrows),
                             constrained_layout=True, squeeze=False)
    axes = axes.flatten()
    fig.suptitle(title, fontsize=16)

    for i, column in enumerate(columns):
        actual[column][-plot_len:].plot(ax=axes[i], label="Actual")
        for model_name, prediction in forecasts[column].items():
            prediction.plot(ax=axes[i], label=model_name, lw=2)
        axes[i].set_title(column)
        axes[i].legend()

    for j in range(len(columns), len(axes)):
        axes[j].axis("off")

    return fig, axes


def plot_zip_population_pivot(pivot_df, zip_list,zip_colname='DerivedZip', county_col=None, state_col=None):
    plt.figure(figsize=(12, 6))
    
    # 1. Handle MultiIndex if present
    plot_df = pivot_df.copy()
    if zip_colname in plot_df.columns:
        plot_df = plot_df.set_index(zip_colname)
    elif isinstance(plot_df.index, pd.MultiIndex):
        plot_df = plot_df.reset_index().set_index(plot_df.index.names[0] if zip_colname not in plot_df.index.names else zip_colname)

    # 2. Isolate ONLY the year columns to prevent plotting text/metric columns
    year_cols = [c for c in plot_df.columns if str(c).isdigit()]
    
    # Ensure index values are strings for consistent lookup
    plot_df.index = plot_df.index.astype(str)
    
    for z in zip_list:
        z_str = str(z)  # Force input to string to handle zero-padding (e.g., '02138') Safely
        
        if z_str not in plot_df.index:
            print(f"Warning: ZIP {z} not found in dataset.")
            continue
            
        row = plot_df.loc[z_str]
        
        # If duplicate ZIP entries exist, grab the first one
        if isinstance(row, pd.DataFrame): 
            row = row.iloc[0]
            
        label = f"ZIP {z_str}"
        location_parts = []
        
        # 3. Extract location strings safely
        if county_col and county_col in plot_df.columns:
            county = row[county_col]
            if pd.notna(county) and str(county).strip(): 
                location_parts.append(str(county))
                
        if state_col and state_col in plot_df.columns:
            state = row[state_col]
            if pd.notna(state) and str(state).strip(): 
                location_parts.append(str(state))
                
        if location_parts:
            label += f" ({', '.join(location_parts)})"
            
        # 4. Plot only the isolated year columns
        plt.plot(year_cols, row[year_cols], marker='.', label=label)
            
    plt.title("Population Over Time by ZCTA")
    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def visualize_st_dbscan_clusters(df_clusters, output_dir="outputs", nickname="", top_n=None):
    """
    Generates three interactive HTML visualizations for ST-DBSCAN clusters.
    
    Parameters:
    -----------
    df_clusters : pd.DataFrame
        The output dataframe from run_small_st_dbscan containing the 'cluster' column.
    output_dir : str
        Directory to save the HTML files.
    nickname : str
        Prefix for the saved HTML files.
    top_n : int, optional
        If provided, only plots the top N clusters with the most total cases.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Format file prefix
    prefix = f"{nickname}_" if nickname else ""
    
    total_clusters_count = df_clusters['cluster'].nunique()

    # --- Filter for Top N clusters ---
    if top_n is not None and top_n < total_clusters_count:
        # Group by cluster and sum the cases (since we expanded rows, sizing by count works)
        cluster_sizes = df_clusters.groupby('cluster').size().reset_index(name='size')
        # Sort descending and take the top N
        top_clusters = cluster_sizes.sort_values(by='size', ascending=False).head(top_n)['cluster']
        # Filter the original dataframe
        df_clusters = df_clusters[df_clusters['cluster'].isin(top_clusters)].copy()
        title_suffix = f" (Showing Top {top_n} of {total_clusters_count} Clusters)"
    else:
        title_suffix = f" (Total Clusters: {total_clusters_count})"

    # Ensure cluster ID is a string so Plotly treats it as a discrete color category
    df_clusters = df_clusters.copy()
    df_clusters['cluster_id'] = 'Cluster ' + df_clusters['cluster'].astype(str)
    
    # Sort by date for better timeline rendering
    df_clusters = df_clusters.sort_values('date')

    print(f"Generating visualizations...{title_suffix}")

# ---------------------------------------------------------
    # 1. 2D Interactive Map (Spatial Focus)
    # ---------------------------------------------------------
    # Points are the individual cases; the shaded polygon behind each cluster
    # is its convex hull, i.e. the footprint the algorithm actually found
    # rather than a circle imposed on it.
    fig_map = px.scatter_map(
        df_clusters,
        lat="lat",
        lon="lon",
        color="cluster_id",
        hover_name="zip",
        hover_data=["date", "n_case", "weight"],
        title=f"ST-DBSCAN Clusters: Geographic View{title_suffix}",
        map_style="carto-positron",
        zoom=10,
        opacity=0.6,
    )
    fig_map.update_traces(marker=dict(size=8))

    # px assigns one colour per cluster_id; reuse it so hulls match their points.
    colour_of = {t.name: t.marker.color for t in fig_map.data}

    no_polygon = []
    for cid, grp in df_clusters.groupby('cluster'):
        label = f"Cluster {cid}"
        pts = grp[['lon', 'lat']].drop_duplicates().values
        if len(pts) < 3:
            no_polygon.append(cid)
            continue
        try:
            ring = pts[ConvexHull(pts).vertices]
        except Exception:      # perfectly collinear points: no polygon
            no_polygon.append(cid)
            continue
        ring = np.vstack([ring, ring[:1]])          # close the ring
        colour = colour_of.get(label, '#888888')
        fig_map.add_trace(go.Scattermap(
            lon=ring[:, 0], lat=ring[:, 1],
            mode='lines', fill='toself',
            fillcolor=_to_rgba(colour, 0.25),
            line=dict(color=colour, width=2),
            name=label, legendgroup=label, showlegend=False,
            hoverinfo='skip',
        ))

    if no_polygon:
        print(f"Note: {len(no_polygon)} cluster(s) span fewer than three distinct "
              f"locations and have no footprint drawn: {sorted(no_polygon)}")

    fig_map.write_html(os.path.join(output_dir, f"{prefix}1_cluster_map.html"))
    print(f"- Saved {prefix}1_cluster_map.html")
    # ---------------------------------------------------------
    # 2. 3D Space-Time Plot (Spatiotemporal Focus)
    # ---------------------------------------------------------
    # X/Y are Latitude and Longitude. Z is Time (Days).
    # This allows you to spin the cube and visually see how clusters 
    # form as "cylinders" or "blobs" moving up through time.
    fig_3d = px.scatter_3d(
        df_clusters, 
        x="lon", 
        y="lat", 
        z="days", 
        color="cluster_id",
        hover_data=["date", "zip"],
        title=f"3D Space-Time Cluster Visualization{title_suffix}",
        opacity=0.7
    )
    # Adjust aspect ratio so the time axis (Z) is stretched out
    fig_3d.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=2)))
    fig_3d.write_html(os.path.join(output_dir, f"{prefix}2_cluster_3d.html"))
    print(f"- Saved {prefix}2_cluster_3d.html")

    # ---------------------------------------------------------
    # 3. Cluster Timeline (Temporal Focus)
    # ---------------------------------------------------------
    # Y-axis is the Cluster ID. X-axis is the actual Date.
    # Shows you the lifespan of a cluster (did it happen over 2 days or 3 weeks?)
    # and whether multiple clusters were active in NYC at the same time.
    fig_time = px.scatter(
        df_clusters,
        x="date",
        y="cluster_id",
        color="cluster_id",
        hover_data=["zip", "lat", "lon"],
        title=f"Cluster Lifespans / Timeline{title_suffix}",
    )
    # Update layout to make it look like a Gantt chart of events
    fig_time.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
    fig_time.write_html(os.path.join(output_dir, f"{prefix}3_cluster_timeline.html"))
    print(f"- Saved {prefix}3_cluster_timeline.html")
    
    print("Done! Open the HTML files in any web browser to explore.")


def _to_rgba(colour, alpha):
    """Accept '#rrggbb' or 'rgb(r,g,b)' and return an rgba() string, so a
    translucent fill doesn't also wash out the outline."""
    if colour.startswith('#'):
        r, g, b = (int(colour[i:i + 2], 16) for i in (1, 3, 5))
    else:
        r, g, b = (float(v) for v in colour.strip('rgb()').split(','))
    return f"rgba({r}, {g}, {b}, {alpha})"


def visualize_st_dbscan_hulls(df_clusters, oe_df=None, output_dir="outputs",
                              nickname="", top_n=None, zoom=9):
    """
    Exploratory map: one filled footprint per ST-DBSCAN cluster, plus its
    member cases as points.

    The footprint is the convex hull of the cluster's ZIP centroids, so it
    shows the shape the algorithm actually found instead of forcing it into a
    circle. If `oe_df` (from cluster_observed_expected) is supplied, hulls are
    shaded by O/E; otherwise each cluster gets its own categorical colour.

    Clusters spanning fewer than three distinct locations cannot form a
    polygon. They are drawn as points and reported, not dropped silently.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    prefix = f"{nickname}_" if nickname else ""

    df = df_clusters.copy()
    total_clusters = df['cluster'].nunique()

    if top_n is not None and top_n < total_clusters:
        keep = (df.groupby('cluster').size()
                  .sort_values(ascending=False).head(top_n).index)
        df = df[df['cluster'].isin(keep)]
        title_suffix = f" (top {top_n} of {total_clusters})"
    else:
        title_suffix = f" (all {total_clusters} clusters)"

    oe_lookup = oe_df.set_index('cluster').to_dict('index') if oe_df is not None else {}
    shade_by_oe = bool(oe_lookup)
    if shade_by_oe:
        vals = [v['oe_ratio'] for v in oe_lookup.values() if pd.notna(v['oe_ratio'])]
        oe_min, oe_max = (min(vals), max(vals)) if vals else (0.0, 1.0)

    palette = px.colors.qualitative.Bold
    fig = go.Figure()
    no_polygon = []

    for i, (cid, grp) in enumerate(df.groupby('cluster')):
        pts = grp[['lon', 'lat']].drop_duplicates().values
        meta = oe_lookup.get(cid, {})

        oe = meta.get('oe_ratio', np.nan)
        if shade_by_oe and pd.notna(oe):
            frac = 0.0 if oe_max == oe_min else (oe - oe_min) / (oe_max - oe_min)
            colour = px.colors.sample_colorscale('Reds', [0.25 + 0.7 * frac])[0]
        else:
            colour = palette[i % len(palette)]

        bits = [f"<b>Cluster {cid}</b>",
                f"cases: {meta.get('observed', len(grp))}",
                f"ZIPs: {meta.get('n_zips', grp['zip'].nunique())}"]
        if pd.notna(oe):
            bits.append(f"O/E: {oe:.2f}")
        if 'start' in meta:
            bits.append(f"{meta['start']:%Y-%m-%d} to {meta['end']:%Y-%m-%d}"
                        f" ({meta['duration_days']}d)")
        if 'radius_km' in meta:
            bits.append(f"radius: {meta['radius_km']:.1f} km")
        label = "<br>".join(bits)

        has_hull = False
        if len(pts) >= 3:
            try:
                ring = pts[ConvexHull(pts).vertices]
                ring = np.vstack([ring, ring[:1]])   # close the polygon
                fig.add_trace(go.Scattermap(
                    lon=ring[:, 0], lat=ring[:, 1],
                    mode='lines', fill='toself',
                    fillcolor=_to_rgba(colour, 0.35),
                    line=dict(color=colour, width=2),
                    name=f"Cluster {cid}", legendgroup=str(cid),
                    hovertemplate=label + "<extra></extra>",
                ))
                has_hull = True
            except Exception:
                pass                                  # collinear: no polygon
        if not has_hull:
            no_polygon.append(cid)

        fig.add_trace(go.Scattermap(
            lon=grp['lon'], lat=grp['lat'], mode='markers',
            marker=dict(size=7, color=colour),
            name=f"Cluster {cid}", legendgroup=str(cid),
            showlegend=not has_hull,
            hovertemplate=label + "<extra></extra>",
        ))

    if no_polygon:
        print(f"Note: {len(no_polygon)} cluster(s) span fewer than three distinct "
              f"locations (or are perfectly collinear) and are drawn as points "
              f"only: {sorted(no_polygon)}")

    fig.update_layout(
        map=dict(style="carto-positron",
                center=dict(lat=df['lat'].mean(), lon=df['lon'].mean()),
                zoom=zoom),
        title=(f"ST-DBSCAN cluster footprints{title_suffix}"
               + (" — shaded by O/E" if shade_by_oe else "")),
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(title="Cluster"),
    )
    path = os.path.join(output_dir, f"{prefix}4_cluster_hulls.html")
    fig.write_html(path)
    print(f"- Saved {os.path.basename(path)}")
    return fig