import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import os


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

    # --- NEW LOGIC: Filter for Top N clusters ---
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
    # Plots the points on a street map of NYC. 
    # Great for seeing *where* clusters happened.
    fig_map = px.scatter_mapbox(
        df_clusters, 
        lat="lat", 
        lon="lon", 
        color="cluster_id",
        hover_name="zip",
        hover_data=["date", "n_case"],
        title=f"ST-DBSCAN Clusters: Geographic View{title_suffix}",
        mapbox_style="carto-positron",
        zoom=10
    )
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