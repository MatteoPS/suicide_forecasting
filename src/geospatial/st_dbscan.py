import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances
from sklearn.cluster import DBSCAN

from src.geospatial.satscan_io import (
    read_satscan_cases,
    read_satscan_geo,
    read_satscan_pop,
)

def run_small_st_dbscan(cas_file, geo_file, pop_file=None, eps1_km=1.5, eps2_days=14, min_threshold=5):
    """
    Executes Space-Time DBSCAN (ST-DBSCAN) using a dense distance matrix.
    Designed for datasets < 10,000 unique records.
    
    Parameters:
    -----------
    cas_file : str
        Path to the case file (format: zip, n_case, date).
    geo_file : str
        Path to the geography file (format: zip, lat, lon).
    pop_file : str, optional
        Path to the population file (format: zip, year, population). If provided, 
        clustering utilizes incidence rates (cases per 100K). Otherwise, raw counts are used.
    eps1_km : float
        Maximum spatial distance (kilometers) for neighborhood evaluation.
    eps2_days : int
        Maximum temporal distance (days) for neighborhood evaluation.
    min_threshold : float
        Minimum cumulative sample weight (counts or rates) required to form a core point.
    
    Returns:
    --------
    pd.DataFrame
        Filtered dataframe containing only clustered records (noise removed).
        Includes a `weight_type` column ('rate' or 'count') indicating what
        `weight` represents, for downstream functions like the map viz.
    """
    
    # 1. Load and prep spatial-temporal data
    # The readers standardize ZIP codes to zero-padded strings, without which the merges below silently return nothing.
    cases = read_satscan_cases(cas_file)
    geo = read_satscan_geo(geo_file)

    baseline_date = cases['date'].min()
    cases['days'] = (cases['date'] - baseline_date).dt.days
    
    # Extract year for population matching
    cases['year'] = cases['date'].dt.year
    
    # 2. Calculate sample weights for DBSCAN density evaluation
    if pop_file:
        pop = read_satscan_pop(pop_file)

        # Merge on BOTH zip and year to get the exact population for that specific case's year
        cases = cases.merge(pop, on=['zip', 'year'], how='left')
        
        # Calculate incidence rate per 100K, handling zero-population ZIPs and NaN failures
        cases['weight'] = np.where(
            (cases['population'] > 0) & (cases['population'].notna()), 
            (cases['n_case'] / cases['population']) * 100000, 
            0.0
        )
        print("Population data provided: Weighting by Incidence Rate (per 100K, dynamically matched by Year).")
    else:
        cases['weight'] = cases['n_case']
        print("No population data provided: Weighting by raw case counts.")

    cases['weight_type'] = 'rate' if pop_file else 'count'
    # Merge geographic coordinates
    df = cases.merge(geo, on='zip', how='inner')
    
    # Convert latitude and longitude to radians for Haversine distance calculations
    df['lat_rad'] = np.radians(df['lat'])
    df['lon_rad'] = np.radians(df['lon'])
    
    N = len(df)
    print(f"Constructing matrices and executing dense ST-DBSCAN on {N} records...")
    
    # 3. Construct Temporal Matrix (Days)
    days = df['days'].values
    temporal_matrix = np.abs(days[:, None] - days[None, :])
    
    # 4. Construct Spatial Matrix (Kilometers)
    EARTH_RADIUS_KM = 6371.0
    coords_rad = df[['lat_rad', 'lon_rad']].values
    spatial_matrix = haversine_distances(coords_rad, coords_rad) * EARTH_RADIUS_KM
    
    # 5. Combine Spatial and Temporal Matrices
    # Apply temporal threshold: invalidate spatial distances where temporal distance exceeds eps2_days
    combined_matrix = np.where(temporal_matrix <= eps2_days, spatial_matrix, 99999.0)
    
    # --- The Scikit-Learn Float Validation Bypass ---
    # Scikit-learn's DBSCAN throws an error if min_samples < 1 or is a float.
    # To use continuous rates (e.g., 0.1 per 100K) as weights, we multiply both 
    # the min_threshold and the sample weights by a large factor to cast them as valid integers,
    # perfectly preserving their mathematical proportions.
    MULTIPLIER = 1000000
    safe_min_samples = int(min_threshold * MULTIPLIER)
    safe_weights = (df['weight'] * MULTIPLIER).astype(int).values

    # 6. Execute Clustering
    db = DBSCAN(eps=eps1_km, min_samples=safe_min_samples, metric='precomputed')
    db.fit(combined_matrix, sample_weight=safe_weights)
    df['cluster'] = db.labels_
    
    # Filter out noise points (label -1)
    clusters = df[df['cluster'] != -1].copy()
    
    print(f"Clustering complete. Discovered {clusters['cluster'].nunique()} unique clusters.")
    return clusters

def cluster_observed_expected(df_clusters, cas_file, pop_file):
    """
    Observed vs expected cases for each ST-DBSCAN cluster.

    Expected is what the cluster's ZIPs would produce over the cluster's own
    date window at the study-wide baseline rate, so O/E lands on roughly the
    same scale as SaTScan's relative risk and is scoped to the window the
    cluster was actually active, not the whole study period.

    Two things this is NOT. There is no p-value. And because clusters are
    selected by density in the first place, an O/E computed afterwards is
    biased upward with no null to correct against. Descriptive, not inferential.

    Parameters
    ----------
    df_clusters : pd.DataFrame
        Output of run_small_st_dbscan (clustered records only).
    cas_file, pop_file : str
        The same files passed to run_small_st_dbscan. The case file is needed
        because df_clusters has noise removed and so cannot supply the
        study-wide denominator.

    Returns
    -------
    pd.DataFrame
        One row per cluster, sorted by O/E descending.
    """
    cases = read_satscan_cases(cas_file)
    pop = read_satscan_pop(pop_file)

    study_start, study_end = cases['date'].min(), cases['date'].max()

    # Person-days year by year, so a ZIP that grows or shrinks across the
    # study is not collapsed to a single flat population.
    dates = pd.date_range(study_start, study_end, freq='D')
    days_per_year = pd.Series(dates.year).value_counts().sort_index()
    days_per_year.index.name = 'year'
    days_per_year.name = 'days'

    pop_days = pop.join(days_per_year, on='year')
    pop_days['person_days'] = pop_days['population'] * pop_days['days']
    total_person_days = pop_days['person_days'].sum()
    if total_person_days <= 0:
        raise ValueError("No person-time at risk; check the population file.")

    baseline_rate = cases['n_case'].sum() / total_person_days  # per person-day
    pop_lookup = pop.set_index(['zip', 'year'])['population']

    EARTH_RADIUS_KM = 6371.0
    rows, missing = [], set()

    for cid, grp in df_clusters.groupby('cluster'):
        start, end = grp['date'].min(), grp['date'].max()
        duration_days = (end - start).days + 1
        member_zips = list(grp['zip'].unique())

        # Population in the cluster's start year. Clusters are short relative
        # to a year, so one year is the right denominator.
        keys = [(z, start.year) for z in member_zips]
        pops = pop_lookup.reindex(keys)
        missing |= {z for (z, _), p in zip(keys, pops) if pd.isna(p)}
        pop_at_risk = pops.sum()

        expected = baseline_rate * pop_at_risk * duration_days
        observed = grp['n_case'].sum()

        lat_c, lon_c = grp['lat'].mean(), grp['lon'].mean()
        d = haversine_distances(
            np.radians(grp[['lat', 'lon']].values),
            np.radians([[lat_c, lon_c]]),
        ) * EARTH_RADIUS_KM

        rows.append({
            'cluster': cid, 'observed': observed, 'expected': expected,
            'oe_ratio': observed / expected if expected > 0 else np.nan,
            'n_zips': len(member_zips), 'n_records': len(grp),
            'start': start, 'end': end, 'duration_days': duration_days,
            'pop_at_risk': pop_at_risk,
            'centroid_lat': lat_c, 'centroid_lon': lon_c,
            'radius_km': float(d.max()),
        })

    if missing:
        print(f"Warning: {len(missing)} ZIP(s) had no population row for their "
                f"cluster's start year and were left out of the denominator, "
                f"which inflates O/E: {sorted(missing)}")

    return (pd.DataFrame(rows)
            .sort_values('oe_ratio', ascending=False)
            .reset_index(drop=True))