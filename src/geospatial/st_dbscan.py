import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances
from sklearn.cluster import DBSCAN

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
        Path to the population file (format: zip, population). If provided, 
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
    """
    # 1. Load and prep spatial-temporal data
    cases = pd.read_csv(cas_file, header=None, names=['zip', 'n_case', 'date'], sep=r'\s+')
    geo = pd.read_csv(geo_file, header=None, names=['zip', 'lat', 'lon'], sep=r'\s+')
    
    # --- CRITICAL FIX: Standardize ZIP codes as strings to ensure merges work ---
    cases['zip'] = cases['zip'].astype(str).str.strip().str.zfill(5)
    geo['zip'] = geo['zip'].astype(str).str.strip().str.zfill(5)
    
    cases['date'] = pd.to_datetime(cases['date'])
    baseline_date = cases['date'].min()
    cases['days'] = (cases['date'] - baseline_date).dt.days
    
    # 2. Calculate sample weights for DBSCAN density evaluation
    if pop_file:
        pop = pd.read_csv(pop_file, header=None, names=['zip', 'population'], sep=r'\s+')
        
        # --- CRITICAL FIX: Standardize pop ZIP codes ---
        pop['zip'] = pop['zip'].astype(str).str.strip().str.zfill(5)
        
        cases = cases.merge(pop, on='zip', how='left')
    else:
        cases['weight'] = cases['n_case']

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
    # DEBUG PRINT
    print(f"Max weight in data: {df['weight'].max()}")
    print(f"Min weight in data (above 0): {df[df['weight'] > 0]['weight'].min()}")
    print(f"Your safe_min_samples is: {safe_min_samples}")
    print(f"Your max safe_weights is: {safe_weights.max()}")

    df['cluster'] = db.labels_
    
    # Filter out noise points (label -1)
    clusters = df[df['cluster'] != -1].copy()
    
    print(f"Clustering complete. Discovered {clusters['cluster'].nunique()} unique clusters.")
    return clusters