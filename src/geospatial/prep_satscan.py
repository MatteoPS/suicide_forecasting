"""
prep_satscan.py

Generates the global artifact files (Cases, Population, Coordinates) required by 
the SaTScan GUI for spatiotemporal cluster analysis. Includes automated healing 
for missing Census coordinates and strict missingness tracking for non-spatial ZIP codes.
"""

import pandas as pd
from uszipcode import SearchEngine

# Local Project Imports
from src.etl.ingest import load_nvdrs, fetch_census
from src.etl.transform import filter_nvdrs_suicides, harmonize_zcta_boundaries
from src.utils.config import get_data_path, PROJECT_ROOT

def prep_satscan_gui(
    nvdrs_cols: list = None, 
    pop_vars: dict = None, 
    pop_years: range = range(2011, 2024)
):
    """
    Ingests raw health and demographic data, applies strict geospatial validations, 
    and outputs mathematically sound files for SaTScan Poisson modeling.
    """
    print("Preparing global SaTScan files...")
    
    # --- 0. Setup and Validation ---
    out_dir = PROJECT_ROOT / "data" / "processed" / "satscan"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Define absolute minimum columns needed for filtering and location mapping
    required_nvdrs = {'IncidentID', 'DeathDate', 'InjuryZip', 'ResidenceZip', 'IncidentCategory_c', 'PersonType'}
    
    if nvdrs_cols is None:
        nvdrs_cols = list(required_nvdrs) + ['Sex', 'AgeYears_c']
    else:
        missing = required_nvdrs - set(nvdrs_cols)
        if missing:
            raise ValueError(f"nvdrs_cols is missing required columns: {missing}")

    if pop_vars is None:
        pop_vars = {"NAME": "ZTCA5", "DP05_0001E": "Population"}

    def clean_zip(series):
        """
        Aggressively standardizes ZIP codes into clean 5-digit strings.
        Prevents Pandas from dropping valid locations due to invisible whitespace or floats.
        """
        return series.astype(str).str.replace(r'\.0$', '', regex=True).str.strip().str.zfill(5)


    # --- 1. CASE FILE (Numerator) ---
    print("Processing Case Data...")
    nvdrs_df = load_nvdrs(file_key="nvdrs", data_folder="raw", usecols=nvdrs_cols)
    nvdrs_s_df = filter_nvdrs_suicides(nvdrs_df)
    nvdrs_s_df = nvdrs_s_df[nvdrs_s_df['DeathDate'] >= '2010-01-01'].copy()

    # Location Logic: Prefer Injury location, fallback to Residence if missing
    nvdrs_s_df['DerivedZip'] = nvdrs_s_df['InjuryZip'].replace(['', '00000', '99999'], float('NaN')).fillna(nvdrs_s_df['ResidenceZip'])
    nvdrs_s_df = nvdrs_s_df.dropna(subset=['DerivedZip'])
    nvdrs_s_df['DerivedZip'] = clean_zip(nvdrs_s_df['DerivedZip'])
    nvdrs_s_df = nvdrs_s_df[~nvdrs_s_df['DerivedZip'].isin(['nan', '00000', '99999'])]

    # Aggregate to SaTScan requirement: [Location ID, Cases, Date]
    cases_df = nvdrs_s_df.groupby(['DerivedZip', 'DeathDate']).size().reset_index(name='Cases')
    cases_df = cases_df[['DerivedZip', 'Cases', 'DeathDate']].rename(columns={'DerivedZip': 'ZIP'})


    # --- 2. POPULATION FILE (Denominator) ---
    print("Processing Population Data...")
    pop_zip = fetch_census(pop_vars, pop_years, geo_level="zip")
    pop_zip.rename(columns={"zip code tabulation area": "ZIP"}, inplace=True)
    pop_zip = pop_zip.drop(columns=['state'], errors='ignore')
    pop_zip['Population'] = pd.to_numeric(pop_zip['Population'], errors='coerce')

    # Methodological Fix: Backfill 2010 using 2011 ACS proxy
    # Prevents artificial denominator shocks caused by switching between Decennial and ACS data
    df_2010_proxy = pop_zip[pop_zip['Year'] == 2011].copy()
    df_2010_proxy['Year'] = 2010
    pop_zip = pd.concat([df_2010_proxy, pop_zip], ignore_index=True)
    
    # Methodological Fix: Harmonize boundary redraws
    # Maps historical ZCTAs to modern boundaries to prevent artifactual population shifts
    pop_zip_c = harmonize_zcta_boundaries(pop_zip, zip_col='ZIP', pop_col='Population', year_col='Year')
    pop_final = pop_zip_c[['ZIP', 'Year', 'Population']].dropna().copy()
    pop_final['ZIP'] = clean_zip(pop_final['ZIP'])


    # --- 3. COORDINATES FILE (Spatial Matrix) ---
    print("Processing Coordinates Data...")
    zip_coord_path = get_data_path("zip_coordinates")
    zip_coord = pd.read_table(zip_coord_path, sep='\t', dtype=str)
    zip_coord.columns = [col.strip() for col in zip_coord.columns]
    
    coord_final = zip_coord[['GEOID', 'INTPTLAT', 'INTPTLONG']].copy()
    coord_final.rename(columns={'GEOID': 'ZIP', 'INTPTLAT': 'Latitude', 'INTPTLONG': 'Longitude'}, inplace=True)
    coord_final['ZIP'] = clean_zip(coord_final['ZIP'])


    # --- 4. GLOBAL HEALER ---
    # The raw Census Gazetteer frequently drops highly populated residential ZCTAs. 
    # This block identifies missing coordinates and patches them in using the uszipcode library.
    print("Validating Coordinate Coverage...")
    needed_zips = set(cases_df['ZIP']).union(set(pop_final['ZIP']))
    known_coords = set(coord_final['ZIP'])
    missing_coords = needed_zips - known_coords

    if missing_coords:
        print(f"  ↳ Triggering Healer: {len(missing_coords)} global ZIPs missing from Census Gazetteer.")
        search = SearchEngine()
        recovered = []
        for z in missing_coords:
            res = search.by_zipcode(z)
            if res and res.lat and res.lng:
                recovered.append({'ZIP': z, 'Latitude': res.lat, 'Longitude': res.lng})
        
        if recovered:
            coord_final = pd.concat([coord_final, pd.DataFrame(recovered)], ignore_index=True)
            print(f"  ↳ Successfully recovered {len(recovered)} coordinates.")


    # --- 5. THE ENFORCER & MISSINGNESS ANALYSIS ---
    # SaTScan is a spatial model. Any case occurring in a non-spatial ZIP (PO Box, Military Base, 
    # Corporate Skyscraper) will crash the software. We must document them, then drop them.
    print("Enforcing strict SaTScan referential integrity...")
    valid_coords = set(coord_final['ZIP'])
    unmappable_case_zips = set(cases_df['ZIP']) - valid_coords
    
    if unmappable_case_zips:
        print(f"  ↳ Analyzing {len(unmappable_case_zips)} unmappable ZIPs with associated cases...")
        
        # Build the Missingness Artifact
        dropped_cases_agg = cases_df[cases_df['ZIP'].isin(unmappable_case_zips)].groupby('ZIP')['Cases'].sum().reset_index()
        dropped_pop_agg = pop_final[pop_final['ZIP'].isin(unmappable_case_zips)].groupby('ZIP')['Population'].mean().reset_index()
        
        search = SearchEngine()
        metadata = []
        for z in unmappable_case_zips:
            res = search.by_zipcode(z)
            if res:
                metadata.append({
                    'ZIP': z,
                    'Type': res.zipcode_type, # Identifies 'PO Box', 'Unique', etc.
                    'City': res.major_city,
                    'County': res.county,
                    'State': res.state,
                    'USPS_Population': res.population 
                })
            else:
                metadata.append({'ZIP': z, 'Type': 'Unknown', 'City': 'Unknown', 'County': 'Unknown', 'State': 'Unknown'})
                
        meta_df = pd.DataFrame(metadata)
        
        # Merge and save diagnostic dataframe
        dropped_df = dropped_cases_agg.merge(dropped_pop_agg, on='ZIP', how='left').merge(meta_df, on='ZIP', how='left')
        dropped_df.rename(columns={'Population': 'Census_Pop_Avg'}, inplace=True)
        dropped_df.fillna({'Census_Pop_Avg': 0, 'USPS_Population': 0}, inplace=True)
        dropped_df = dropped_df.sort_values(by='Cases', ascending=False)
        
        dropped_df.to_csv(out_dir / "dropped_unmappable_cases.csv", index=False)
        print(f"  ↳ Saved missingness analysis to 'dropped_unmappable_cases.csv'.")
    
    # Safely enforce the drop to guarantee SaTScan stability
    initial_cases = len(cases_df)
    cases_df = cases_df[cases_df['ZIP'].isin(valid_coords)]
    dropped_cases_total = initial_cases - len(cases_df)
    
    pop_final = pop_final[pop_final['ZIP'].isin(valid_coords)]
    nvdrs_s_df = nvdrs_s_df[nvdrs_s_df['DerivedZip'].isin(valid_coords)]
    
    if dropped_cases_total > 0:
        print(f"  ↳ Dropped {dropped_cases_total} total cases occurring in unmappable ZIPs.")


    # --- 6. EXPORT ---
    print("Exporting mathematically validated artifacts...")
    cases_df.to_csv(out_dir / "satscan_cases.csv", index=False)
    pop_final.to_csv(out_dir / "satscan_population.csv", index=False)
    coord_final.to_csv(out_dir / "satscan_coordinates.csv", index=False)
    
    # Save the analytical baseline so it can be filtered regionally later
    nvdrs_s_df.to_csv(out_dir / "nvdrs_analytical.csv", index=False)

    print(f"Success! Global files ready in: {out_dir}")


if __name__ == "__main__":
    prep_satscan_gui()