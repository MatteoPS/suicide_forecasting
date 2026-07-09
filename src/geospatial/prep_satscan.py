import pandas as pd

from src.etl.ingest import load_nvdrs, fetch_census
from src.etl.transform import filter_nvdrs_suicides, harmonize_zcta_boundaries
from src.utils.config import get_data_path, PROJECT_ROOT

def prep_satscan_gui(
    nvdrs_cols: list = None, 
    pop_vars: dict = None, 
    pop_years: range = range(2011, 2024)
):
    """Generates the 3 required CSV files for the SaTScan GUI."""
    print("Preparing SaTScan files...")
    
    # --- 0. Setup and Validation ---
    out_dir = PROJECT_ROOT / "data" / "processed" / "satscan"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the absolute minimum columns needed for filtering and location mapping
    required_nvdrs = {'IncidentID', 'DeathDate', 'InjuryZip', 'ResidenceZip', 'IncidentCategory_c', 'PersonType'}
    
    # Use provided columns, or default to the minimum set plus common demographic variables
    if nvdrs_cols is None:
        nvdrs_cols = list(required_nvdrs) + ['Sex', 'AgeYears_c']
    else:
        missing = required_nvdrs - set(nvdrs_cols)
        if missing:
            raise ValueError(f"nvdrs_cols is missing required columns: {missing}")

    if pop_vars is None:
        pop_vars = {"NAME": "ZTCA5", "DP05_0001E": "Population"}

    # --- 1. CASE FILE (Location, Cases, Date) ---
    print("Processing Case Data...")
    
    # Load only the validated columns from NVDRS
    nvdrs_df = load_nvdrs(file_key="nvdrs", data_folder="raw", usecols=nvdrs_cols)
    
    # Isolate suicides and filter to our target time window
    nvdrs_s_df = filter_nvdrs_suicides(nvdrs_df)
    nvdrs_s_df = nvdrs_s_df[nvdrs_s_df['DeathDate'] >= '2010-01-01'].copy()

    # SaTScan requires a single location identifier. We prioritize InjuryZip.
    # If InjuryZip is missing or invalid ('00000', '99999'), we fall back to ResidenceZip.
    nvdrs_s_df['DerivedZip'] = nvdrs_s_df['InjuryZip'].replace(['', '00000', '99999'], float('NaN')).fillna(nvdrs_s_df['ResidenceZip'])
    nvdrs_s_df = nvdrs_s_df.dropna(subset=['DerivedZip'])
    nvdrs_s_df = nvdrs_s_df.rename(columns={'DerivedZip': 'ZIP'})
        # Save the rich, filtered metadata as a separate artifact for EDA
    nvdrs_s_df.to_csv(out_dir / "satscan_cases_enriched.csv", index=False)

    # Group into SaTScan's required long format: [Location_ID, Number_of_Cases, Date]
    cases_df = nvdrs_s_df.groupby(['ZIP', 'DeathDate']).size().reset_index(name='Cases')
    cases_df = cases_df[['ZIP', 'Cases', 'DeathDate']]
    cases_df.to_csv(out_dir / "satscan_cases.csv", index=False)

    


    # --- 2. POPULATION FILE (Location, Year, Population) ---
    print("Processing Population Data...")
    
    # Fetch standard ACS estimates for the provided timeframe
    pop_zip = fetch_census(pop_vars, pop_years, geo_level="zip")
    pop_zip.rename(columns={"zip code tabulation area": "ZIP"}, inplace=True)
    pop_zip = pop_zip.drop(columns=['state'], errors='ignore')
    pop_zip['Population'] = pd.to_numeric(pop_zip['Population'], errors='coerce')

    # Backfill 2010 using 2011 ACS data to avoid Decennial-to-ACS denominator shocks
    # (The 2011 ACS 5-year estimate covers 2007-2011, making it mathematically valid as a proxy for 2010)
    df_2010_proxy = pop_zip[pop_zip['Year'] == 2011].copy()
    df_2010_proxy['Year'] = 2010
    pop_zip = pd.concat([df_2010_proxy, pop_zip], ignore_index=True)

    # Map historical 2010-2019 ZCTAs to modern 2020 boundaries
    # This prevents boundary redraws from looking like sudden demographic shifts to the model
    pop_zip_c = harmonize_zcta_boundaries(pop_zip, zip_col='ZIP', pop_col='Population', year_col='Year')
    
    pop_final = pop_zip_c[['ZIP', 'Year', 'Population']].dropna()
    pop_final.to_csv(out_dir / "satscan_population.csv", index=False)

    # --- 3. COORDINATES FILE (Location, Lat, Lon) ---
    print("Processing Coordinates Data...")
    
    # Load Census Gazetteer geographic centers
    zip_coord_path = get_data_path("zip_coordinates")
    zip_coord = pd.read_table(zip_coord_path, sep='\t', dtype=str)
    
    zip_coord.columns = [col.strip() for col in zip_coord.columns]
    
    coord_final = zip_coord[['GEOID', 'INTPTLAT', 'INTPTLONG']].copy()
    coord_final.rename(columns={'GEOID': 'ZIP', 'INTPTLAT': 'Latitude', 'INTPTLONG': 'Longitude'}, inplace=True)
    coord_final.to_csv(out_dir / "satscan_coordinates.csv", index=False)

    print(f"Success! Files ready for SaTScan GUI in: {out_dir}")

if __name__ == "__main__":
    prep_satscan_gui()