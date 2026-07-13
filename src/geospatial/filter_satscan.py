import pandas as pd
from src.utils.config import get_data_path, PROJECT_ROOT

def create_regional_satscan_files(
    nickname_regional: str,
    nickname: str, 
    state_names: list = None, 
    county_names: list = None
):
    """Filters global SaTScan artifacts for a specific region."""
    base_dir = PROJECT_ROOT / "data" / "processed" / "satscan"
    out_dir = base_dir / nickname_regional
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting SaTScan files for: {nickname_regional}...")

    def clean_zip(series):
        return series.astype(str).str.replace(r'\.0$', '', regex=True).str.strip().str.zfill(5)

    # 1. Map Region ZIPs
    map_path = get_data_path("zip_to_state_county")
    zip_mapping = pd.read_csv(map_path, dtype={'ZIP': str})
    zip_mapping['ZIP'] = clean_zip(zip_mapping['ZIP'])
    zip_mapping['state'] = zip_mapping['state'].str.strip()
    zip_mapping['county'] = zip_mapping['county'].str.strip()

    if county_names:
        mask = zip_mapping['county'].isin(county_names)
        if state_names:
            mask = mask & zip_mapping['state'].isin(state_names)
        regional_zips = zip_mapping[mask]['ZIP'].tolist()
    elif state_names:
        regional_zips = zip_mapping[zip_mapping['state'].isin(state_names)]['ZIP'].tolist()
    else:
        raise ValueError("Provide either 'county_names' + 'state_names' or 'state_names'.")

    # 2. Load Global Artifacts
    coord_df = pd.read_csv(base_dir / f"{nickname}_full_geo.csv", dtype={'ZIP': str})
    coord_df['ZIP'] = clean_zip(coord_df['ZIP'])
    zips_with_coords = set(coord_df['ZIP'])

    pop_df = pd.read_csv(base_dir / f"{nickname}_full_pop.csv", dtype={'ZIP': str})
    pop_df['ZIP'] = clean_zip(pop_df['ZIP'])
    
    cases_df = pd.read_csv(base_dir / f"{nickname}_full_cas.csv", dtype={'ZIP': str})
    cases_df['ZIP'] = clean_zip(cases_df['ZIP'])
    
    # Uses analytical file as the base for enriched
    enriched_df = pd.read_csv(base_dir /f"{nickname}_nvdrs_analytical.csv", dtype={'DerivedZip': str})
    enriched_df.rename(columns={'DerivedZip': 'ZIP'}, inplace=True)
    enriched_df['ZIP'] = clean_zip(enriched_df['ZIP'])

    # 3. Identify Valid ZIPs (Must exist in Coords AND Pop > 0)
    reg_pop_unfiltered = pop_df[pop_df['ZIP'].isin(regional_zips)]
    pop_sums = reg_pop_unfiltered.groupby('ZIP')['Population'].sum()
    zips_with_pop = set(pop_sums[pop_sums > 0].index)

    usable_zips = set(regional_zips).intersection(zips_with_coords).intersection(zips_with_pop)
    dropped_zips = set(regional_zips) - usable_zips

    print(f"  ↳ Found {len(regional_zips)} regional USPS ZIPs.")
    print(f"  ↳ Dropped {len(dropped_zips)} ZIPs.")
    print(f"  ↳ Proceeding with {len(usable_zips)} valid ZCTAs.")

    # 4. Filter and Save Output Files
    reg_coord = coord_df[coord_df['ZIP'].isin(usable_zips)]
    reg_coord.to_csv(out_dir / f"{nickname_regional}_geo.csv", index=False)

    reg_pop = pop_df[pop_df['ZIP'].isin(usable_zips)]
    reg_pop.to_csv(out_dir / f"{nickname_regional}_pop.csv", index=False)

    reg_cases = cases_df[cases_df['ZIP'].isin(usable_zips)]
    reg_cases.to_csv(out_dir / f"{nickname_regional}_cas.csv", index=False)

    reg_enriched = enriched_df[enriched_df['ZIP'].isin(usable_zips)]
    reg_enriched.to_csv(out_dir / f"{nickname_regional}_enriched_cas.csv", index=False)

    print(f"Success! Regional files saved to: {out_dir}\n")

    # 5. Generate Summary
    case_sums = reg_cases.groupby('ZIP')['Cases'].sum().reset_index()
    summary_df = case_sums.merge(pop_sums.reset_index(), on='ZIP', how='right').fillna({'Cases': 0})
    summary_df.rename(columns={'Population': 'Total_Pop_14_Years', 'Cases': 'Total_Cases'}, inplace=True)
    summary_df = summary_df[summary_df['ZIP'].isin(usable_zips)].sort_values(by='Total_Cases', ascending=False).reset_index(drop=True)

    return summary_df, list(dropped_zips)