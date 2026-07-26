from typing import Literal

import pandas as pd
import numpy as np
from uszipcode import SearchEngine
from src.utils.config import get_data_path

def filter_nvdrs_suicides(df: pd.DataFrame) -> pd.DataFrame:
    """Filters dataset for suicides and isolates the primary actor in multi-person incidents."""
    # Keep only rows involving suicide
    suicide_df = df[df['IncidentCategory_c'].str.contains('suicide', case=False, na=False)].copy()
    
    # Split into single vs. multi-person incidents
    mask = suicide_df['IncidentCategory_c'] == 'Single suicide'
    single = suicide_df[mask]
    others = suicide_df[~mask]
    
    # For multi-person incidents, keep only the suspect who is also a victim
    idx = (others['PersonType'] == 'Both victim and suspect').groupby(others['IncidentID']).idxmax()
    
    # Recombine single suicides with the filtered multi-person incidents
    cleaned_df = pd.concat([single, others.loc[idx]], ignore_index=True)
    
    # Standardize date format for time series usage
    cleaned_df['DeathDate'] = pd.to_datetime(cleaned_df['DeathDate'])
    return cleaned_df

def aggregate_nvdrs_daily_injury(df: pd.DataFrame, geo_level: Literal["county", "state"] = None) -> pd.DataFrame:
    """uses 'InjuryDate' column to Aggregates incidents by day (one day per row), optionally separating counts by county. Default = Country (no separation)"""
    if geo_level == "state":
        # Count daily incidents per state
        daily_df = df.groupby(['InjuryDate', 'InjuryState']).size().reset_index(name='incident_count')
        # Reshape data so each state has its own column, filling missing days with 0
        return daily_df.pivot(index='InjuryDate', columns='InjuryState', values='incident_count').reset_index().fillna(0)
    elif geo_level == "county":
        # Count daily incidents per county
        daily_df = df.groupby(['InjuryDate', 'InjuryFIPS']).size().reset_index(name='incident_count')
        # Reshape data so each county has its own column, filling missing days with 0
        return daily_df.pivot(index='InjuryDate', columns='InjuryFIPS', values='incident_count').reset_index().fillna(0)
    # Count total daily incidents globally
    return df.groupby('InjuryDate').size().reset_index(name='incident_count')

def aggregate_nvdrs_daily(df: pd.DataFrame, geo_level: Literal["county", "state", "zip"] = None, geo_col: str= 'DeathFIPS') -> pd.DataFrame:
    """Uses 'DeathDate' column to aggregate incidents by day. Default = Country (no separation)"""
    if geo_level:
        daily_df = df.groupby(['DeathDate', geo_col]).size().reset_index(name='incident_count')
        return daily_df.pivot(index='DeathDate', columns=geo_col, values='incident_count').reset_index().fillna(0)
        
    return df.groupby('DeathDate').size().reset_index(name='incident_count')

def aggregate_nvdrs_monthly(df: pd.DataFrame, geo_level: Literal["county", "state"] = None) -> pd.DataFrame:
    """Uses 'DeathDate_myr' column (mm/yyyy) to aggregate incidents by month, optionally separating counts by state or county. Default = Country."""
    if geo_level == "state":
        monthly_df = df.groupby(['DeathDate_myr', 'DeathState']).size().reset_index(name='incident_count')
        return monthly_df.pivot(index='DeathDate_myr', columns='DeathState', values='incident_count').reset_index().fillna(0)
    
    elif geo_level == "county":
        monthly_df = df.groupby(['DeathDate_myr', 'DeathFIPS']).size().reset_index(name='incident_count')
        return monthly_df.pivot(index='DeathDate_myr', columns='DeathFIPS', values='incident_count').reset_index().fillna(0)
    
    return df.groupby('DeathDate_myr').size().reset_index(name='incident_count')

def enrich_zip_data(df: pd.DataFrame, zip_col: str = 'ZIP', zip_col_c: str = 'ZIP_c') -> pd.DataFrame:
    search = SearchEngine()
    
    # Clean ZIPs: convert to string, remove '.0', pad to 5 digits
    df[zip_col_c] = df[zip_col].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(5)
    
    # Process only unique ZIPs for performance
    unique_zips = df[zip_col_c].unique() # Updated to use the clean column
    z_map = {z: search.by_zipcode(z) for z in unique_zips}
    
    df['City'] = df[zip_col_c].map(lambda z: z_map[z].major_city if z_map.get(z) else None)
    df['County'] = df[zip_col_c].map(lambda z: z_map[z].county if z_map.get(z) else None)
    df['State'] = df[zip_col_c].map(lambda z: z_map[z].state if z_map.get(z) else None)
    
    return df


def enrich_fips_data(df: pd.DataFrame, fips_col: str = 'HFIPSSTCO') -> pd.DataFrame:
    # 1. Guard clause to prevent KeyError
    if fips_col not in df.columns:
        df['Hospital_State'] = np.nan
        df['Hospital_County'] = np.nan
        return df
        
    # 2. Clean FIPS safely
    df[fips_col] = df[fips_col].apply(
        lambda x: str(x).replace('.0', '').zfill(5) if pd.notna(x) and str(x).lower() != 'nan' else np.nan
    )
    
    # 3. Process crosswalk
    crosswalk_path = get_data_path("fips_crosswalk", "raw")
    col_names = ['State_Abbr', 'State_FIPS', 'County_FIPS', 'County_Name', 'Class_Code']
    fips_df = pd.read_csv(crosswalk_path, names=col_names, dtype=str)
    
    fips_df['FIPS'] = fips_df['State_FIPS'] + fips_df['County_FIPS']
    
    df = df.merge(
        fips_df[['FIPS', 'State_Abbr', 'County_Name']], 
        left_on=fips_col, 
        right_on='FIPS', 
        how='left'
    )
    df = df.rename(columns={'State_Abbr': 'Hospital_State', 'County_Name': 'Hospital_County'})
    df = df.drop(columns=['FIPS'])
    
    return df

def harmonize_zcta_boundaries(df: pd.DataFrame, zip_col: str = 'ZIP', pop_col: str = 'Population', year_col: str = 'Year', 
        state_col: str = None, county_col: str = None) -> pd.DataFrame:
    """
    Harmonizes 2010-2019 ZCTA populations to 2020 ZCTA boundaries using the 
    local Census Bureau Relationship File.
    """
    # 1. Split the time series at the actual API boundary change (2021)
    df_pre_2020 = df[df[year_col] < 2021].copy()
    df_post_2020 = df[df[year_col] >= 2021].copy()

    # 2. Load the local crosswalk file using the utility function
    crosswalk_path = get_data_path("zcta_crosswalk", "raw")
    crosswalk = pd.read_csv(crosswalk_path, sep='|', dtype={'GEOID_ZCTA5_10': str, 'GEOID_ZCTA5_20': str})
    
    # 3. Calculate the Area-Based Allocation Factor
    crosswalk['AREALAND_ZCTA5_10'] = pd.to_numeric(crosswalk['AREALAND_ZCTA5_10'], errors='coerce')
    crosswalk['AREALAND_PART'] = pd.to_numeric(crosswalk['AREALAND_PART'], errors='coerce')
    
    crosswalk['AF'] = np.where(
        crosswalk['AREALAND_ZCTA5_10'] > 0,
        crosswalk['AREALAND_PART'] / crosswalk['AREALAND_ZCTA5_10'],
        0
    )
    
    # Keep only necessary crosswalk columns
    cw_subset = crosswalk[['GEOID_ZCTA5_10', 'GEOID_ZCTA5_20', 'AF']]
    
    # 4. Merge pre-2020 data with the crosswalk
    merged = df_pre_2020.merge(cw_subset, left_on=zip_col, right_on='GEOID_ZCTA5_10', how='inner')
    
    # 5. Apportion the historical population
    merged['Adjusted_Pop'] = merged[pop_col] * merged['AF']
    
    # 6. Group by the NEW 2020 ZCTA and re-aggregate
    group_cols = ['GEOID_ZCTA5_20', year_col] 
    
    # dynamically add state and county to aggregation if provided
    agg_dict = {'Adjusted_Pop': 'sum'}
    if state_col and state_col in merged.columns:
        agg_dict[state_col] = 'first'
    if county_col and county_col in merged.columns:
        agg_dict[county_col] = 'first'
        
    harmonized_pre_2020 = merged.groupby(group_cols, as_index=False).agg(agg_dict)
    
    # 7. Rename columns to match standard input
    harmonized_pre_2020.rename(columns={
        'GEOID_ZCTA5_20': zip_col,
        'Adjusted_Pop': pop_col
    }, inplace=True)
    
    # 8. Recombine the dataset
    final_df = pd.concat([harmonized_pre_2020, df_post_2020], ignore_index=True)
    
    # 9. Clean up formatting
    final_df[pop_col] = final_df[pop_col].round(0)
    final_df = final_df.sort_values(by=[zip_col, year_col]).reset_index(drop=True)
    
    return final_df
    
def calc_pct_change(df, year_old, year_new):
    return np.where(df[year_old] > 0,
                    ((df[year_new] - df[year_old]) / df[year_old]) * 100,
                    np.nan)


# --- HCUP -------------------------------------------------------------------
# Moved out of notebooks/0.4. Behaviour is unchanged from the notebook version;
# see the caveat in clean_hcup_missing_codes about NaN stringification.

HCUP_MISSING_CODES = [-9999, -9998, -99, -9, '-9999', '-9998', '-99', '-9']

# Identifier columns that arrive with inconsistent formatting across states and
# years, and therefore have to be normalized before they can be merged on.
HCUP_ID_COLS = ['DSHOSPID', 'HOSPID', 'AYEAR', 'YEAR', 'KEY']


def clean_hcup_missing_codes(df: pd.DataFrame, id_cols: list = None) -> pd.DataFrame:
    """Replace HCUP sentinel missing codes with NaN and normalize ID columns.

    HCUP encodes missingness as negative sentinels (-9, -99, -9998, -9999),
    which would otherwise be read as real values. The identifier columns are
    then stripped of float suffixes ('1234.0') and leading zeros so that the
    same hospital compares equal across a CORE and an AHAL file.

    Caveat carried over from the notebook: the final `astype(str)` turns NaN
    into the literal string 'nan'. Rows with a missing identifier will
    therefore match each other on merge. Filter on NaN before merging if that
    matters for your analysis.

    Operates in place and also returns the frame, matching the notebook's use.
    """
    if id_cols is None:
        id_cols = HCUP_ID_COLS

    df.replace(HCUP_MISSING_CODES, np.nan, inplace=True)

    for col in id_cols:
        if col in df.columns:
            mask = df[col].notna()
            df.loc[mask, col] = (
                df.loc[mask, col]
                .astype(str)
                .str.replace(r'\.0$', '', regex=True)
                .str.strip()
                .str.replace(r'^0+(?!$)', '', regex=True)
            )
            df[col] = df[col].astype(str)

    return df


def smart_merge_ahal(df_core: pd.DataFrame, df_ahal: pd.DataFrame) -> pd.DataFrame:
    """Attach AHAL hospital attributes to a CORE discharge file.

    HCUP ships the hospital crosswalk in two incompatible shapes:

    * patient-level (Iowa and friends), keyed by the discharge KEY;
    * hospital-level (most states), keyed by HOSPID or DSHOSPID plus YEAR.

    Which hospital identifier is populated also varies by state and year, so
    the merge key is chosen from whichever column actually carries data. If
    neither does, the function gives up and returns the CORE frame with an
    empty HFIPSSTCO so downstream geocoding degrades to NaN rather than
    raising.
    """
    if df_core.empty or df_ahal.empty:
        return df_core

    # A. Patient-level crosswalk: both sides carry the discharge KEY.
    if 'KEY' in df_core.columns and 'KEY' in df_ahal.columns:
        overlap = [c for c in df_ahal.columns if c in df_core.columns and c != 'KEY']
        ahal_clean = df_ahal.drop(columns=overlap, errors='ignore')
        return df_core.merge(ahal_clean, on='KEY', how='left')

    # B. Hospital-level crosswalk: merge on a hospital ID plus the year.
    if 'AYEAR' in df_core.columns:
        df_core = df_core.rename(columns={'AYEAR': 'YEAR'})

    has_hospid = 'HOSPID' in df_core.columns and df_core['HOSPID'].notna().sum() > 0
    has_dshospid = 'DSHOSPID' in df_core.columns and df_core['DSHOSPID'].notna().sum() > 0

    if has_hospid:
        merge_key, drop_key = 'HOSPID', 'DSHOSPID'
    elif has_dshospid:
        merge_key, drop_key = 'DSHOSPID', 'HOSPID'
    else:
        df_core['HFIPSSTCO'] = np.nan
        return df_core

    ahal_clean = df_ahal.drop(columns=[drop_key], errors='ignore')
    df_core[merge_key] = df_core[merge_key].astype(str)
    ahal_clean[merge_key] = ahal_clean[merge_key].astype(str)

    return df_core.merge(ahal_clean, on=[merge_key, 'YEAR'], how='left')