import glob
import os
import re
from typing import Literal

import pandas as pd
import requests
import warnings

from src.utils.config import BRFSS_PATH, CENSUS_API_KEY, get_data_path


def load_nvdrs(file_key: str, data_folder: str, usecols: list | None = None, nrows: int | None = None) -> pd.DataFrame:
    """Loads the NVDRS dataset from local storage, handling Windows encoding."""
    nvdrs_path = get_data_path(file_key, data_folder)
    
    df = pd.read_csv(
        nvdrs_path, 
        encoding="cp1252", 
        encoding_errors="replace", # Replaces problematic characters instead of crashing
        low_memory=True, 
        dtype={'DeathDate': str, 'DeathDate_myr': str, 'DeathDate_year': str},
        nrows=nrows,
        usecols=usecols
    )
    return df

def load_brfss(year: int | str, brfss_path: str | None = None) -> pd.DataFrame:
    """Loads local BRFSS SAS data for a given year, warning if versioned files exist."""
    if not brfss_path:
        brfss_path = BRFSS_PATH
    else:
        print(f"Loading from {brfss_path} folder")
    
    # Catch missing environment variables immediately
    if not brfss_path:
        raise ValueError("BRFSS_PATH is missing. Specify it in your .env file or pass it directly.")   
        
    base_file = os.path.join(brfss_path, f"LLCP{year}.XPT")
    version_files = glob.glob(os.path.join(brfss_path, f"LLCP{year}V*.XPT"))
    
    if version_files:
        print(f"WARNING: Alternative versions found for {year}: {[os.path.basename(f) for f in version_files]}. Proceeding with base file 'LLCP{year}.XPT'...")
        
    if not os.path.exists(base_file):
        raise FileNotFoundError(f"Base file not found: {base_file}")
        
    return pd.read_sas(base_file, format='xport')

def fetch_census(variables_dict: dict, years: list, geo_level: Literal["county", "state"] = "county", states: str | list = "*") -> pd.DataFrame:
    """Fetches ACS 5-Year Data Profiles from the US Census API."""
    
    # Convert list of states to comma-separated string for the API
    if isinstance(states, list):
        states = ",".join(states)

    all_data = []
    
    for year in years:
        base_url = f"https://api.census.gov/data/{year}/acs/acs5/profile"
        
        # Dynamically build geography parameters
        params = {
            "get": ",".join(variables_dict.keys()),
            "key": CENSUS_API_KEY
        }
        
        if geo_level == "state":
            params["for"] = f"state:{states}"
        elif geo_level == "county":
            params["for"] = "county:*"
            if states != "*":
                params["in"] = f"state:{states}"
        
        response = requests.get(base_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            df_year = pd.DataFrame(data[1:], columns=data[0])
            df_year['Year'] = year 
            all_data.append(df_year)
        else:
            print(f"Warning: Data for {year} not available. Status: {response.status_code} | Details: {response.text}")
    
    if not all_data:
        print("No data retrieved for any requested years.")
        return pd.DataFrame()
        
    df_final = pd.concat(all_data, ignore_index=True)
    df_final.rename(columns=variables_dict, inplace=True)
    return df_final

def fetch_hcup(catalog, state_name: str, db_type: Literal["sedd", "sid"], years: list, cols_to_pull: list, icd_prefix: str, icd_vals: list, return_icd_cols: bool = False) -> pd.DataFrame:
    """Builds a dynamic BigQuery SQL string and fetches HCUP data via Redivis."""
    
    # 1. Fetch tables from the local cache
    dataset_ref = catalog.datasets[catalog.datasets["Dataset_Name"].str.contains(state_name, case=False)]["Reference"].iloc[0]
    df_tables = catalog.get_tables(dataset_ref)
    
    year_pattern = "|".join([str(y) for y in years])
    mask = (
        df_tables["Table_Name"].str.contains(db_type, case=False) &
        df_tables["Table_Name"].str.contains("CORE", case=False) &
        df_tables["Table_Name"].str.contains(year_pattern, flags=re.IGNORECASE, regex=True)
    )
    target_tables = df_tables[mask]
    print(f"Fetching {state_name} {db_type} {year_pattern}...")
    if target_tables.empty:
        raise ValueError(f"No tables matched your criteria for {state_name}, {db_type}, years: {years}")

    # 2. Fetch variables from cache and build master list
    table_vars = {}
    master_icd_cols = set()

    for _, row in target_tables.iterrows():
        t_ref = row["Reference"]
        df_vars = catalog.get_variables(dataset_ref, t_ref)
        vars_in_table = df_vars["Variable"].str.upper().tolist()
        
        table_vars[t_ref] = vars_in_table
        
        icd_in_table = [v for v in vars_in_table if v.startswith(icd_prefix.upper())]
        master_icd_cols.update(icd_in_table)

    master_icd_cols = sorted(list(master_icd_cols))
    
    if return_icd_cols:
        master_schema = [c.upper() for c in cols_to_pull] + master_icd_cols
    else:
        master_schema = [c.upper() for c in cols_to_pull]

    regex_pattern = r"^(" + "|".join([str(x) for x in icd_vals]) + ")"

    # 3 & 4. Build SQL and execute PER TABLE to catch year-specific errors
    dfs = []
    for _, row in target_tables.iterrows():
        t_ref = row["Reference"]
        
        qualified_ref = f"{catalog.org_name}.{dataset_ref}.{t_ref}"
        vars_in_table = table_vars[t_ref]
        
        valid_table_icds = [c for c in master_icd_cols if c in vars_in_table]
        if not valid_table_icds: 
            continue
            
        select_elements = []
        for col in master_schema:
            if col in vars_in_table:
                select_elements.append(col)
            else:
                select_elements.append(f"NULL AS {col}") 
        
        select_clause = ", ".join(select_elements)
        
        where_conditions = [f"REGEXP_CONTAINS(CAST({col} AS STRING), r'{regex_pattern}')" for col in valid_table_icds]
        where_clause = " OR \n    ".join(where_conditions)

        query = f"SELECT {select_clause} \nFROM `{qualified_ref}` \nWHERE {where_clause}"
        
        # Execute individual table query
        try:
            df_year = catalog.org.dataset(dataset_ref).query(query).to_pandas_dataframe()
            dfs.append(df_year)
        except Exception as e:
            warnings.warn(
                f"\nWARNING: Skipping dataset. {state_name} - {t_ref} is too large or access was denied.\n"
                f"This state/year will be missing in the final df.\nError Details: {e}"
            )
        
    if not dfs:
        raise ValueError(f"No data could be retrieved for {state_name}. All tables failed or were empty.")

    # Combine successful queries
    df_out = pd.concat(dfs, ignore_index=True)

    return df_out

def fetch_hcup_ahal(catalog, state_name: str, years: list) -> pd.DataFrame:
    """Fetches HCUP AHA Linkage (AHAL) data via Redivis."""
    
    dataset_ref = catalog.datasets[catalog.datasets["Dataset_Name"].str.contains(state_name, case=False)]["Reference"].iloc[0]
    df_tables = catalog.get_tables(dataset_ref)
    
    # Extract unique 4-digit years to handle quarters (e.g., "2015q1q3" -> "2015")
    clean_years = list(set([str(y)[:4] for y in years]))
    year_pattern = "|".join(clean_years)
    
    mask = (
        df_tables["Table_Name"].str.contains("AHAL", case=False) &
        df_tables["Table_Name"].str.contains(year_pattern, flags=re.IGNORECASE, regex=True)
    )
    target_tables = df_tables[mask]
    
    print(f"Fetching {state_name} AHAL {year_pattern}...")
    if target_tables.empty:
        raise ValueError(f"No AHAL tables matched your criteria for {state_name}, years: {clean_years}")

    dfs = []
    for _, row in target_tables.iterrows():
        t_ref = row["Reference"]
        qualified_ref = f"{catalog.org_name}.{dataset_ref}.{t_ref}"
        
        query = f"SELECT * \nFROM `{qualified_ref}`"
        
        try:
            df_year = catalog.org.dataset(dataset_ref).query(query).to_pandas_dataframe()
            dfs.append(df_year)
        except Exception as e:
            warnings.warn(
                f"\nWARNING: Skipping AHAL dataset. {state_name} - {t_ref} failed.\n"
                f"This state/year will be missing in the final df.\nError Details: {e}"
            )
        
    if not dfs:
        raise ValueError(f"No AHAL data could be retrieved for {state_name}. All tables failed or were empty.")

    df_out = pd.concat(dfs, ignore_index=True)

    return df_out