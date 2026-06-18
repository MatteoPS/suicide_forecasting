import os
import pandas as pd
import requests
import glob
from src.utils.config import get_data_path, CENSUS_API_KEY, BRFSS_PATH
import re

def load_nvdrs(file_key: str, data_folder: str, nrows: int = None) -> pd.DataFrame:
    """Loads the primary NVDRS dataset, handling Windows encoding."""
    nvdrs_path = get_data_path(file_key, data_folder)
    
    # Add the encoding parameter here
    df = pd.read_csv(
        nvdrs_path, 
        encoding="cp1252", 
        encoding_errors="replace", #if files have some encoding issues, this will replace problematic characters instead of throwing an error
        low_memory=False,  # for large files
        nrows=nrows  # to limit the number of rows read
    )
    
    return df

def load_brfss(year, brfss_path: str = None) -> pd.DataFrame:
    """Loads BRFSS data for a specific year, flagging if versioned files exist."""
    if not brfss_path:
        brfss_path = BRFSS_PATH
    else:
        print(f"loading from {brfss_path} folder")
    
    if not brfss_path:
        raise ValueError("BRFSS_PATH is missing. Specify it in your .env file or pass it directly to the function `load_brfss(year, brfss_path`.")   
        
    base_file = os.path.join(brfss_path, f"LLCP{year}.XPT")
    version_files = glob.glob(os.path.join(brfss_path, f"LLCP{year}V*.XPT"))
    
    if version_files:
        print(f"WARNING: Alternative versions found for {year}: {[os.path.basename(f) for f in version_files]}. Proceeding with base file 'LLCP{year}.XPT'...")
        
    if not os.path.exists(base_file):
        raise FileNotFoundError(f"Base file not found: {base_file}")
        
    return pd.read_sas(base_file, format='xport')

def fetch_census(variables_dict: dict, years: list, geo_level: str = "county", states="*" ) -> pd.DataFrame:
    """
    Fetches Census ACS 5-Year Data Profiles.
    geo_level: 'county' or 'state'
    states: '*' (all), a single FIPS string (e.g., '36'), or a list of FIPS strings (e.g., ['36', '34'])
    """
    api_key = CENSUS_API_KEY


    # Convert list of states to comma-separated string for the API
    if isinstance(states, list):
        states = ",".join(states)

    all_data = []
    
    for year in years:
        base_url = f"https://api.census.gov/data/{year}/acs/acs5/profile"
        
        # Dynamically build geography parameters
        params = {
            "get": ",".join(variables_dict.keys()),
            "key": api_key
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

def fetch_hcup(catalog, state_name, db_type, years, cols_to_pull, icd_prefix, icd_vals):

    dataset_ref = catalog.datasets[catalog.datasets["Dataset_Name"].str.contains(state_name, case=False)]["Reference"].iloc[0]
    # 1. Fetch tables from the local cache
    df_tables = catalog.get_tables(dataset_ref)
    
    # Filter tables using pandas string matching
    year_pattern = "|".join([str(y) for y in years])
    mask = (
        df_tables["Table_Name"].str.contains(db_type, case=False) &
        df_tables["Table_Name"].str.contains("CORE", case=False) &
        df_tables["Table_Name"].str.contains(year_pattern, flags=re.IGNORECASE, regex=True)
    )
    target_tables = df_tables[mask]

    if target_tables.empty:
        return "No tables matched your criteria."

    # 2. Fetch variables from cache and build master list
    table_vars = {}
    master_icd_cols = set()

    for _, row in target_tables.iterrows():
        t_ref = row["Reference"]
        
        # Fetch from local schema cache
        df_vars = catalog.get_variables(dataset_ref, t_ref)
        vars_in_table = df_vars["Variable"].str.upper().tolist()
        
        table_vars[t_ref] = vars_in_table
        
        icd_in_table = [v for v in vars_in_table if v.startswith(icd_prefix.upper())]
        master_icd_cols.update(icd_in_table)

    master_icd_cols = sorted(list(master_icd_cols))
    master_schema = [c.upper() for c in cols_to_pull] + master_icd_cols

    # --- DIRECT REGEX FORMATTING (SAFE) ---
    regex_pattern = r"^(" + "|".join([str(x) for x in icd_vals]) + ")"

    # 3. Build SQL
    sql_parts = []
    for _, row in target_tables.iterrows():
        t_ref = row["Reference"]
        
        # Reconstruct the full BigQuery reference string manually
        qualified_ref = f"{catalog.org_name}.{dataset_ref}.{t_ref}"
        
        vars_in_table = table_vars[t_ref]
        
        select_elements = []
        for col in master_schema:
            if col in vars_in_table:
                select_elements.append(col)
            else:
                select_elements.append(f"NULL AS {col}")
        
        select_clause = ", ".join(select_elements)
        
        valid_table_icds = [c for c in master_icd_cols if c in vars_in_table]
        where_conditions = [f"REGEXP_CONTAINS({col}, r'{regex_pattern}')" for col in valid_table_icds]
        where_clause = " OR \n    ".join(where_conditions)

        query = f"SELECT {select_clause} \nFROM `{qualified_ref}` \nWHERE {where_clause}"
        sql_parts.append(query)
    sql_string= "\nUNION ALL\n".join(sql_parts)

    # Execute the SQL query via the Redivis API
    df_out = catalog.org.dataset(dataset_ref).query(sql_string).to_pandas_dataframe()

    return df_out, sql_string