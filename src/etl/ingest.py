import os
import pandas as pd
import requests
import glob
from dotenv import load_dotenv
from src.utils.config import get_data_path

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
        load_dotenv()
        brfss_path = os.getenv("BRFSS_PATH")
        
    base_file = os.path.join(brfss_path, f"LLCP{year}.XPT")
    version_files = glob.glob(os.path.join(brfss_path, f"LLCP{year}V*.XPT"))
    
    if version_files:
        print(f"WARNING: Alternative versions found for {year}: {[os.path.basename(f) for f in version_files]}. Proceeding with base file 'LLCP{year}.XPT'...")
        
    if not os.path.exists(base_file):
        raise FileNotFoundError(f"Base file not found: {base_file}")
        
    return pd.read_sas(base_file, format='xport')

def load_census(variables_dict: dict, years: list, geo_level: str = "county", states="*" ) -> pd.DataFrame:
    """
    Fetches Census ACS 5-Year Data Profiles.
    geo_level: 'county' or 'state'
    states: '*' (all), a single FIPS string (e.g., '36'), or a list of FIPS strings (e.g., ['36', '34'])
    """
    load_dotenv()
    api_key = os.getenv("CENSUS_API_KEY")
    
    if not api_key:
        raise ValueError("CENSUS_API_KEY not found in .env file.")

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