import yaml
import pandas as pd
from pathlib import Path

# This dynamically finds the root of your repository, 
# no matter where the code is run from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def load_config(config_name: str = "data.yaml") -> dict:
    """Loads a YAML configuration file."""
    config_path = PROJECT_ROOT / "config" / config_name
    
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        
    return config

def get_data_path(file_key: str, data_folder: str ) -> Path:
    """Convenience function to get the absolute path to a data file."""
    config = load_config()
    relative_path = config["paths"][data_folder][file_key]
    return PROJECT_ROOT / relative_path

def load_nvdrs(file_key: str, data_folder: str, nrows: int = None) -> pd.DataFrame:
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
