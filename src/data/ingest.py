import pandas as pd
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


