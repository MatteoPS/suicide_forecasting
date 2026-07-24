
from uszipcode import SearchEngine, SimpleZipcode
import pandas as pd

import os
import sys
from pathlib import Path
# Force Python to see the project root directory
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils.config import get_data_path

def download_uszipcode_map():
    """
    One-time ingestion utility to extract the uszipcode SQLite database 
    into a static CSV for fast, repeatable merges in the ETL pipeline.
    """
    out_path = get_data_path("zip_to_state_county", "raw")

    print("Initializing SearchEngine (this may take a minute)...")
    search = SearchEngine()
    
    # CORRECTED: Query the SimpleZipcode model directly
    all_zips = search.ses.query(SimpleZipcode).all()

    print(f"Extracting and formatting {len(all_zips)} ZIP codes...")
    df_zips = pd.DataFrame([{
        'ZIP': getattr(z, 'zipcode'),
        'state': getattr(z, 'state'),
        'county': getattr(z, 'county')
    } for z in all_zips])

    os.makedirs(out_path.parent, exist_ok=True)

    print(f"Saving static mapping to {out_path}...")
    df_zips.to_csv(out_path, index=False)
    print("Static ZIP mapping generated successfully.")

if __name__ == "__main__":
    download_uszipcode_map()