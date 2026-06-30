import pandas as pd
import redivis
from IPython.display import display

class RedivisCatalog:
    """Manages and caches Redivis organization metadata to minimize API calls."""
    
    def __init__(self, org_name: str):
        self.org_name = org_name
        self.org = redivis.organization(org_name)
        
        # Initialize empty caches for datasets, tables, and schemas
        self._datasets_cache = None
        self._tables_cache = {}
        self._schema_cache = {}

    @property
    def datasets(self) -> pd.DataFrame:
        """Returns a cached DataFrame of all datasets in the organization."""
        if self._datasets_cache is None:
            ds_list = self.org.list_datasets()
            self._datasets_cache = pd.DataFrame({
                "Dataset_Name": [d.name for d in ds_list],
                "Reference": [d.qualified_reference.split('.')[-1] for d in ds_list],
                "Updated_At": [pd.to_datetime(d.properties.get("updatedAt"), unit='ms') if d.properties and d.properties.get("updatedAt") else None for d in ds_list]
            })
        return self._datasets_cache

    def get_tables(self, dataset_ref: str) -> pd.DataFrame:
        """Returns a cached DataFrame of all tables for a specific dataset reference."""
        if dataset_ref not in self._tables_cache:
            tables = self.org.dataset(dataset_ref).list_tables()
            self._tables_cache[dataset_ref] = pd.DataFrame({
                "Table_Name": [t.name for t in tables],
                "Reference": [t.qualified_reference.split('.')[-1] for t in tables],
                "Rows": [t.properties.get("numRows") if t.properties else None for t in tables],
                "Columns": [t.properties.get("variableCount") if t.properties else None for t in tables],
                "Size_Bytes": [t.properties.get("numBytes") if t.properties else None for t in tables],
                "Updated_At": [pd.to_datetime(t.properties.get("updatedAt"), unit='ms') if t.properties and t.properties.get("updatedAt") else None for t in tables],
                "Description": [t.properties.get("description") if t.properties else None for t in tables]
            })
        return self._tables_cache[dataset_ref]

    def get_variables(self, dataset_ref: str, table_ref: str) -> pd.DataFrame:
        """Returns a cached DataFrame of variable schemas for a specific table."""
        cache_key = f"{dataset_ref}.{table_ref}"
        if cache_key not in self._schema_cache:
            variables = self.org.dataset(dataset_ref).table(table_ref).list_variables()
            self._schema_cache[cache_key] = pd.DataFrame({
                "Variable": [v.name for v in variables],
                "Type": [v.properties.get("type") if v.properties else None for v in variables],
                "Label": [v.properties.get("label") if v.properties else None for v in variables],
                "Description": [v.properties.get("description") if v.properties else None for v in variables]
            })
        return self._schema_cache[cache_key]
    
    def clear_cache(self):
        """Resets all internal metadata caches."""
        self._datasets_cache = None
        self._tables_cache = {}
        self._schema_cache = {}

def print_redivis_tree(org_name: str, limit_tables: int=3):
    """Prints a hierarchical CLI view of datasets and their tables."""
    org = redivis.organization(org_name)
    print(f"Organization: {org_name}")
    
    for ds in org.list_datasets():
        # Isolate the dataset reference ID (e.g., 'new_jersey_hcup:5bca')
        ds_ref = ds.qualified_reference.split('.')[-1] 
        print(f"\nDataset: {ds_ref}")
        
        tables = ds.list_tables()
        for t in tables[:limit_tables]: 
            # Isolate the table reference ID (e.g., 'nj_sedd_2012_core:wrs0')
            t_ref = t.qualified_reference.split('.')[-1]
            print(f"  ↳ {t_ref}")
            
        if len(tables) > limit_tables:
            print(f"  ↳ ... (+{len(tables) - limit_tables} more tables)")


def display_hcup_variables(catalog, state: str, year: str | int, db_type: str, table: str = 'core'):
    """Finds and displays the variables for a specific HCUP table."""
    
    # 1. Get Dataset Reference
    ds_match = catalog.datasets['Dataset_Name'].str.contains(state, case=False, regex=False)
    if not ds_match.any():
        raise ValueError(f"No dataset found matching state: {state}")
    dataset_ref = catalog.datasets[ds_match]['Reference'].iloc[0]
    
    # 2. Get Table Reference
    df_tables = catalog.get_tables(dataset_ref)
    tbl_mask = (
        df_tables['Table_Name'].str.contains(str(year), case=False) &
        df_tables['Table_Name'].str.contains(db_type, case=False) &
        df_tables['Table_Name'].str.contains(table, case=False)
    )
    if not tbl_mask.any():
        raise ValueError(f"No table found matching year: {year}, db_type: {db_type}, table: {table}")
    table_ref = df_tables[tbl_mask]['Reference'].iloc[0]
    
    # 3. Get and Display Variables
    df_vars = catalog.get_variables(dataset_ref, table_ref)
    display(df_vars)
    
    return df_vars