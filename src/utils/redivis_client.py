
import pandas as pd
import redivis

class RedivisCatalog:
    def __init__(self, org_name):
        self.org_name = org_name
        self.org = redivis.organization(org_name)
        
        # Internal caches to prevent redundant API calls
        self._datasets_cache = None
        self._tables_cache = {}
        self._schema_cache = {}

    @property
    def datasets(self):
        if self._datasets_cache is None:
            ds_list = self.org.list_datasets()
            self._datasets_cache = pd.DataFrame({
                "Dataset_Name": [d.name for d in ds_list],
                "Reference": [d.qualified_reference.split('.')[-1] for d in ds_list],
                "Updated_At": [pd.to_datetime(d.properties.get("updatedAt"), unit='ms') if d.properties and d.properties.get("updatedAt") else None for d in ds_list]
            })
        return self._datasets_cache

    def get_tables(self, dataset_ref):
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

    def get_variables(self, dataset_ref, table_ref):
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
        self._datasets_cache = None
        self._tables_cache = {}
        self._schema_cache = {}

def print_redivis_tree(org_name: str, limit_tables: int=3):
    org = redivis.organization(org_name)
    print(f"Organization: {org_name}")
    for ds in org.list_datasets():
        # Extracts 'new_jersey_hcup:5bca' from the full reference
        ds_ref = ds.qualified_reference.split('.')[-1] 
        print(f"\nDataset: {ds_ref}")
        
        tables = ds.list_tables()
        for t in tables[:limit_tables]: 
            # Extracts 'nj_sedd_2012_core:wrs0'
            t_ref = t.qualified_reference.split('.')[-1]
            print(f"  ↳ {t_ref}")
            
        if len(tables) > limit_tables:
            print(f"  ↳ ... (+{len(tables) - limit_tables} more tables)")
