import pandas as pd
from typing import Literal

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
    cleaned_df['InjuryDate'] = pd.to_datetime(cleaned_df['InjuryDate'])
    return cleaned_df

def aggregate_nvdrs_daily(df: pd.DataFrame, geo_level: Literal["county", "state"] = None) -> pd.DataFrame:
    """Aggregates incidents by day (one day per row), optionally separating counts by county. Default = Country (no separation)"""
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