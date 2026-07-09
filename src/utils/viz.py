import matplotlib.pyplot as plt
import pandas as pd

def plot_zip_population_pivot(pivot_df, zip_list, county_col=None, state_col=None):
    plt.figure(figsize=(12, 6))
    
    # 1. Handle MultiIndex if present
    plot_df = pivot_df.copy()
    if 'ZIP' in plot_df.columns:
        plot_df = plot_df.set_index('ZIP')
    elif isinstance(plot_df.index, pd.MultiIndex):
        plot_df = plot_df.reset_index().set_index(plot_df.index.names[0] if 'ZIP' not in plot_df.index.names else 'ZIP')

    # 2. Isolate ONLY the year columns to prevent plotting text/metric columns
    year_cols = [c for c in plot_df.columns if str(c).isdigit()]
    
    # Ensure index values are strings for consistent lookup
    plot_df.index = plot_df.index.astype(str)
    
    for z in zip_list:
        z_str = str(z)  # Force input to string to handle zero-padding (e.g., '02138') Safely
        
        if z_str not in plot_df.index:
            print(f"Warning: ZIP {z} not found in dataset.")
            continue
            
        row = plot_df.loc[z_str]
        
        # If duplicate ZIP entries exist, grab the first one
        if isinstance(row, pd.DataFrame): 
            row = row.iloc[0]
            
        label = f"ZIP {z_str}"
        location_parts = []
        
        # 3. Extract location strings safely
        if county_col and county_col in plot_df.columns:
            county = row[county_col]
            if pd.notna(county) and str(county).strip(): 
                location_parts.append(str(county))
                
        if state_col and state_col in plot_df.columns:
            state = row[state_col]
            if pd.notna(state) and str(state).strip(): 
                location_parts.append(str(state))
                
        if location_parts:
            label += f" ({', '.join(location_parts)})"
            
        # 4. Plot only the isolated year columns
        plt.plot(year_cols, row[year_cols], marker='.', label=label)
            
    plt.title("Population Over Time by ZCTA")
    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()