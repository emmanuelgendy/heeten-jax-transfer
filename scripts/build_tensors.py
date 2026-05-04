import pandas as pd
import numpy as np

# Notice freq="1h" (lowercase 'h') to fix the pandas version issue
def build_heeten_tensor(csv_paths, weather_csv_path, pricing_csv_path, output_path="data/processed/heeten_complex_hems_data.npz", freq="1h"):
    print(f"Processing {len(csv_paths)} buildings...")
    building_dfs = []
    
    for path in csv_paths:
        # 1. Read the CSV without trying to parse dates yet
        df = pd.read_csv(path)
        
        # 2. Convert the 'unixtime' column into a proper datetime 'timestamp'
        df['timestamp'] = pd.to_datetime(df['unixtime'], unit='s')
        df = df.set_index('timestamp')
        
        # 3. Resample to 1 Hour and handle missing values
        df = df.resample(freq).mean()
        
        # Using the exact column names you provided: 'pv' and 'load'
        df['pv'] = df['pv'].ffill(limit=2).fillna(0.0)
        df['load'] = df['load'].ffill(limit=2).fillna(df['load'].median())
        building_dfs.append(df)

    # Find common timeframe across all houses
    common_index = building_dfs[0].index
    for df in building_dfs[1:]:
        common_index = common_index.intersection(df.index)
        
    print(f"Common timeframe aligned: {common_index.min()} to {common_index.max()}")

    # Stack 2D arrays (Shape: 5, T)
    pv_stacked = np.stack([df.loc[common_index, 'pv'].values for df in building_dfs])
    load_stacked = np.stack([df.loc[common_index, 'load'].values for df in building_dfs])

    # Process 1D Exogenous Data (Weather & Prices)
    weather_df = pd.read_csv(weather_csv_path, parse_dates=['timestamp']).set_index('timestamp')
    weather_df = weather_df.resample(freq).mean().interpolate()
    outdoor_temp = weather_df.loc[common_index, 'temperature_c'].values

    price_df = pd.read_csv(pricing_csv_path, parse_dates=['timestamp']).set_index('timestamp')
    price_df = price_df.resample(freq).ffill()
    import_price = price_df.loc[common_index, 'import_price_eur_kwh'].values
    export_price = price_df.loc[common_index, 'export_price_eur_kwh'].values

    # Export to NPZ
    np.savez_compressed(
        output_path,
        pv_data=pv_stacked.astype(np.float32),
        load_data=load_stacked.astype(np.float32),
        outdoor_temp_data=outdoor_temp.astype(np.float32),
        import_price_data=import_price.astype(np.float32),
        export_price_data=export_price.astype(np.float32)
    )
    print(f"Successfully saved tensors to {output_path} (T={len(common_index)})")

if __name__ == "__main__":
    csv_files = [
        "data/raw/heeten_building_df14.csv",
        "data/raw/heeten_building_df18.csv",
        "data/raw/heeten_building_df20.csv",
        "data/raw/heeten_building_df60.csv",
        "data/raw/heeten_building_df69.csv",
    ]
    build_heeten_tensor(csv_files, "data/raw/weather.csv", "data/raw/prices.csv")