import pandas as pd
import numpy as np

def generate_realistic_prices(start_date="2023-01-01", end_date="2023-01-31", output_path="data/raw/prices.csv"):
    print("Generating representative dynamic day-ahead tariff...")
    
    # Create hourly index
    dates = pd.date_range(start=start_date, end=end_date, freq="1h")
    df = pd.DataFrame({"timestamp": dates})
    
    # Base market dynamics (Euro / kWh)
    base_price = 0.15
    
    # Create realistic daily curves (Morning peak, evening peak, midday dip)
    hour = df["timestamp"].dt.hour
    daily_profile = (
        0.05 * np.exp(-0.5 * ((hour - 8) / 2)**2) +   # Morning peak (8 AM)
        0.10 * np.exp(-0.5 * ((hour - 19) / 2)**2) -  # Evening peak (7 PM)
        0.05 * np.exp(-0.5 * ((hour - 13) / 3)**2)    # Midday solar dip
    )
    
    # Weekend discount (prices are usually lower on weekends)
    is_weekend = df["timestamp"].dt.dayofweek >= 5
    weekend_multiplier = np.where(is_weekend, 0.7, 1.0)
    
    # Market volatility (random noise)
    noise = np.random.normal(0, 0.02, len(df))
    
    # Calculate final import price (ensure it never goes absurdly negative)
    df["import_price_eur_kwh"] = np.maximum(0.01, (base_price + daily_profile + noise) * weekend_multiplier)
    
    # Export price (Feed-in tariff) is usually a flat rate or a fraction of import
    df["export_price_eur_kwh"] = 0.05 
    
    df.to_csv(output_path, index=False)
    print(f"Prices saved to {output_path}! (Rows: {len(df)})")

if __name__ == "__main__":
    # IMPORTANT: Change these dates to match your Heeten dataset!
    generate_realistic_prices("2018-08-23", "2020-08-31", "data/raw/prices.csv")