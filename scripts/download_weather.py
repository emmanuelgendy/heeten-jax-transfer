import requests
import pandas as pd

def fetch_heeten_weather(start_date="2022-01-01", end_date="2022-12-31", output_path="data/raw/weather.csv"):
    print(f"Downloading real historical weather for Heeten from {start_date} to {end_date}...")
    
    # Heeten, Netherlands coordinates
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude=52.33&longitude=6.28&"
        f"start_date={start_date}&end_date={end_date}&"
        f"hourly=temperature_2m&timezone=Europe%2FAmsterdam"
    )
    
    response = requests.get(url)
    data = response.json()
    
    # Convert to DataFrame
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(data["hourly"]["time"]),
        "temperature_c": data["hourly"]["temperature_2m"]
    })
    
    df.to_csv(output_path, index=False)
    print(f"Weather saved to {output_path}! (Rows: {len(df)})")

if __name__ == "__main__":
    # IMPORTANT: Change these dates to match your Heeten dataset!
    fetch_heeten_weather("2018-03-23", "2020-08-31", "data/raw/weather.csv")