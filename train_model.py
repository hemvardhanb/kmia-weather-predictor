import pandas as pd
import requests
import json
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

# Setup
lat, lon = 25.7959, -80.2870
headers = {"User-Agent": "WeatherModelMVP/1.0 (contact@example.com)", "Accept": "application/geo+json"}

def get_forecast():
    points_url = f"https://api.weather.gov/points/{lat},{lon}"
    data = requests.get(points_url, headers=headers).json()
    forecast = requests.get(data['properties']['forecast'], headers=headers).json()
    
    data_list = []
    for p in forecast['properties']['periods']:
        if p['isDaytime']:
            data_list.append({
                'temp': p['temperature'],
                'pop': p['probabilityOfPrecipitation']['value'] or 0,
                'time': p['startTime']
            })
    return pd.DataFrame(data_list)

# Get current data (placeholder for training loop, usually you need 365 days)
# For MVP, we simulate a small dataset based on recent trends + forecast
df = get_forecast()
# Feature engineering (basic)
df['day_of_year'] = pd.to_datetime(df['time']).dt.dayofyear
X = df[['day_of_year', 'pop']]
y = df['temp']

model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)
print("Model trained on 7-day forecast sample.")

# Prediction for next available day
next_day = X.iloc[0:1]
prediction = model.predict(next_day)
print(f"Predicted high for next period: {prediction[0]:.1f} F")
