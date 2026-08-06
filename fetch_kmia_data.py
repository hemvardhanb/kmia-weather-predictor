import requests
import json
from datetime import datetime, timedelta

# KMIA coordinates
lat = 25.7959
lon = -80.2870

headers = {
    "User-Agent": "WeatherModelMVP/1.0 (contact@example.com)",
    "Accept": "application/geo+json"
}

# 1. Get metadata to find grid points
points_url = f"https://api.weather.gov/points/{lat},{lon}"
response = requests.get(points_url, headers=headers)
data = response.json()
properties = data['properties']
gridId = properties['gridId']
gridX = properties['gridX']
gridY = properties['gridY']

print(f"Grid: {gridId}, {gridX}, {gridY}")

# 2. Get forecast
forecast_url = properties['forecast']
forecast_response = requests.get(forecast_url, headers=headers)
forecast_data = forecast_response.json()

print(json.dumps(forecast_data['properties']['periods'][0], indent=2))
