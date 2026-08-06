import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

# MVP Weather Prediction Model for KMIA
# Note: For production-level accuracy, replace 'get_forecast()' 
# with a function that reads historical CSV files from a site like 
# Iowa State Mesonet (ASOS/AWOS archive).

class WeatherModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100)
        self.lat, self.lon = 25.7959, -80.2870
        self.headers = {"User-Agent": "WeatherModelMVP/1.0 (contact@example.com)", "Accept": "application/geo+json"}

    def get_data(self):
        # Fetch current forecast as a baseline/example
        points_url = f"https://api.weather.gov/points/{self.lat},{self.lon}"
        data = requests.get(points_url, headers=self.headers).json()
        forecast = requests.get(data['properties']['forecast'], headers=self.headers).json()
        
        data_list = []
        for p in forecast['properties']['periods']:
            if p['isDaytime']:
                data_list.append({
                    'temp': p['temperature'],
                    'pop': p['probabilityOfPrecipitation']['value'] or 0,
                    'time': p['startTime']
                })
        return pd.DataFrame(data_list)

    def train(self):
        df = self.get_data()
        df['day_of_year'] = pd.to_datetime(df['time']).dt.dayofyear
        X = df[['day_of_year', 'pop']]
        y = df['temp']
        self.model.fit(X, y)
        return "Model trained on sample set."

    def predict(self, pop):
        day = datetime.now().timetuple().tm_yday
        return self.model.predict([[day, pop]])[0]

if __name__ == "__main__":
    wm = WeatherModel()
    print(wm.train())
    print(f"Predicted high with 50% pop: {wm.predict(50):.1f} F")
