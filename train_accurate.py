import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import numpy as np

# 1. Load Extended Data
df = pd.read_csv('kmia_extended.csv')
for col in ['tmpf', 'dwpf', 'sknt', 'drct']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna()
df['valid'] = pd.to_datetime(df['valid'])

# Extract daily max for features
daily = df.groupby(df['valid'].dt.date).agg({
    'tmpf': 'max',
    'dwpf': 'mean',
    'sknt': 'mean',
    'drct': 'mean'
}).reset_index()
daily.columns = ['date', 'max_temp', 'mean_dewpoint', 'mean_wind_speed', 'mean_wind_dir']
daily['date'] = pd.to_datetime(daily['date'])
daily['day_of_year'] = daily['date'].dt.dayofyear
daily['month'] = daily['date'].dt.month

# 2. Anomaly setup
climate_means = daily.groupby('month')['max_temp'].mean()
daily['anomaly'] = daily['max_temp'] - daily['month'].map(climate_means)

# 3. Add Lags + New Features
daily['lag_1_anomaly'] = daily['anomaly'].shift(1)
daily['lag_7_anomaly'] = daily['anomaly'].shift(7)
daily = daily.dropna()

# 4. Train
X = daily[['day_of_year', 'mean_dewpoint', 'mean_wind_speed', 'lag_1_anomaly', 'lag_7_anomaly']]
y = daily['anomaly']

model = RandomForestRegressor(n_estimators=200, n_jobs=-1)
model.fit(X, y)

def predict_high(target_date_str):
    target_date = pd.to_datetime(target_date_str)
    doy = target_date.timetuple().tm_yday
    
    # Use most recent observations as baseline
    last_row = daily.iloc[-1]
    
    anomaly_pred = model.predict([[doy, last_row['mean_dewpoint'], last_row['mean_wind_speed'], last_row['lag_1_anomaly'], last_row['lag_7_anomaly']]])[0]
    
    raw_pred = anomaly_pred + climate_means[target_date.month]
    return (climate_means[target_date.month] * 0.7) + (raw_pred * 0.3)

print(f"Predicted high for Aug 7: {predict_high('2026-08-07'):.1f} F")
