import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import numpy as np

# 1. Load Data
df = pd.read_csv('kmia_clean.csv')
df['tmpf'] = pd.to_numeric(df['tmpf'], errors='coerce')
df = df.dropna(subset=['tmpf'])
df['valid'] = pd.to_datetime(df['valid'])

# Extract daily max
daily_highs = df.groupby(df['valid'].dt.date)['tmpf'].max().reset_index()
daily_highs.columns = ['date', 'max_temp']
daily_highs['date'] = pd.to_datetime(daily_highs['date'])
daily_highs['month'] = daily_highs['date'].dt.month
daily_highs['day_of_year'] = daily_highs['date'].dt.dayofyear

# 2. Anomaly-based learning (Crucial fix)
climate_means = daily_highs.groupby('month')['max_temp'].mean()
daily_highs['climate_mean'] = daily_highs['month'].map(climate_means)
daily_highs['anomaly'] = daily_highs['max_temp'] - daily_highs['climate_mean']

# 3. Lags
daily_highs['lag_1_anomaly'] = daily_highs['anomaly'].shift(1)
daily_highs['lag_7_anomaly'] = daily_highs['anomaly'].shift(7)
daily_highs = daily_highs.dropna()

# 4. Train
X = daily_highs[['day_of_year', 'lag_1_anomaly', 'lag_7_anomaly']]
y = daily_highs['anomaly']

model = RandomForestRegressor(n_estimators=200, n_jobs=-1)
model.fit(X, y)

def predict_high(target_date_str):
    target_date = pd.to_datetime(target_date_str)
    doy = target_date.timetuple().tm_yday
    
    last_anomaly = daily_highs.iloc[-1]['lag_1_anomaly']
    lag_7_anomaly = daily_highs.iloc[-1]['lag_7_anomaly']
    
    anomaly_pred = model.predict([[doy, last_anomaly, lag_7_anomaly]])[0]
    
    # Add back the climate mean for that month
    return anomaly_pred + climate_means[target_date.month]

print(f"Predicted high for Aug 7: {predict_high('2026-08-07'):.1f} F")
print(f"Predicted high for Dec 25: {predict_high('2026-12-25'):.1f} F")
