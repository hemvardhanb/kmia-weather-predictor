import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

# 1. Load Data
df = pd.read_csv('kmia_clean.csv')
# 'M' is scattered, coerce to numeric
df['tmpf'] = pd.to_numeric(df['tmpf'], errors='coerce')
df = df.dropna(subset=['tmpf'])
df['valid'] = pd.to_datetime(df['valid'])

# Extract daily max temperature
daily_highs = df.groupby(df['valid'].dt.date)['tmpf'].max().reset_index()
daily_highs.columns = ['date', 'max_temp']
daily_highs['date'] = pd.to_datetime(daily_highs['date'])
daily_highs['day_of_year'] = daily_highs['date'].dt.dayofyear

# 2. Add features
daily_highs['lag_1'] = daily_highs['max_temp'].shift(1)
daily_highs['lag_7'] = daily_highs['max_temp'].shift(7)
daily_highs = daily_highs.dropna()

# 3. Train
X = daily_highs[['day_of_year', 'lag_1', 'lag_7']]
y = daily_highs['max_temp']

# Use recent data for prediction
model = RandomForestRegressor(n_estimators=200, n_jobs=-1)
model.fit(X, y)

# 4. Predict
tomorrow_doy = (datetime.now().timetuple().tm_yday)
pred_tomorrow = model.predict([[tomorrow_doy, daily_highs.iloc[-1]['max_temp'], daily_highs.iloc[-7]['max_temp']]])
print(f"Predicted high for tomorrow: {pred_tomorrow[0]:.1f} F")
