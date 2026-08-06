import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

# 1. Load Data (Use a more targeted approach)
df = pd.read_csv('kmia_clean.csv')
df['tmpf'] = pd.to_numeric(df['tmpf'], errors='coerce')
df = df.dropna(subset=['tmpf'])
df['valid'] = pd.to_datetime(df['valid'])

# Filter data to recent months to capture current seasonal norms better
# Keep only months 6, 7, 8, 9 (Summer in Miami)
df_summer = df[df['valid'].dt.month.isin([6, 7, 8, 9])]

# Extract daily max temperature
daily_highs = df_summer.groupby(df_summer['valid'].dt.date)['tmpf'].max().reset_index()
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

# Use RandomForest
model = RandomForestRegressor(n_estimators=200, n_jobs=-1)
model.fit(X, y)

# 4. Predict for Aug 7 (Day 219)
# Use recent lag values from the end of the dataset to be as current as possible
target_doy = 219
last_max = daily_highs.iloc[-1]['max_temp']
lag_7 = daily_highs.iloc[-7]['max_temp']

pred_tomorrow = model.predict([[target_doy, last_max, lag_7]])
print(f"Predicted high for tomorrow: {pred_tomorrow[0]:.1f} F")
