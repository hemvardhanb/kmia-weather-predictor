# KMIA Weather Predictor

An accurate, year-round daily high temperature prediction model for Miami International Airport (KMIA).

## Overview
This project implements a production-grade machine learning model to forecast the next-day high temperature at KMIA. Unlike heuristic or forecast‑based approaches, the model learns from **years of historical ASOS/AWOS observations** (temperature, dew point, wind, pressure) using a `GradientBoostingRegressor`. It employs anomaly‑based learning and time‑ordered validation to provide skillful predictions in any season.

## Features
- **Historical Data Integration:** Automatically downloads 5+ years of hourly observations from the Iowa State Mesonet (ASOS network).
- **Advanced Feature Engineering:** Includes cyclic day‑of‑year, lagged temperatures, rolling averages, and meteorological variables (dew point, wind speed, pressure).
- **Robust Modeling:** Uses a Gradient Boosting Regressor with a time‑ordered train/test split to avoid data leakage.
- **Performance Evaluation:** Reports Mean Absolute Error (MAE) against persistence and climatology baselines.
- **Live Sanity Check:** Optionally compares its prediction to the current NWS forecast.
- **Year‑Round Capability:** Anomaly‑based formulation enables accurate forecasts for any date.

## Usage

### Prerequisites
- Python 3.11+
- `uv` package manager (or `pip`)

### Setup
1. Clone the repository.
2. (Optional) Create a virtual environment:
   ```bash
   uv venv
   # On Windows:
   .\.venv\Scripts\activate
   # On Unix/macOS:
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   uv pip install pandas scikit-learn requests numpy
   # or: pip install pandas scikit-learn requests numpy
   ```

### Running the Predictor
To update the historical dataset and generate a next‑day high prediction:
```bash
python data_collector.py   # Fetch/latest ASOS data (creates kmia_extended.csv)
python weather_predictor.py # Train/evaluate model and predict tomorrow's high
```

The script will output:
- Backtest MAE for the model, persistence, and climatology baselines.
- The model's predicted next‑day high temperature.
- The current NWS forecast high (if available) for comparison.

## Data Sources
- **Historical Observations:** Iowa State Mesonet ASOS network ([https://mesonet.agron.iastate.edu/ASOS/](https://mesonet.agron.iastate.edu/ASOS/)).
- **Climatology:** Computed from the historical record for KMIA.
- **Live Forecast:** National Weather Service API ([api.weather.gov](https://api.weather.gov)).

## Model Details
The core model (`WeatherModel` in `weather_predictor.py`) performs the following steps:
1. Loads historical ASOS data and aggregates to daily records.
2. Engineers features: day‑of‑year (sine/cosine), lagged high/low temperatures, rolling means, previous‑day humidity, wind speed, and pressure.
3. Splits data chronologically (older → train, newer → test).
4. Fits a `GradientBoostingRegressor` to predict the target (tomorrow's high).
5. Provides a `predict_next_day()` method using the most recent observed day.

## License
MIT