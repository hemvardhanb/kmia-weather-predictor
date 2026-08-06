# KMIA Weather Predictor

An accurate, year-round daily high temperature prediction model for Miami International Airport (KMIA). To help me bet on kalshi ;)

## Overview
This project uses a `RandomForestRegressor` trained on historical ASOS/AWOS meteorological data. Unlike basic heuristic models, this system uses anomaly-based learning, which allows it to provide stable year-round predictions by modeling deviations from long-term climatological means.

## Features
- **Anomaly-Based Learning:** Models the difference between actual temperatures and long-term monthly averages to support prediction in all seasons.
- **Multivariate Input:** Incorporates temperature, dew point (humidity), and wind speed to account for coastal meteorological effects.
- **MOS Blending:** Uses a 70/30 blending approach between climate averages and model-predicted anomalies to ensure predictions remain grounded in meteorological reality.

## Usage

### Prerequisites
- Python 3.11+
- `uv` package manager

### Setup
1. Clone the repository.
2. Install dependencies:
   ```bash
   uv venv
   # Activate your venv
   uv pip install pandas scikit-learn requests numpy
   ```

### Running the Predictor
To update data and run a prediction:
```bash
python data_collector.py
python train_accurate.py
```

## Data Sources
- **Historical Observations:** Data sourced from the [Iowa State Mesonet](https://mesonet.agron.iastate.edu/ASOS/).
- **Climatology:** Derived from historical long-term averages for the KMIA station.

## License
MIT
