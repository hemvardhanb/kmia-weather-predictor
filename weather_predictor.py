"""
Next-Day High Temperature Model for Miami Intl Airport (KMIA)
==============================================================
Fixes the core flaw of the original MVP: that version trained on ~7 rows
pulled from the *current* NWS forecast (a forecast fit to a forecast has
no new information to learn from). This version:
1. Pulls YEARS of real historical observations from the Iowa Environmental
   Mesonet ASOS archive (free, no key required).
2. Builds real predictive features: cyclic day-of-year, lag temps, rolling
   averages, humidity, pressure, wind.
3. Uses a time-ordered train/test split (never randomly shuffled -- that
   would leak future information into training).
4. Reports MAE against two baselines (persistence and climatology) so you
   can tell whether the model is actually adding skill.
5. Uses the live NWS forecast only as an extra input feature / sanity check,
   not as the training target.

Requires: pandas, requests, scikit-learn
"""

import io
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

STATION = "MIA"  # Iowa Mesonet ASOS station id for Miami Intl
NETWORK = "FL_ASOS"
LAT, LON = 25.7959, -80.2870
HEADERS = {"User-Agent": "WeatherModelMIA/2.0 (contact@example.com)"}
CACHE_FILE = "mia_asos_daily.csv"


def fetch_historical_asos(years_back: int = 8) -> pd.DataFrame:
    """Download hourly ASOS observations and aggregate to daily records."""
    end = datetime.utcnow()
    start = end - timedelta(days=365 * years_back)
    url = (
        "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
        f"?station={STATION}&network={NETWORK}"
        "&data=tmpf,dwpf,relh,sped,mslp,p01i"
        f"&year1={start.year}&month1={start.month}&day1={start.day}"
        f"&year2={end.year}&month2={end.month}&day2={end.day}"
        "&tz=Etc/UTC&format=onlycomma&latlon=no&missing=empty&trace=empty"
    )
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()

    hourly = pd.read_csv(io.StringIO(resp.text), na_values=["M", ""])
    for col in ["tmpf", "dwpf", "relh", "sped", "mslp", "p01i"]:
        hourly[col] = pd.to_numeric(hourly[col], errors="coerce")

    hourly["date"] = pd.to_datetime(hourly["valid"]).dt.date
    daily = (
        hourly.groupby("date")
        .agg(
            temp_high=("tmpf", "max"),
            temp_low=("tmpf", "min"),
            humidity_mean=("relh", "mean"),
            pressure_mean=("mslp", "mean"),
            wind_mean=("sped", "mean"),
            precip_total=("p01i", "sum"),
        )
        .reset_index()
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily = (
        daily.dropna(subset=["temp_high"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    daily.to_csv(CACHE_FILE, index=False)
    return daily


def load_historical(years_back: int = 8, use_cache: bool = True) -> pd.DataFrame:
    if use_cache:
        try:
            df = pd.read_csv(CACHE_FILE, parse_dates=["date"])
            print(f"Loaded {len(df)} cached daily records.")
            return df
        except FileNotFoundError:
            pass

    print("Downloading historical ASOS data (this can take a minute)...")
    df = fetch_historical_asos(years_back)
    print(f"Downloaded {len(df)} daily records.")
    return df


def engineer_features(df: pd.DataFrame):
    """Build predictive features and the next-day-high target."""
    df = df.sort_values("date").reset_index(drop=True)
    doy = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # Lag / rolling features -- only use info available *before* the
    # target day to avoid leakage.
    df["lag1_high"] = df["temp_high"].shift(1)
    df["lag1_low"] = df["temp_low"].shift(1)
    df["roll3_high"] = df["temp_high"].shift(1).rolling(3).mean()
    df["roll7_high"] = df["temp_high"].shift(1).rolling(7).mean()
    df["lag1_humidity"] = df["humidity_mean"].shift(1)
    df["lag1_pressure"] = df["pressure_mean"].shift(1)
    df["lag1_wind"] = df["wind_mean"].shift(1)
    df["lag1_precip"] = df["precip_total"].shift(1)

    # Target: tomorrow's high, i.e. predict day t's high using data
    # available at the end of day t-1.
    df["target_high"] = df["temp_high"]

    feature_cols = [
        "doy_sin",
        "doy_cos",
        "lag1_high",
        "lag1_low",
        "roll3_high",
        "roll7_high",
        "lag1_humidity",
        "lag1_pressure",
        "lag1_wind",
        "lag1_precip",
    ]
    df = df.dropna(subset=feature_cols + ["target_high"]).reset_index(
        drop=True
    )
    return df, feature_cols


def time_split(df: pd.DataFrame, test_days: int = 180):
    cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    train = df[df["date"] <= cutoff]
    test = df[df["date"] > cutoff]
    return train, test


def evaluate_baselines(test: pd.DataFrame, train: pd.DataFrame):
    # Persistence baseline: tomorrow = today's high (i.e. lag1_high).
    persistence_mae = mean_absolute_error(
        test["target_high"], test["lag1_high"]
    )

    # Climatology baseline: mean historical high for that day-of-year.
    train_clim = train.copy()
    train_clim["doy"] = train_clim["date"].dt.dayofyear
    clim_lookup = train_clim.groupby("doy")["temp_high"].mean()
    test_doy = test["date"].dt.dayofyear
    clim_pred = test_doy.map(clim_lookup).fillna(clim_lookup.mean())
    climatology_mae = mean_absolute_error(test["target_high"], clim_pred)

    return persistence_mae, climatology_mae


class WeatherModel:

    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42
        )
        self.feature_cols = None
        self._last_row = None

    def train(
        self, years_back: int = 8, test_days: int = 180, use_cache: bool = True
    ):
        raw = load_historical(years_back, use_cache)
        df, self.feature_cols = engineer_features(raw)
        train, test = time_split(df, test_days)

        self.model.fit(train[self.feature_cols], train["target_high"])
        preds = self.model.predict(test[self.feature_cols])
        model_mae = mean_absolute_error(test["target_high"], preds)

        persistence_mae, climatology_mae = evaluate_baselines(test, train)

        print(
            "\n--- Backtest results (held-out last "
            f"{test_days} days) ---"
        )
        print(f"Model (GBM) MAE: {model_mae:.2f} F")
        print(f"Persistence MAE: {persistence_mae:.2f} F")
        print(f"Climatology MAE: {climatology_mae:.2f} F")

        self._last_row = df.iloc[-1]
        return model_mae, persistence_mae, climatology_mae

    def predict_next_day(self) -> float:
        """Predict tomorrow's high using the most recent available day."""
        row = self._last_row
        X = pd.DataFrame([{c: row[c] for c in self.feature_cols}])
        return float(self.model.predict(X)[0])

    def get_nws_forecast_high(self) -> float | None:
        """Pull today's NWS forecast high as an independent sanity check."""
        try:
            points = requests.get(
                f"https://api.weather.gov/points/{LAT},{LON}",
                headers=HEADERS,
                timeout=15,
            ).json()
            forecast = requests.get(
                points["properties"]["forecast"], headers=HEADERS, timeout=15
            ).json()
            for p in forecast["properties"]["periods"]:
                if p["isDaytime"]:
                    return float(p["temperature"])
        except Exception as e:
            print(f"NWS forecast unavailable: {e}")
        return None


if __name__ == "__main__":
    wm = WeatherModel()
    wm.train(years_back=8, test_days=180, use_cache=True)
    prediction = wm.predict_next_day()
    nws = wm.get_nws_forecast_high()

    print(f"\nModel's predicted next-day high: {prediction:.1f} F")
    if nws is not None:
        print(f"NWS official forecast high: {nws:.1f} F")
