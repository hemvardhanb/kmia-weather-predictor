import requests
import pandas as pd
import io

def download_miami_weather_data():
    # Fetching tmpf (temp), dwpf (dew point), sknt (wind speed), drct (wind direction)
    # This gives us a much richer dataset for ML prediction
    url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py/?station=KMIA&data=tmpf&data=dwpf&data=sknt&data=drct&year1=2020&month1=1&day1=1&year2=2025&month2=12&day2=31&tz=America%2FNew_York&format=comma&direct=yes"
    
    response = requests.get(url)
    if response.status_code == 200:
        # Filter out comment lines starting with #
        lines = [line for line in response.text.split('\n') if not line.startswith('#')]
        df = pd.read_csv(io.StringIO('\n'.join(lines)))
        df.to_csv("kmia_extended.csv", index=False)
        print("Data saved to kmia_extended.csv")
    else:
        print(f"Failed to fetch data: {response.status_code}")

if __name__ == "__main__":
    download_miami_weather_data()
