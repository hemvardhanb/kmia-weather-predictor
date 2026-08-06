import requests
import pandas as pd
import io

def download_miami_weather_data():
    # Use Iowa State Mesonet (ASOS) for historical KMIA data (2020-2025)
    url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py/?station=KMIA&data=max_tmpf&year1=2020&month1=1&day1=1&year2=2025&month2=12&day2=31&tz=America%2FNew_York&format=onlycomma&latlon=no&missing=M&trace=T&direct=yes&report_type=1"
    
    response = requests.get(url)
    if response.status_code == 200:
        # Load directly into pandas
        df = pd.read_csv(io.StringIO(response.text))
        df.to_csv("kmia_historical.csv", index=False)
        print("Data saved to kmia_historical.csv")
    else:
        print(f"Failed to fetch data: {response.status_code}")

if __name__ == "__main__":
    download_miami_weather_data()
