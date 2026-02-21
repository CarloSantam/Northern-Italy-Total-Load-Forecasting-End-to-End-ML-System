import boto3
import cdsapi
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime
from joblib import Parallel, delayed

from datetime import timedelta

import os

api=os.getenv("COPERNICUS_KEY ")

# --- cities ---
cities = [
    {"citta": "Torino",   "lat": 45.0703, "lon": 7.6869},
    {"citta": "Milano",   "lat": 45.4642, "lon": 9.1900},
    {"citta": "Venezia",  "lat": 45.4408, "lon": 12.3155},
    {"citta": "Genova",   "lat": 44.4056, "lon": 8.9463},
    {"citta": "Bologna",  "lat": 44.4949, "lon": 11.3426},
    {"citta": "Trieste",  "lat": 45.6495, "lon": 13.7768},
    {"citta": "Aosta",    "lat": 45.7370, "lon": 7.3201},
    {"citta": "Trento",   "lat": 46.0748, "lon": 11.1217},
    {"citta": "Bolzano",  "lat": 46.4983, "lon": 11.3548},
]

DATASET = "cams-global-atmospheric-composition-forecasts"
OUTDIR = Path("Data/Weather Data")
OUTDIR.mkdir(parents=True, exist_ok=True)

HALF = 0.25
def area(lat, lon):
    # ADS/CDS expects: [North, West, South, East]
    return [lat + HALF, lon - HALF, lat - HALF, lon + HALF]


def grib_to_df(grib_path: Path, city_name: str) -> pd.DataFrame:
    # Read GRIB via cfgrib engine
    ds = xr.open_dataset(grib_path, engine="cfgrib")

    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"

    lat0 = float(ds[lat_name].mean())
    lon0 = float(ds[lon_name].mean())
    p = ds.sel({lat_name: lat0, lon_name: lon0}, method="nearest")

    df = p.to_dataframe().reset_index()

    # Build valid_time if missing
    if "valid_time" not in df.columns:
        if "time" in df.columns and "step" in df.columns:
            df["valid_time"] = pd.to_datetime(df["time"]) + pd.to_timedelta(df["step"])

    df["run_time"] = pd.to_datetime(df["time"]) if "time" in df.columns else pd.NaT
    df["citta"] = city_name

    rename_map = {
        "t2m": "temp_k",
        "d2m": "dewpoint_k",
        "sp": "surface_pressure_pa",
        "u10": "u10",
        "v10": "v10",
    }
    for k, v in rename_map.items():
        if k in df.columns:
            df = df.rename(columns={k: v})

    long_map = {
        "2m_temperature": "temp_k",
        "2m_dewpoint_temperature": "dewpoint_k",
        "surface_pressure": "surface_pressure_pa",
        "10m_u_component_of_wind": "u10",
        "10m_v_component_of_wind": "v10",
    }
    for k, v in long_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})

    # Wind speed and direction
    if "u10" in df.columns and "v10" in df.columns:
        df["wind_speed"] = np.sqrt(df["u10"]**2 + df["v10"]**2)
        df["wind_deg"] = (np.degrees(np.arctan2(-df["u10"], -df["v10"])) + 360) % 360

    # Relative humidity estimate from T and Td (Magnus)
    if "temp_k" in df.columns and "dewpoint_k" in df.columns:
        t_c = df["temp_k"] - 273.15
        td_c = df["dewpoint_k"] - 273.15
        a, b = 17.625, 243.04
        es = np.exp((a * t_c) / (b + t_c))
        e = np.exp((a * td_c) / (b + td_c))
        df["humidity_rh"] = (100.0 * (e / es)).clip(0, 100)

    return df


def build_windows(start: str, end: str, step_days: int = 14):
    # Build inclusive windows [t, min(t+step, end)]
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    ticks = pd.date_range(start=start_ts, end=end_ts, freq=f"{step_days}D")
    if ticks[-1] != end_ts:
        ticks = ticks.append(pd.DatetimeIndex([end_ts]))

    windows = []
    for t0 in ticks:
        t1 = (t0 + pd.Timedelta(days=step_days)).normalize()
        if t1 > end_ts:
            t1 = end_ts
        windows.append((t0, t1))
    return windows


def process_task(city: dict, window: tuple[pd.Timestamp, pd.Timestamp]) -> pd.DataFrame | None:
    # Create a fresh client per task (safer for threading)
    client = cdsapi.Client(
        url="https://ads.atmosphere.copernicus.eu/api",
        key=f"{api}"
    )

    t0, t1 = window
    date_range = f"{t0:%Y-%m-%d}/{t1:%Y-%m-%d}"
    target = OUTDIR / f"{city['citta']}_{t0:%Y-%m-%d}_{t1:%Y-%m-%d}.grib"

    request = {
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_dewpoint_temperature",
            "2m_temperature",
            "surface_pressure",
        ],
        "date": [date_range],
        "time": ["00:00"],
        "leadtime_hour": [str(h) for h in range(24, 120)],
        "type": ["forecast"],
        "data_format": "grib",
        "area": area(city["lat"], city["lon"]),
    }

    try:
        # Download only if missing
        if not target.exists():
            client.retrieve(DATASET, request, str(target))

        # Parse GRIB -> DataFrame
        df_part = grib_to_df(target, city["citta"])
        return df_part

    except Exception as e:
        # Return None on failure so the pipeline continues
        print(f"Task failed for {city['citta']} {date_range}: {e}")
        return None
    
access_key=os.getenv("AWS_ACCESS_KEY_ID")

aws_s3_key=os.getenv("AWS_SECRET_ACCESS_KEY")

bucket_name='loadforecastingdata'

s3 = boto3.client("s3",aws_access_key_id=access_key,
    aws_secret_access_key=aws_s3_key,
    region_name="eu-west-1")

s3.download_file(bucket_name,"Data/Weather_data_.csv","Data/Weather_data_.csv")

df_weather_old=pd.read_csv("Data/Weather_data_.csv")[['citta', 'valid_time',
       'time', 'u10', 'v10', 'dewpoint_k', 'temp_k', 'surface_pressure_pa',
       'wind_speed', 'wind_deg', 'humidity_rh']]

start = "2023-01-01"

if df_weather_old.empty==False:
    start=(pd.to_datetime(max(df_weather_old['time']))+timedelta(hours=0)+timedelta(days=1)).strftime('%Y-%m-%d')

# ---- run ----
end = (datetime.now()+timedelta(days=1)).strftime("%Y-%m-%d")
windows = build_windows(start, end, step_days=3)

tasks = [(city, w) for city in cities for w in windows]

results = Parallel(n_jobs=1, backend="threading", verbose=10)(
    delayed(process_task)(city, w) for city, w in tasks
)

dfs = [r for r in results if r is not None]
if dfs!=[]:
    df_weather = pd.concat(dfs, ignore_index=True)
else:
    df_weather=pd.DataFrame(columns=["citta", "run_time", "valid_time",'time', 'u10', 'v10', 'dewpoint_k', 'temp_k', 'surface_pressure_pa', 'wind_speed', 'wind_deg', 'humidity_rh'])

cols_front = ["citta", "run_time", "valid_time"]
other_cols = [c for c in df_weather.columns if c not in cols_front]
df_weather = df_weather[cols_front + other_cols].sort_values(["citta", "run_time", "valid_time"])

print(df_weather.head())

df_weather_filtred=df_weather[['citta', 'valid_time', 'time','u10', 'v10', 'dewpoint_k', 'temp_k',
'surface_pressure_pa', 'wind_speed', 'wind_deg', 'humidity_rh']]


to_export=pd.concat([df_weather_old,df_weather_filtred.drop_duplicates()])

to_export['valid_time']=pd.to_datetime(to_export['valid_time'])

to_export=to_export.loc[pd.to_datetime(to_export['valid_time'],format='mixed')>=pd.to_datetime(to_export['time'],format='mixed')+timedelta(days=1)]

to_export['time']=pd.to_datetime(to_export['time'],format='mixed')

to_export['time_max']=to_export.groupby(['citta','valid_time'])['time'].transform(max)

to_export=to_export.loc[to_export['time']==to_export['time_max']]

to_export.to_csv(
    f"s3://{bucket_name}/Data/Weather_data_.csv",
    index=False,  # # Avoid writing index column
    storage_options={
        "key": access_key,
        "secret": aws_s3_key,
        "client_kwargs": {
            "region_name": "eu-west-1"
        }
    }
)

to_export=to_export.drop_duplicates(subset=["citta", "valid_time"],keep="first")

to_export_qrth=pd.DataFrame()

città=list(to_export['citta'].unique())

numeric_columns=to_export.select_dtypes(np.number).columns

non_numeric_columns=to_export.select_dtypes('object').columns


for citta in città:
    to_export_filtred=to_export.loc[to_export['citta']==citta]
        
    to_export_filtred=to_export_filtred.set_index('valid_time')
    
    to_export_filtred=to_export_filtred.drop(columns={'time'}).drop_duplicates().resample('15T').asfreq()
    
    
    to_export_filtred[numeric_columns]=to_export_filtred[numeric_columns].interpolate(method='cubic')
    
    to_export_filtred[non_numeric_columns]=to_export_filtred[non_numeric_columns].fillna(method='ffill')
    
    to_export_qrth=pd.concat([to_export_filtred,to_export_qrth])




to_export_qrth=to_export_qrth.reset_index().rename(columns={'valid_time':'Date'})

to_export_qrth=to_export_qrth.pivot_table(columns=['citta'],index=['Date'],values=['u10', 'v10', 'dewpoint_k', 'temp_k',
       'surface_pressure_pa', 'wind_speed', 'wind_deg', 'humidity_rh'],aggfunc='mean').reset_index()

to_export_qrth.columns=[' '.join(col) for col in to_export_qrth.columns]

to_export_qrth=to_export_qrth.rename(columns={'Date ':'Date'})

to_export_qrth.to_csv(
    f"s3://{bucket_name}/Data/Weather_data_qrth.csv",
    # index=False,  # # Avoid writing index column
    storage_options={
        "key": access_key,
        "secret": aws_s3_key,
        "client_kwargs": {
            "region_name": "eu-west-1"
        }
    }
)

################ download loads ##################

from entsoe import EntsoePandasClient
import pandas as pd
from datetime import datetime,timedelta

# your API key here
API_KEY = os.getenv("ENTSOE_API_KEY ")

# Create client using your API key
client = EntsoePandasClient(api_key=API_KEY)

start_desired = pd.Timestamp("20210101", tz="Europe/Rome")

end_desired = pd.Timestamp.now(tz="Europe/Rome")+timedelta(days=5)

country_code='IT_NORD'

load_forecast = client.query_load_forecast(country_code, start=start_desired, end=end_desired).reset_index().rename(columns={'index':'Date'})

# # Actual / realised load
load_actual = client.query_load(country_code, start=start_desired, end=end_desired).reset_index().rename(columns={'index':'Date'})

# Wind Onshore
wind_on = client.query_generation(
    country_code,
    start=start_desired,
    end=end_desired,
    psr_type="B19"
).reset_index().rename(columns={'index':'Date'})

# Solar
solar = client.query_generation(
    country_code,
    start=start_desired,
    end=end_desired,
    psr_type="B16"
).reset_index().rename(columns={'index':'Date'})

res_forecast = client.query_wind_and_solar_forecast(
    country_code,
    start=start_desired,
    end=end_desired
).reset_index().rename(columns={'index':'Date'})

res_forecast=res_forecast.rename(columns={'Wind Onshore':'Wind Onshore_forecast'})

res_forecast=res_forecast.rename(columns={'Solar':'Solar_forecast'})

forecast=load_forecast.merge(res_forecast,on=['Date'],how='left')

actual=load_actual.merge(wind_on.merge(solar,on=['Date'],how='left'),on=['Date'],how='left')

load_forecast_actual=actual.merge(forecast,how='outer',on=['Date'])

load_forecast_actual.to_csv(
    f"s3://{bucket_name}/Data/Load.csv",
    index=False,  # # Avoid writing index column
    storage_options={
        "key": access_key,
        "secret": aws_s3_key,
        "client_kwargs": {
            "region_name": "eu-west-1"
        }
    }
)

to_export_qrth['Date']=pd.to_datetime(to_export_qrth['Date'])

load_forecast_actual['Date']=pd.to_datetime(load_forecast_actual['Date'].astype(str).str[:19])

Final=load_forecast_actual.merge(to_export_qrth,on=['Date'],how='right')

Final=Final.sort_values('Date')

max_d=Final.loc[Final['Actual Load'].isna()==False,'Date'].max()

Final.loc[Final['Date']<=max_d,['Actual Load', 'Wind Onshore', 'Solar', 'Solar_forecast',
       'Wind Onshore_forecast']]=Final.loc[Final['Date']<=max_d,['Actual Load', 'Wind Onshore', 'Solar', 'Solar_forecast',
              'Wind Onshore_forecast']].interpolate(method='cubic')

Final.to_csv(
    f"s3://{bucket_name}/Data/Final.csv",
    index=False,  # Avoid writing index column
    storage_options={
        "key": access_key,
        "secret": aws_s3_key,
        "client_kwargs": {
            "region_name": "eu-west-1"
        }
    }
)