from prophet import Prophet
import pandas as pd
from prophet.plot import plot_plotly, plot_components_plotly
import logging
import warnings
import pickle
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
logging.getLogger('prophet').setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# 

# from google.colab import drive
# drive.mount('/content/drive/')

df = pd.read_csv('availability_dataset_one_month.csv')
df.dropna(inplace= True)
df.reset_index(drop=True, inplace=True)

df.info()

df.head()

# Assuming df is your DataFrame with the relevant columns

# Data Wrangling
df['ds'] = pd.to_datetime(df['date'] + ' ' + df['start_time'])
df['y'] = (df['availability'] == 'available').astype(int)

# Train the Prophet Model
model = Prophet()
model.fit(df[['ds', 'y']])

# Generate a list of unique dates for the next 7 days
unique_dates = pd.date_range(start=df['ds'].max() + pd.DateOffset(days=1), periods=7)

# Generate a list of unique time intervals
time_intervals = df['time'].unique()

# Generate predictions for each combination of date and time interval
for date in unique_dates:
    for time_interval in time_intervals:
        time_df = df[df['time'] == time_interval]
        start_time = pd.to_datetime(time_df['start_time'].iloc[0], format='%I:%M %p')
        end_time = pd.to_datetime(time_df['end_time'].iloc[0], format='%I:%M %p')
        future_start = date.replace(hour=start_time.hour, minute=start_time.minute)
        future_end = date.replace(hour=end_time.hour, minute=end_time.minute)
        future = pd.DataFrame({'ds': [future_start, future_end]})
        forecast = model.predict(future)
        availability = 1 if (forecast['yhat'].values[0] + forecast['yhat'].values[1]) / 2 > 0.5 else 0
        print(f"On  {date.date()}, Time Interval: {time_interval}, Availability: {'Available' if availability == 1 else 'Unavailable'}")
# plot_plotly(model ,forecast)
# plot_components_plotly(model, forecast)