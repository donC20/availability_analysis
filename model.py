from prophet import Prophet
import pandas as pd
from prophet.plot import plot_plotly, plot_components_plotly
import logging
import warnings
import pickle
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

logging.getLogger('prophet').setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

df = pd.read_csv('availability_dataset_one_week.csv')
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# Data Wrangling
df['ds'] = pd.to_datetime(df['date'] + ' ' + df['start_time'])
df['y'] = (df['availability'] == 'available').astype(int)
df['day_of_week'] = df['ds'].dt.day_name()

# Train the Prophet Model
model = Prophet()
model.add_seasonality(name='custom', period=7, fourier_order=3)  # Add a custom weekly seasonality
model.fit(df[['ds', 'day_of_week', 'start_time', 'end_time', 'time', 'y']])

# Generate a list of unique dates for the next 7 days
unique_dates = pd.date_range(start=datetime.now(), periods=7)

# Generate a list of unique time intervals
time_intervals = df['time'].unique()

# Generate predictions for each combination of date, time interval, and day of the week
for date in unique_dates:
    print(f"Predictions for {date.date()}:")
    for time_interval in time_intervals:
        for day_of_week in df['day_of_week'].unique():
            time_df = df[(df['time'] == time_interval) & (df['day_of_week'] == day_of_week)]
            start_time = pd.to_datetime(time_df['start_time'].iloc[0], format='%I:%M %p')
            end_time = pd.to_datetime(time_df['end_time'].iloc[0], format='%I:%M %p')
            future_start = date.replace(hour=start_time.hour, minute=start_time.minute)
            future_end = date.replace(hour=end_time.hour, minute=end_time.minute)
            availability= (df['availability'] == 'available').astype(int)
            future = pd.DataFrame({'ds': [future_start, future_end]})
            future['day_of_week'] = day_of_week
            future['start_time'] = start_time
            future['end_time'] = end_time
            future['time'] = time_interval
            future['availability'] = availability

            forecast = model.predict(future)
            availability = 1 if (forecast['yhat'].values[0] + forecast['yhat'].values[1]) / 2 > 0.5 else 0
            print(f"   - Time Interval: {time_interval}, Day of the Week: {day_of_week}, Availability: {'Available' if availability == 1 else 'Unavailable'}")
    print()  # Add an empty line for better separation between days
# with open('prophet_model.pkl', 'wb') as f:
#     pickle.dump(model, f)