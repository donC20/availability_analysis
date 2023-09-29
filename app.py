from flask import Flask, jsonify, request
from prophet import Prophet
import pandas as pd
from datetime import datetime, timedelta

app = Flask(__name__)

# Load the model
model = Prophet()
df = pd.read_csv('availability_dataset_one_month.csv') # Adjust the path
df['ds'] = pd.to_datetime(df['date'] + ' ' + df['start_time'])
df['y'] = (df['availability'] == 'available').astype(int)
model.fit(df[['ds', 'y']])

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        date = data['date']
    elif request.method == 'GET':
        date = request.args.get('date')
    else:
        return jsonify({"error": "Invalid request method"}), 405

    # Convert date to datetime object
    date = datetime.strptime(date, '%Y-%m-%d')

    # Define time intervals
    time_intervals = ['morning', 'forenoon', 'afternoon', 'evening', 'night']

    # Generate a list of unique dates for the next 7 days
    next_7_days = [date + timedelta(days=i) for i in range(7)]

    # Initialize an empty list to store predictions
    predictions = []

    for next_date in next_7_days:
        for time_interval in time_intervals:
            time_df = df[df['time'] == time_interval]
            start_time = pd.to_datetime(time_df['start_time'].iloc[0], format='%I:%M %p')
            end_time = pd.to_datetime(time_df['end_time'].iloc[0], format='%I:%M %p')
            future_start = next_date.replace(hour=start_time.hour, minute=start_time.minute)
            future_end = next_date.replace(hour=end_time.hour, minute=end_time.minute)
            future = pd.DataFrame({'ds': [future_start, future_end]})
            forecast = model.predict(future)
            availability = 1 if (forecast['yhat'].values[0] + forecast['yhat'].values[1]) / 2 > 0.5 else 0

            result = {
                "date": next_date.date(),
                "time_interval": time_interval,
                "availability": "Available" if availability == 1 else "Unavailable"
            }

            predictions.append(result)

    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
