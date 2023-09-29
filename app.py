from flask import Flask, jsonify, request
from prophet import Prophet
import pandas as pd
import pickle
from datetime import datetime, timedelta

app = Flask(__name__)

# Load the model
with open('prophet_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
    elif request.method == 'GET':
        # You can handle the GET request here if needed
        pass
    else:
        return jsonify({"error": "Invalid request method"}), 405
    
    unique_dates = pd.date_range(start=datetime.now(), periods=7)
    time_intervals = ['morning', 'forenoon', 'afternoon', 'evening', 'night']
    predictions = []

    for date in unique_dates:
        for time_interval in time_intervals:
            time_data = [entry for entry in data if entry['time'] == time_interval]
            if not time_data:
                continue
            start_time = pd.to_datetime(time_data[0]['start_time'], format='%I:%M %p')
            end_time = pd.to_datetime(time_data[0]['end_time'], format='%I:%M %p')
            future_start = date.replace(hour=start_time.hour, minute=start_time.minute)
            future_end = date.replace(hour=end_time.hour, minute=end_time.minute)

            future = pd.DataFrame({'ds': [future_start, future_end]})
            availability = 1 if time_data[0]['availability'] == 'available' else 0
            future['availability'] = availability

            forecast = model.predict(future)
            availability = 1 if forecast['yhat'].mean() > 0.5 else 0

            result = {
                "date": date.date(),
                "time_interval": time_interval,
                "availability": "Available" if availability == 1 else "Unavailable"
            }

            predictions.append(result)

    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
