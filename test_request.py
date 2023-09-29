import requests
import json

url = 'http://localhost:5000/predict'

data = [
    {
        "date": "2023-09-30",
        "start_time": "09:00 AM",
        "end_time": "11:00 AM",
        "time": "morning",
        "availability": "available"
    },
    # Add more data points as needed
]

response = requests.post(url, data=json.dumps(data), headers={'Content-Type': 'application/json'})
print(response.json())
