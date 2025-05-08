# test_client.py

import requests

url = "http://127.0.0.1:8000/predict"
sample_data = {
    "age": 40,
    "sex": "male",
    "bmi": 29.5,
    "children": 1,
    "smoker": "no",
    "region": "northeast"
}

response = requests.post(url, json=sample_data)
print("Status Code:", response.status_code)
print("Prediction:", response.json())
