import requests

def get_weather(location, api_key):
    base_url = "http://api.weatherapi.com/v1/current.json"
    params = {"key": api_key, "q": location}
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        return {
            "temperature": data['current']['temp_c'],
            "humidity": data['current']['humidity'],
            "precip": data['current']['precip_mm'],
            "condition": data['current']['condition']['text']
        }
    else:
        return {
            "temperature": 25,
            "humidity": 50,
            "precip": 0,
            "condition": "Unavailable"
        }
