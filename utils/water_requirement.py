import requests
from datetime import datetime
import calendar
import matplotlib.pyplot as plt

crop_base_needs = {
    "wheat": 0.5,
    "rice": 1.2,
    "sugarcane": 1.0,
    "maize": 0.7,
}

crop_growth_stages = {
    "wheat": {
        "November": "initial",
        "December": "development",
        "January": "development",
        "February": "mid",
        "March": "mid",
        "April": "late"
    },
    "rice": {
        "June": "initial",
        "July": "development",
        "August": "mid",
        "September": "late"
    }
}

kc_values = {
    "initial": 0.4,
    "development": 0.7,
    "mid": 1.15,
    "late": 0.6
}

def get_coordinates(location):
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1"
    response = requests.get(url)
    data = response.json()
    if data.get("results"):
        return data["results"][0]["latitude"], data["results"][0]["longitude"]
    return None, None

def get_hourly_weather(lat, lon):
    today = datetime.today().strftime('%Y-%m-%d')
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m,precipitation&start_date={today}&end_date={today}&timezone=auto"
    )
    response = requests.get(url)
    return response.json().get("hourly", {})

def get_kc(crop, month):
    crop = crop.lower()
    stage = crop_growth_stages.get(crop, {}).get(month, "initial")
    return kc_values.get(stage, 0.6), stage

def estimate_water(hourly_data, crop, month):
    timestamps = hourly_data.get("time", [])
    temps = hourly_data.get("temperature_2m", [])
    rain = hourly_data.get("precipitation", [])

    base_need = crop_base_needs.get(crop.lower(), 0.5)
    kc, stage = get_kc(crop, month)
    adjusted_base = base_need * kc

    water_required = []
    for i in range(len(timestamps)):
        temp_factor = 1 + max(0, (temps[i] - 25)) * 0.05
        rain_factor = max(0, 1 - rain[i] / adjusted_base)
        water = adjusted_base * temp_factor * rain_factor
        water_required.append(water)

    return timestamps, water_required, stage, kc

def plot_water_requirement(timestamps, water_required, location, crop, stage, kc):
    hours = [datetime.strptime(t, "%Y-%m-%dT%H:%M") for t in timestamps]
    plt.figure(figsize=(12, 6))
    plt.plot(hours, water_required, marker='o', color='blue', linewidth=2, markersize=5)
    
    plt.title(f"{crop.capitalize()} ({stage} stage, Kc={kc}) Water Need in {location}", fontsize=14)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Water Requirement (mm)", fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("static/water_need_plot.png")
