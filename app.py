from flask import Flask, render_template, request
import pandas as pd
import joblib
from datetime import datetime
import calendar

from utils.crop_recommend import get_crop_recommendation
from utils.fertilizer_recommend import (
    suggest_alternate_fertilizer,
    generate_fertilizer_recommendation,
    adjust_fertilizer_dosage,
    find_best_fertilizer
)
from utils.weather import get_weather
from utils.problem_diagnosis import diagnose_problem_and_fertilizer
from utils.water_requirement import get_coordinates, get_hourly_weather, estimate_water, plot_water_requirement

app = Flask(__name__)
csv_path = 'data/1.csv'
df = pd.read_csv(csv_path)
WEATHER_API_KEY = 'dc627238a1d14d7aa84142427250204'

# 🔁 Load ML model & label encoder using joblib
model = joblib.load("model/model.pkl")
le = joblib.load("model/label_encoder.pkl")

@app.route('/', methods=['GET', 'POST'])
def index():
    result = {}
    if request.method == 'POST':
        ph = float(request.form['ph'])
        location = request.form['location']
        current_crop = request.form['current_crop']
        weather = get_weather(location, WEATHER_API_KEY)
        result = get_crop_recommendation(ph, location, current_crop, df, weather)
        result['weather'] = weather
    return render_template('index.html', result=result)

@app.route('/get_crop_recommendation', methods=['POST'])
def get_crop_recommendation_route():
    try:
        ph = float(request.form['ph'])
        location = request.form['location']
        current_crop = request.form['current_crop']

        weather = get_weather(location, WEATHER_API_KEY)
        result = get_crop_recommendation(ph, location, current_crop, df, weather)
        result['weather'] = weather

        return render_template("index.html", result=result)
    except Exception as e:
        return render_template("index.html", error=f"Error in crop recommendation: {e}")

@app.route('/problem-finder', methods=['POST'])
def problem_finder():
    crop = request.form.get('current_crop')
    location = request.form.get('location')
    ph_raw = request.form.get('ph')
    fertilizer = request.form.get('fertilizer')

    try:
        ph = float(ph_raw)
    except (TypeError, ValueError):
        ph = 7.0  # Default neutral pH

    # 🧪 Read user-input nutrient values
    try:
        nitrogen = float(request.form.get("nitrogen", 0))
        phosphorus = float(request.form.get("phosphorus", 0))
        potassium = float(request.form.get("potassium", 0))
        sulfur = float(request.form.get("sulfur", 0))
    except (TypeError, ValueError):
        nitrogen = phosphorus = potassium = sulfur = 0.0

    actual_nutrients = {
        "N": nitrogen,
        "P": phosphorus,
        "K": potassium,
        "S": sulfur
    }

    if not crop:
        return render_template('index.html', fertilizer_result={
            "error": "Crop name is missing. Please select or enter a valid crop."
        })

    if crop not in le.classes_:
        return render_template('index.html', fertilizer_result={
            "error": f"Unknown crop: '{crop}'. It was not seen during training. Try a different one."
        })

    # ✅ Perform actual diagnosis
    diagnosis = diagnose_problem_and_fertilizer(crop, ph, fertilizer, actual_nutrients)

    # 📊 Example nutrient data for dosage adjustment
    fertilizer_data = {
        "Nitrogen (N)": 8,
        "Phosphorus (P)": 5,
        "Potassium (K)": 12
    }

    soil_data = {
        "Nitrogen_Content": nitrogen,
        "Phosphorus_Content": phosphorus,
        "Potassium_Content": potassium
    }

    dosage_recommendation = generate_fertilizer_recommendation(fertilizer_data, soil_data, thresholds={
        "Nitrogen (N)": (5, 10, 15),
        "Phosphorus (P)": (3, 6, 10),
        "Potassium (K)": (10, 15, 18)
    })

    adjusted_dosage = adjust_fertilizer_dosage(fertilizer_data, soil_data, thresholds={
        "Nitrogen (N)": (5, 10, 15),
        "Phosphorus (P)": (3, 6, 10),
        "Potassium (K)": (10, 15, 18)
    })

    # 🌿 Fertilizer recommendation from CSV
    best_fertilizer_info = find_best_fertilizer(soil_data)
    alt_fertilizer = suggest_alternate_fertilizer(crop, location, ph, fertilizer)

    web_result = {
        "recommended_fertilizer": alt_fertilizer["recommended_fertilizer"],
        "next_crop": alt_fertilizer["next_crop"],
        "best_fertilizer_info": best_fertilizer_info
    }

    return render_template('index.html', fertilizer_result={
        "diagnosis": diagnosis,
        "dosage_recommendation": dosage_recommendation,
        "adjusted_dosage": adjusted_dosage,
        "web_result": web_result
    })

@app.route('/water-requirement', methods=["POST"])
def water_requirement():
    location = request.form.get("location")
    crop = request.form.get("crop")

    if not location or not crop:
        return render_template("index.html", error="Please enter both location and crop.")

    lat, lon = get_coordinates(location)
    if lat is None:
        return render_template("index.html", error="Location not found.")

    hourly_data = get_hourly_weather(lat, lon)
    if not hourly_data:
        return render_template("index.html", error="Could not fetch weather data.")

    month = calendar.month_name[datetime.today().month]
    timestamps, water_required, stage, kc = estimate_water(hourly_data, crop, month)
    plot_water_requirement(timestamps, water_required, location, crop, stage, kc)

    return render_template("index.html", water_plot="water_need_plot.png", location=location, crop=crop.capitalize(), stage=stage, kc=kc)

if __name__ == '__main__':
    app.run(debug=True)


