import pandas as pd
import random
import joblib
import numpy as np

# Load datasets
crop_df = pd.read_csv("data/1.csv")
rules_df = pd.read_csv("data/soil_diagnostic_rules.csv")

# Load model, label encoder, and class names
model = joblib.load("model/model.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")
class_names = joblib.load("model/class_names.pkl")

# Fertilizer nutrient values (simplified)
fertilizer_nutrients = {
    "Urea": {"N": 46, "P": 0, "K": 0},
    "DAP": {"N": 18, "P": 46, "K": 0},
    "MOP": {"N": 0, "P": 0, "K": 60},
    "Compost": {"N": 2, "P": 1, "K": 1},
    "Vermicompost": {"N": 3, "P": 1.5, "K": 2},
    "Biofertilizer": {"N": 10, "P": 5, "K": 5}
}

# 🧪 Suggest improved fertilizer based on nutrient deficiency
def suggest_better_fertilizer(current, missing_nutrient):
    for fert, nutrients in fertilizer_nutrients.items():
        if fert != current and nutrients.get(missing_nutrient, 0) > fertilizer_nutrients[current].get(missing_nutrient, 0):
            return fert
    return "Organic Compost"

# ⚙️ Get predicted crop and confidence
def get_model_prediction(ph):
    feature_template = {
        'Soil_pH': ph,
        'Nitrogen_Content': crop_df['Nitrogen_Content'].mean(),
        'Phosphorus_Content': crop_df['Phosphorus_Content'].mean(),
        'Potassium_Content': crop_df['Potassium_Content'].mean(),
        'Sulfur_Content': crop_df['Sulfur_Content'].mean(),
        'Zinc_Content': crop_df['Zinc_Content'].mean(),
        'Specific_Humidity_Summer': crop_df['Specific_Humidity_Summer'].mean(),
        'Max_Temperature_Summer': crop_df['Max_Temperature_Summer'].mean(),
        'Total_Precipitation_Summer': crop_df['Total_Precipitation_Summer'].mean()
    }

    X_input = pd.DataFrame([feature_template])
    probs = model.predict_proba(X_input)[0]
    top_index = np.argmax(probs)
    confidence = round(probs[top_index] * 100, 2)
    predicted_crop_encoded = top_index
    predicted_crop = label_encoder.inverse_transform([predicted_crop_encoded])[0]
    return predicted_crop, confidence

# 🧠 Nutrient problem detection using actual input values
def diagnose_problem(crop, ph, fertilizer, actual_nutrients):
    # Clean and load rules
    rules_df = pd.read_csv("data/soil_diagnostic_rules.csv")
    rules_df.columns = [col.strip().replace('"', '') for col in rules_df.columns]
    rules_df['Parameter'] = rules_df['Parameter'].str.strip()

    issues = []

    for nutrient, value in actual_nutrients.items():
        nutrient = nutrient.strip().capitalize()

        row = rules_df[rules_df['Parameter'].str.lower() == nutrient.lower()]
        if not row.empty:
            min_val = float(str(row.iloc[0]['Min_Threshold']).split()[0])
            max_val = float(str(row.iloc[0]['Max_Threshold']).split()[0])

            deficiency_problem = row.iloc[0]['Deficiency Problem']
            excess_problem = row.iloc[0]['Excess Problem']
            deficiency_solution = row.iloc[0]['Deficiency Solution']
            excess_solution = row.iloc[0]['Excess Solution']

            if value < min_val:
                issues.append({
                    "nutrient": nutrient,
                    "actual": value,
                    "problem": deficiency_problem,
                    "solution": deficiency_solution,
                    "status": "Deficiency",
                    "better_fertilizer": suggest_better_fertilizer(fertilizer, nutrient[0])
                })
            elif value > max_val:
                issues.append({
                    "nutrient": nutrient,
                    "actual": value,
                    "problem": excess_problem,
                    "solution": excess_solution,
                    "status": "Excess",
                    "better_fertilizer": "Reduce or rebalance inputs"
                })

    return issues

# 🔍 Main function used in app.py
def diagnose_problem_and_fertilizer(crop, ph, fertilizer, actual_nutrients):
    issues = diagnose_problem(crop, ph, fertilizer, actual_nutrients)
    predicted_crop, confidence = get_model_prediction(ph)

    valid_crop_types = crop_df["Crop_Type"].dropna().unique().tolist()
    next_crop = random.choice(valid_crop_types) if valid_crop_types else "No suitable crop found"

    return {
        "problems_detected": bool(issues),
        "issues": issues,  # list of all issues
        "current_fertilizer": fertilizer,
        "recommended_fertilizer": issues[0]["better_fertilizer"] if issues else "Balanced Organic Mix",
        "next_crop": next_crop,
        "model_prediction": predicted_crop,
        "confidence": f"{confidence}%"
    }
