import requests

# 🔄 Original logic (you wrote this!)
def suggest_alternate_fertilizer(crop, location, ph, fertilizer):
    better_fertilizers = {
        "Urea": "Ammonium sulfate",
        "NPK": "Compost mix",
        "DAP": "Vermicompost",
        "Compost": "Biofertilizer"
    }

    alt = better_fertilizers.get(fertilizer, "Organic compost")

    crop_rotation = {
        "Rice": "Pulses",
        "Wheat": "Soybean",
        "Maize": "Groundnut",
        "Barley": "Chickpea"
    }

    next_crop = crop_rotation.get(crop, "Legumes")

    return {
        "recommended_fertilizer": alt,
        "next_crop": next_crop
    }

# 📊 Thresholds (adjustable)
thresholds = {
    "Nitrogen (N)": (5, 10, 15),
    "Phosphorus (P)": (3, 6, 10),
    "Potassium (K)": (10, 15, 18)
}

# 🌱 Analyze fertilizer dosage effect
def generate_fertilizer_recommendation(fertilizer_data, soil_data, thresholds):
    recommendations = {}

    for nutrient in ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)"]:
        soil_value = soil_data[nutrient.replace(" (N)", "_Content").replace(" (P)", "_Content").replace(" (K)", "_Content")]
        fertilizer_value = fertilizer_data[nutrient]

        low, optimal, high = thresholds[nutrient]
        new_value = soil_value + fertilizer_value

        if new_value < low:
            recommendations[nutrient] = "Increase dosage or use a stronger fertilizer."
        elif low <= new_value <= optimal:
            recommendations[nutrient] = "Optimal level achieved. Maintain dosage."
        else:
            recommendations[nutrient] = "Reduce dosage or switch to a fertilizer with lower content."

    return recommendations

# ⚖️ Auto-adjust fertilizer to safe limits
def adjust_fertilizer_dosage(fertilizer_data, soil_data, thresholds):
    adjusted_dosage = {}

    for nutrient in ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)"]:
        soil_value = soil_data[nutrient.replace(" (N)", "_Content").replace(" (P)", "_Content").replace(" (K)", "_Content")]
        fertilizer_value = fertilizer_data[nutrient]

        low, optimal, high = thresholds[nutrient]
        max_fertilizer_addition = high - soil_value

        if max_fertilizer_addition <= 0:
            adjusted_dosage[nutrient] = 0
        else:
            adjusted_dosage[nutrient] = min(fertilizer_value, max_fertilizer_addition)

    return adjusted_dosage

# 🌐 Find best fertilizer from SerpAPI (optional but powerful)
import pandas as pd  # Add this to the top if not already imported

def find_best_fertilizer(soil_data):
    deficiencies = []
    if soil_data["Nitrogen_Content"] < 5:
        deficiencies.append("N (%)")
    if soil_data["Phosphorus_Content"] < 3:
        deficiencies.append("P2O5 (%)")
    if soil_data["Potassium_Content"] < 10:
        deficiencies.append("K2O (%)")

    if not deficiencies:
        return {
            "name": "No fertilizer needed.",
            "N": 0,
            "P": 0,
            "K": 0,
            "category": "Balanced",
            "use": "Soil has balanced nutrients"
        }

    try:
        chem_df = pd.read_csv("data/fertilizers_chemical.csv")
        org_df = pd.read_csv("data/fertilizers_organic.csv")

        combined = pd.concat([chem_df, org_df], ignore_index=True).fillna(0)

        def score(row):
            return sum(row[n] for n in deficiencies)

        combined["score"] = combined.apply(score, axis=1)
        top = combined.sort_values(by="score", ascending=False).iloc[0]

        # 🛠️ Map to expected return fields
        return {
            "name": top["Fertilizer"],
            "N": top.get("N (%)", 0),
            "P": top.get("P2O5 (%)", 0),
            "K": top.get("K2O (%)", 0),
            "category": "Organic" if top["Fertilizer"] in org_df["Fertilizer"].values else "Chemical",
            "use": f"Best suited for {' & '.join([d.split()[0] for d in deficiencies])} deficiency."
        }

    except Exception as e:
        return {
            "name": "Error reading fertilizer data",
            "N": 0,
            "P": 0,
            "K": 0,
            "category": "Error",
            "use": str(e)
        }
