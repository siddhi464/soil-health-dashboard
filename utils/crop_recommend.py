import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import random

# Load dataset once
df = pd.read_csv("data/1.csv")
df.fillna(df.mean(numeric_only=True), inplace=True)

# Load dynamic crop rotation map from CSV
def load_rotation_map(csv_path="data/crop_rotation.csv"):
    rotation_df = pd.read_csv(csv_path)
    return dict(zip(rotation_df['Current_Crop'], rotation_df['Next_Crop']))

rotation_map = load_rotation_map()

# Label encode crop type
le = LabelEncoder()
df['Crop_Label'] = le.fit_transform(df['Crop_Type'])

# Features used for crop prediction
features = [
    'Soil_pH', 'Nitrogen_Content', 'Phosphorus_Content', 'Potassium_Content',
    'Sulfur_Content', 'Zinc_Content',
    'Specific_Humidity_Summer', 'Max_Temperature_Summer', 'Total_Precipitation_Summer'
]

X = df[features]
y = df['Crop_Label']

model = RandomForestClassifier()
model.fit(X, y)

# Dynamically assign sulfur levels using quantiles
df['Sulfur_Level'] = pd.qcut(df['Sulfur_Content'], q=3, labels=["Low", "Moderate", "High"])

def get_crop_recommendation(ph, location, current_crop, csv_df, weather):
    closest_row = csv_df.iloc[(csv_df['Soil_pH'] - ph).abs().argsort()[:1]]

    input_data = pd.DataFrame({
        'Soil_pH': [ph],
        'Nitrogen_Content': closest_row['Nitrogen_Content'].values[0],
        'Phosphorus_Content': closest_row['Phosphorus_Content'].values[0],
        'Potassium_Content': closest_row['Potassium_Content'].values[0],
        'Sulfur_Content': closest_row['Sulfur_Content'].values[0],
        'Zinc_Content': closest_row['Zinc_Content'].values[0],
        'Specific_Humidity_Summer': weather['humidity'],
        'Max_Temperature_Summer': weather['temperature'],
        'Total_Precipitation_Summer': weather['precip']
    }, index=[0])

    crop_index = model.predict(input_data)[0]
    crop = le.inverse_transform([crop_index])[0]

    # Microbial biodiversity logic using pH and sulfur level
    sulfur_val = input_data['Sulfur_Content'].values[0]
    closest_match = df.iloc[(df['Sulfur_Content'] - sulfur_val).abs().argsort()[:1]]
    sulfur_level = closest_match['Sulfur_Level'].values[0]

    if 6.0 <= ph <= 7.5 and sulfur_level == "High":
        microbial_health = "High"
    elif (5.5 <= ph < 6.0 or 7.5 < ph <= 8.0) or sulfur_level == "Moderate":
        microbial_health = "Moderate"
    else:
        microbial_health = "Low"

    explanation = (
        f"Based on your soil pH of {ph}, which falls in the "
        f"{'optimal' if 6.0 <= ph <= 7.5 else 'moderate' if 5.5 <= ph <= 8.0 else 'less ideal'} range for most crops, "
        f"and the weather conditions in {location} — specifically a temperature of {weather['temperature']}°C, "
        f"humidity of {weather['humidity']}%, and {weather['precip']} mm of rainfall — "
        f"{crop} is a suitable choice for this season."
    )

    if current_crop:
        explanation += (
            f" Additionally, rotating from {current_crop} to {crop} helps maintain soil fertility and reduce pest buildup."
        )

    if microbial_health == "High":
        explanation += " The soil’s high microbial activity indicates excellent biological health, further supporting robust crop growth."
    elif microbial_health == "Moderate":
        explanation += " The microbial activity is moderate, which is generally sufficient, but adding organic matter can boost it."
    else:
        explanation += " However, the microbial health appears low, possibly due to poor sulfur levels or unbalanced pH. Improving this could enhance crop resilience."

    # Suggested next crop from rotation CSV
    next_crop = rotation_map.get(crop, random.choice(list(rotation_map.values())))

    return {
        "crop": crop,
        "nutrients": {
            "N": input_data['Nitrogen_Content'].values[0],
            "P": input_data['Phosphorus_Content'].values[0],
            "K": input_data['Potassium_Content'].values[0],
            "S": input_data['Sulfur_Content'].values[0]
        },
        "microbial_health": microbial_health,
        "explanation": explanation,
        "next_crop": next_crop
    }
