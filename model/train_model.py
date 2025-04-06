import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import os

# Load and clean data
df = pd.read_csv("data/1.csv")
df.fillna(df.mean(numeric_only=True), inplace=True)

# Label encode crop
le = LabelEncoder()
df['Crop_Label'] = le.fit_transform(df['Crop_Type'])

# Features
features = [
    'Soil_pH', 'Nitrogen_Content', 'Phosphorus_Content', 'Potassium_Content',
    'Sulfur_Content', 'Zinc_Content',
    'Specific_Humidity_Summer', 'Max_Temperature_Summer', 'Total_Precipitation_Summer'
]

X = df[features]
y = df['Crop_Label']

# Handle imbalance
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)

# Train model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42
)
model.fit(X_resampled, y_resampled)

# Save directory
os.makedirs("model", exist_ok=True)

# ✅ Save model and label encoder
joblib.dump(model, "model/model.pkl")
joblib.dump(le, "model/label_encoder.pkl")

# ✅ Save readable class names (e.g., ["Rice", "Wheat", ...])
class_names = le.inverse_transform(sorted(set(y_resampled)))
joblib.dump(class_names, "model/class_names.pkl")

print("✅ Model trained and saved with SMOTE, LabelEncoder, and class names.")
