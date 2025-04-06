import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib

# Load model and encoder
model = joblib.load("model/model.pkl")
le = joblib.load("model/label_encoder.pkl")

# Load data
df = pd.read_csv("data/1.csv")
df.fillna(df.mean(numeric_only=True), inplace=True)
df['Crop_Label'] = le.transform(df['Crop_Type'])

features = [
    'Soil_pH', 'Nitrogen_Content', 'Phosphorus_Content', 'Potassium_Content',
    'Sulfur_Content', 'Zinc_Content',
    'Specific_Humidity_Summer', 'Max_Temperature_Summer', 'Total_Precipitation_Summer'
]

X = df[features]
y = df['Crop_Label']

# Split and evaluate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.2f}")

print("\n🔍 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
