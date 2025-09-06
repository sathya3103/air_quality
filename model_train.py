# model_train.py
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ✅ Expected pollutants
FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]

# Paths
DATA_PATH = "data/air_quality.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "aqi_model.pkl")

# Load dataset
if os.path.exists(DATA_PATH):
    print("✅ Using local dataset:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
else:
    raise FileNotFoundError(f"❌ Dataset not found at {DATA_PATH}. Please place it inside the data/ folder.")

# Ensure AQI column exists
if "AQI" not in df.columns:
    raise ValueError("❌ Dataset must contain an 'AQI' column.")

print("🔧 Cleaning dataset...")

# Ensure all required features exist
for f in FEATURES:
    if f not in df.columns:
        df[f] = 0   # add missing pollutant column
        print(f"⚠️ Added missing column: {f} (filled with 0)")

# Fill missing values
df = df.fillna(0)
df["AQI"] = df["AQI"].fillna(method="ffill")

# Prepare training data
X = df[FEATURES]
y = df["AQI"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
print("🚀 Training RandomForestRegressor...")
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Model trained successfully!")
print(f"📊 MAE: {mae:.2f}, R²: {r2:.2f}")

# Save model
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"💾 Model saved to {MODEL_PATH}")
print("🎉 Training complete!")