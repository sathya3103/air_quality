import streamlit as st
import pandas as pd
import joblib
import os

# --- Load dataset ---
DATA_PATH = "data/air_quality.csv"
MODEL_PATH = "model/aqi_model.pkl"

if not os.path.exists(DATA_PATH):
    st.error("❌ Dataset not found! Please check data/air_quality.csv")
    st.stop()

df = pd.read_csv(DATA_PATH)

# Ensure consistent column names
df.columns = [c.strip() for c in df.columns]

# --- Load trained model ---
if not os.path.exists(MODEL_PATH):
    st.error("❌ Trained model not found! Run model_train.py first.")
    st.stop()

model = joblib.load(MODEL_PATH)

# --- Features ---
FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]

# --- Streamlit UI ---
st.title("🌍 Air Quality Prediction in Indian Cities")
st.subheader("Select a city to view AQI predictions & history")

cities = sorted(df["City"].dropna().unique())
city = st.selectbox("🏙️ Choose a City", cities)

df_city = df[df["City"] == city].copy()

if df_city.empty:
    st.warning(f"No data available for {city}")
    st.stop()

# --- Latest record ---
latest = df_city.iloc[-1]

# --- Prepare features for prediction ---
X = pd.DataFrame([latest[FEATURES].to_dict()])
pred_aqi = model.predict(X)[0]

# --- AQI Category ---
def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good 🟢"
    elif aqi <= 100:
        return "Satisfactory 🟡"
    elif aqi <= 200:
        return "Moderate 🟠"
    elif aqi <= 300:
        return "Poor 🔴"
    elif aqi <= 400:
        return "Very Poor 🟣"
    else:
        return "Severe ⚫"

aqi_category = get_aqi_category(pred_aqi)

# --- Display Predicted AQI ---
st.markdown(f"### Predicted AQI for {city}: **{int(pred_aqi)} ({aqi_category})**")

# --- Pollutant Breakdown ---
st.subheader("📊 Latest Pollutant Levels")
pollutant_data = latest[FEATURES].to_dict()
st.bar_chart(pd.DataFrame(pollutant_data, index=["Concentration (µg/m³)"]).T)

# --- Latest Pollutants Table ---
st.table(pd.DataFrame([pollutant_data]))

# --- Historical AQI Trend ---
datetime_cols = [c for c in df_city.columns if c.lower() in ["datetime", "date", "timestamp"]]

if datetime_cols:
    time_col = datetime_cols[0]
    df_city[time_col] = pd.to_datetime(df_city[time_col], errors="coerce")
    df_city = df_city.dropna(subset=[time_col])
    df_city = df_city.sort_values(time_col)

    st.subheader(f"📈 Historical AQI trend for {city}")
    st.line_chart(df_city.set_index(time_col)["AQI"])
else:
    st.info("ℹ️ No datetime column found for plotting trends.")
