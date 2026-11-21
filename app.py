import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

BP = {"PM2_5": [0, 30, 60, 90, 120, 250, 380], "PM10": [0, 50, 100, 250, 350, 430, 510],
      "NO2": [0, 40, 80, 180, 280, 400, 520], "SO2": [0, 40, 80, 380, 800, 1600, 2100],
      "CO": [0, 1, 2, 10, 17, 34, 50], "O3": [0, 50, 100, 168, 208, 748, 1000]}
IDX = [0, 50, 100, 200, 300, 400, 500]
FEATURES = ["PM2_5", "PM10", "NO2", "SO2", "CO", "O3", "Temp", "Humidity", "WindSpeed", "Rainfall",
            "AQI", "AQI_lag1", "AQI_roll7", "PM2_5_lag1", "Month_sin", "Month_cos",
            "Wind_next", "Rain_next", "NextDay_Weekend"]
ADVICE = {"Good": "Air quality is good. Normal outdoor activity is fine.",
          "Satisfactory": "Acceptable. Unusually sensitive people may feel minor discomfort.",
          "Moderate": "Children, elderly people and people with asthma should limit long outdoor exertion.",
          "Poor": "Most people should reduce prolonged outdoor activity.",
          "Very Poor": "Avoid outdoor exercise. Keep windows closed.",
          "Severe": "Stay indoors. Sensitive groups should seek medical advice if symptoms appear."}

def aqi_from(values):
    return round(max(np.interp(values[c], BP[c], IDX) for c in BP))

def category(a):
    for limit, name in [(50, "Good"), (100, "Satisfactory"), (200, "Moderate"), (300, "Poor"), (400, "Very Poor")]:
        if a <= limit:
            return name
    return "Severe"

@st.cache_resource
def train_model():
    d = pd.read_csv("bengaluru_aqi_dataset.csv", parse_dates=["Date"]).set_index("Date")
    d = d.interpolate(method="linear", limit_direction="both")
    d["AQI"] = d.apply(lambda r: aqi_from(r), axis=1)
    d["AQI_lag1"] = d["AQI"].shift(1)
    d["AQI_roll7"] = d["AQI"].rolling(7).mean()
    d["PM2_5_lag1"] = d["PM2_5"].shift(1)
    d["Month_sin"] = np.sin(2 * np.pi * d.index.month / 12)
    d["Month_cos"] = np.cos(2 * np.pi * d.index.month / 12)
    d["Wind_next"] = d["WindSpeed"].shift(-1)
    d["Rain_next"] = d["Rainfall"].shift(-1)
    d["NextDay_Weekend"] = ((d.index + pd.Timedelta(days=1)).dayofweek >= 5).astype(int)
    d["Target"] = d["AQI"].shift(-1)
    d = d.dropna()
    return LinearRegression().fit(d[FEATURES], d["Target"])

def predict(model, p):
    row = dict(p)
    row["AQI"] = aqi_from(p)
    row["Month_sin"] = np.sin(2 * np.pi * p["month"] / 12)
    row["Month_cos"] = np.cos(2 * np.pi * p["month"] / 12)
    aqi = int(round(min(max(model.predict(pd.DataFrame([row])[FEATURES])[0], 0), 500)))
    return row["AQI"], aqi, category(aqi)

st.set_page_config(page_title="Bengaluru AQI Predictor")
st.title("Bengaluru AQI Predictor")
st.write("Enter today's readings to estimate tomorrow's Air Quality Index.")
model = train_model()
c1, c2 = st.columns(2)
with c1:
    pm25 = st.number_input("PM2.5 (µg/m³)", 0.0, 500.0, 30.0)
    pm10 = st.number_input("PM10 (µg/m³)", 0.0, 800.0, 60.0)
    no2 = st.number_input("NO2 (µg/m³)", 0.0, 500.0, 32.0)
    so2 = st.number_input("SO2 (µg/m³)", 0.0, 500.0, 10.0)
    co = st.number_input("CO (mg/m³)", 0.0, 50.0, 0.6)
    o3 = st.number_input("O3 (µg/m³)", 0.0, 500.0, 30.0)
    month = st.selectbox("Month (1-12)", list(range(1, 13)), index=0)
with c2:
    temp = st.number_input("Temperature (°C)", 5.0, 45.0, 25.0)
    hum = st.number_input("Humidity (%)", 0.0, 100.0, 65.0)
    wind = st.number_input("Wind speed today (km/h)", 0.0, 80.0, 10.0)
    rain = st.number_input("Rainfall today (mm)", 0.0, 300.0, 0.0)
    wnext = st.number_input("Forecast wind tomorrow (km/h)", 0.0, 80.0, 10.0)
    rnext = st.number_input("Forecast rain tomorrow (mm)", 0.0, 300.0, 0.0)
    wk = st.checkbox("Tomorrow is a weekend")
lag1 = st.number_input("Yesterday's AQI", 0, 500, 60)
roll7 = st.number_input("Average AQI of last 7 days", 0, 500, 60)
pmlag = st.number_input("Yesterday's PM2.5 (µg/m³)", 0.0, 500.0, 30.0)
if st.button("Predict tomorrow's AQI"):
    today, aqi, cat = predict(model, dict(PM2_5=pm25, PM10=pm10, NO2=no2, SO2=so2, CO=co, O3=o3,
        Temp=temp, Humidity=hum, WindSpeed=wind, Rainfall=rain, AQI_lag1=lag1, AQI_roll7=roll7,
        PM2_5_lag1=pmlag, Wind_next=wnext, Rain_next=rnext, NextDay_Weekend=int(wk), month=month))
    st.metric("Today's AQI (calculated)", today)
    st.metric("Predicted AQI for tomorrow", f"{aqi} - {cat}")
    st.info(ADVICE[cat])
st.caption("Estimate only. Always check official CPCB bulletins.")
