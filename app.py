import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page config
st.set_page_config(
    page_title="Food Delivery Time Prediction",
    page_icon="📦",
    layout="wide"
)

# Custom styling
st.markdown("""
    <style>
    .main {
        background-color: #f4f4f4;
        padding: 20px;
    }
    .stButton>button {
        background-color: #007BFF;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        height: 3em;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .stSelectbox>div>div {
        background-color: #fff;
        color: black;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📦 Food Delivery Time Prediction")
st.markdown("""
Welcome! Estimate how long a food delivery will take based on several real-world factors like traffic, weather, and courier details.
""")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("order_delivery_model.pkl")

try:
    model = load_model()
except FileNotFoundError:
    st.error("❌ Model file not found. Please train the model first.")
    st.stop()

# Sidebar inputs
with st.sidebar:
    st.header("📋 Input Order Details")

    col1, col2 = st.columns(2)
    with col1:
        distance_km = st.number_input("🚚 Delivery Distance (km)", min_value=1.0, max_value=100.0, step=0.1)
        preparation_time_min = st.number_input("👨‍🍳 Preparation Time (min)", min_value=1, max_value=60, step=1)
    with col2:
        weather = st.selectbox("⛅ Weather", ["Clear", "Rainy", "Windy", "Foggy"])
        traffic_level = st.selectbox("🚦 Traffic Level", ["Low", "Medium", "High"])

    st.markdown("---")
    st.header("🚴 Courier & Timing Details")

    col3, col4 = st.columns(2)
    with col3:
        courier_experience_yrs = st.number_input("📆 Courier Experience (Years)", min_value=0, max_value=20, step=1)
    with col4:
        vehicle_type = st.selectbox("🚗 Vehicle Type", ["Bike", "Car", "Van"])

    time_of_day = st.selectbox("🕒 Time of Day", ["Morning", "Afternoon", "Evening", "Night"])

    st.markdown(" ")
    predict_btn = st.button("🔮 Predict Delivery Time")

# Preprocess inputs
def preprocess_input(distance, weather, traffic, time_of_day, vehicle, prep_time, experience):
    weather_map = {"Clear": 0, "Rainy": 1, "Windy": 2, "Foggy": 3}
    traffic_map = {"Low": 0, "Medium": 1, "High": 2}
    time_map = {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3}
    vehicle_map = {"Bike": 0, "Car": 1, "Van": 2}

    return np.array([[distance,
                      weather_map[weather],
                      traffic_map[traffic],
                      time_map[time_of_day],
                      vehicle_map[vehicle],
                      prep_time,
                      experience]])

# Prediction
if predict_btn:
    input_data = preprocess_input(
        distance_km,
        weather,
        traffic_level,
        time_of_day,
        vehicle_type,
        preparation_time_min,
        courier_experience_yrs
    )
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"🕑 Estimated Delivery Time: **{round(prediction, 2)} minutes**")
    except Exception as e:
        st.error(f"❌ An error occurred during prediction: {e}")

# Footer
st.markdown("""
---
Made with ❤️ using Streamlit | [GitHub](https://github.com)
""", unsafe_allow_html=True)
