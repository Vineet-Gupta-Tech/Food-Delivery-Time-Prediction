import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="Delivery Time Predictor",
    layout="centered"
)

st.title("Delivery Time Prediction")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("order_delivery_model.pkl")

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file not found. Please train the model first.")
    st.stop()

# Sidebar inputs
with st.sidebar:
    st.header("Input Parameters")
    product_category = st.selectbox("Product Category", ["Electronics", "Clothing", "Furniture"])
    customer_location = st.selectbox("Customer Location", ["Urban", "Suburban", "Rural"])
    shipping_method = st.selectbox("Shipping Method", ["Standard", "Express", "Same-Day"])
    predict_btn = st.button("Predict Delivery Time")

# Preprocessing
def preprocess_input(category, location, shipping):
    category_map = {"Electronics": 0, "Clothing": 1, "Furniture": 2}
    location_map = {"Urban": 0, "Suburban": 1, "Rural": 2}
    shipping_map = {"Standard": 0, "Express": 1, "Same-Day": 2}
    return np.array([[category_map[category], location_map[location], shipping_map[shipping]]])

# Prediction
if predict_btn:
    input_data = preprocess_input(product_category, customer_location, shipping_method)
    prediction = model.predict(input_data)[0]
    st.subheader("Estimated Delivery Time")
    st.write(f"{round(prediction, 2)} days")
