import streamlit as st
import numpy as np
import pandas as pd
import gzip
import pickle

# -------------------------------
# Load Model (Cached)
# -------------------------------
@st.cache_resource
def load_model():
    try:
        with gzip.open("model_compressed.pkl.gz", "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# -------------------------------
# App Title
# -------------------------------
st.set_page_config(page_title="Prediction App", layout="centered")
st.title("📦 Shipment / Prediction App")

st.write("Enter details below to get prediction")

# -------------------------------
# User Inputs (Example Inputs)
# Modify these based on YOUR dataset
# -------------------------------

age = st.number_input("Age", min_value=0, max_value=100, value=25)
salary = st.number_input("Salary", min_value=0, value=30000)

# Example categorical input
gender = st.selectbox("Gender", ["Male", "Female"])

# Encode categorical manually (update based on training)
gender_encoded = 1 if gender == "Male" else 0

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("Predict"):

    if model is None:
        st.error("Model not loaded properly.")
    else:
        try:
            # Create input array (IMPORTANT: match training format)
            input_data = np.array([[age, salary, gender_encoded]])

            # Prediction
            prediction = model.predict(input_data)

            # Output
            st.success(f"Prediction Result: {prediction[0]}")

        except Exception as e:
            st.error("Prediction failed. Check model features.")
            st.exception(e)

# -------------------------------
# Debug Section (Optional)
# -------------------------------
with st.expander("🔍 Debug Info"):
    st.write("Model Loaded:", model is not None)
    st.write("Input Data Shape:", np.array([[age, salary, gender_encoded]]).shape)
