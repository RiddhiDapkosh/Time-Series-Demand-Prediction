import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import gzip
import pickle

# ---------------------------
# Load Model (TOP)
# ---------------------------
@st.cache_resource
def load_model():
    with gzip.open("model_compressed.pkl.gz", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# ---------------------------
# Title
# ---------------------------
st.title("📦 Demand Forecasting Dashboard")

# ---------------------------
# Prediction UI
# ---------------------------
st.subheader("🔮 Predict Future Demand")

day = st.slider("Day", 1, 31, 15)
month = st.slider("Month", 1, 12, 6)
year = st.slider("Year", 2013, 2025, 2018)

# ---------------------------
# Prediction Logic (CONNECTED)
# ---------------------------
if st.button("Predict"):
    # Create input DataFrame (VERY IMPORTANT)
    input_data = pd.DataFrame({
        'day': [day],
        'month': [month],
        'year': [year]
    })

    # Predict
    prediction = model.predict(input_data)

    # Output
    st.success(f"📦 Predicted Demand: {int(prediction[0])}")

# ---------------------------
# (Optional) Info
# ---------------------------
st.info("Model uses day, month, and year features for prediction.")
