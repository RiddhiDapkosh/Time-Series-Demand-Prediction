import streamlit as st
import pandas as pd
import gzip
import pickle

# ---------------------------
# Load Model
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
# User Inputs
# ---------------------------
st.subheader("🔮 Predict Future Demand")

day = st.number_input("Day", min_value=1, max_value=31, value=15)
month = st.number_input("Month", min_value=1, max_value=12, value=6)
year = st.number_input("Year", min_value=2013, max_value=2030, value=2018)

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Demand"):

    # Create input data (must match training features)
    input_df = pd.DataFrame({
        "day": [day],
        "month": [month],
        "year": [year]
    })

    try:
        prediction = model.predict(input_df)
        result = int(prediction[0])

        # Avoid negative values
        if result < 0:
            result = 0

        st.success(f"📦 Predicted Demand: {result}")

    except Exception as e:
        st.error("❌ Prediction failed. Check model features.")
        st.write(e)

# ---------------------------
# Info
# ---------------------------
st.info("This model predicts demand based on day, month, and year.")
