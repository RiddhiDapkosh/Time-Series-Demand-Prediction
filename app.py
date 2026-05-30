import streamlit as st
import numpy as np
import pandas as pd
import gzip
import pickle

# -------------------------------
# Load Model
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
# Page Config
# -------------------------------
st.set_page_config(page_title="Demand Prediction App", layout="centered")
st.title("📊 Demand Prediction System")

st.write("Enter product and market details")

# -------------------------------
# User Inputs (MATCH DATASET)
# -------------------------------

date = st.date_input("Date")

product_id = st.number_input("Product ID", value=1000)
category_id = st.number_input("Category ID", value=1)
store_id = st.number_input("Store ID", value=1)

historical_sales = st.number_input("Historical Sales", value=10)
price = st.number_input("Price", value=50.0)

promotion_flag = st.selectbox("Promotion", [0, 1])
holiday_flag = st.selectbox("Holiday", [0, 1])

economic_index = st.number_input("Economic Index", value=100.0)

# -------------------------------
# Feature Engineering (IMPORTANT)
# -------------------------------

# Convert date into useful features
year = date.year
month = date.month
day = date.day
day_of_week = date.weekday()

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Demand"):

    if model is None:
        st.error("Model not loaded.")
    else:
        try:
            # Create DataFrame (BEST PRACTICE)
            input_data = pd.DataFrame({
                'year': [year],
                'month': [month],
                'day': [day],
                'day_of_week': [day_of_week],
                'product_id': [product_id],
                'category_id': [category_id],
                'store_id': [store_id],
                'historical_sales': [historical_sales],
                'price': [price],
                'promotion_flag': [promotion_flag],
                'holiday_flag': [holiday_flag],
                'economic_index': [economic_index]
            })

            # Prediction
            prediction = model.predict(input_data)

            st.success(f"📦 Predicted Demand: {int(prediction[0])}")

        except Exception as e:
            st.error("Prediction failed. Check feature mismatch.")
            st.exception(e)

# -------------------------------
# Debug Section
# -------------------------------
with st.expander("🔍 Debug Info"):
    st.write("Model Loaded:", model is not None)
