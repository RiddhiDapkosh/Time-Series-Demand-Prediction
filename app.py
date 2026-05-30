import streamlit as st
import pandas as pd
import numpy as np
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
# Page Setup
# -------------------------------
st.set_page_config(page_title="Demand Prediction", layout="centered")
st.title("📦 Demand Prediction App")

# -------------------------------
# Inputs
# -------------------------------
date = st.date_input("Select Date")

product_id = st.number_input("Product ID", value=1000)
category_id = st.number_input("Category ID", value=1)
store_id = st.number_input("Store ID", value=1)

historical_sales = st.number_input("Historical Sales", value=10)
price = st.number_input("Price", value=50.0)

promotion_flag = st.selectbox("Promotion", [0, 1])
holiday_flag = st.selectbox("Holiday", [0, 1])

economic_index = st.number_input("Economic Index", value=100.0)

# -------------------------------
# Predict
# -------------------------------
if st.button("Predict Demand"):

    if model is None:
        st.error("Model not loaded")
    else:
        try:
            # ---- Convert Date ----
            year = date.year
            month = date.month
            day = date.day
            day_of_week = date.weekday()

            # ---- Create DataFrame ----
            input_df = pd.DataFrame({
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

            # -------------------------------
            # 🔥 CRITICAL FIX: Match model features
            # -------------------------------
            if hasattr(model, "feature_names_in_"):
                expected_cols = model.feature_names_in_
                input_df = input_df.reindex(columns=expected_cols, fill_value=0)

            # -------------------------------
            # Prediction
            # -------------------------------
            prediction = model.predict(input_df)

            st.success(f"📊 Predicted Demand: {int(prediction[0])}")

        except Exception as e:
            st.error("❌ Prediction failed. Feature mismatch or preprocessing issue.")
            st.exception(e)

# -------------------------------
# Debug
# -------------------------------
with st.expander("🔍 Debug Info"):
    if model is not None and hasattr(model, "feature_names_in_"):
        st.write("Model expects columns:", model.feature_names_in_)
