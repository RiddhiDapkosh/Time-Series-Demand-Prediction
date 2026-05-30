import streamlit as st
import pandas as pd
import numpy as np
import gzip
import pickle

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Demand Prediction", layout="wide")

# -------------------------------
# Custom Styling (🔥 UI Upgrade)
# -------------------------------
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .title {
        font-size: 40px;
        font-weight: bold;
        color: white;
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">📦 Demand Prediction Dashboard</p>', unsafe_allow_html=True)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    with gzip.open("model_compressed.pkl.gz", "rb") as f:
        return pickle.load(f)

model = load_model()

# -------------------------------
# Layout
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📅 Date Info")
    date = st.date_input("Select Date")

    year = date.year
    month = date.month
    day = date.day
    day_of_week = date.weekday()

with col2:
    st.markdown("### 🏪 Product Info")
    product_id = st.number_input("Product ID", value=1000)
    category_id = st.number_input("Category ID", value=1)
    store_id = st.number_input("Store ID", value=1)

st.markdown("### 📊 Sales Info")
col3, col4, col5 = st.columns(3)

with col3:
    historical_sales = st.number_input("Historical Sales", value=10)

with col4:
    price = st.number_input("Price", value=50.0)

with col5:
    economic_index = st.number_input("Economic Index", value=100.0)

col6, col7 = st.columns(2)

with col6:
    promotion_flag = st.selectbox("Promotion", [0, 1])

with col7:
    holiday_flag = st.selectbox("Holiday", [0, 1])

# -------------------------------
# Prediction
# -------------------------------
if st.button("🚀 Predict Demand"):

    try:
        # Create input DataFrame EXACTLY
        input_df = pd.DataFrame([{
            'year': year,
            'month': month,
            'day': day,
            'day_of_week': day_of_week,
            'product_id': product_id,
            'category_id': category_id,
            'store_id': store_id,
            'historical_sales': historical_sales,
            'price': price,
            'promotion_flag': promotion_flag,
            'holiday_flag': holiday_flag,
            'economic_index': economic_index
        }])

        # 🔥 FIX: Match training columns
        if hasattr(model, "feature_names_in_"):
            input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

        # Predict
        prediction = model.predict(input_df)[0]

        # -------------------------------
        # Output UI
        # -------------------------------
        st.markdown("### 📈 Prediction Result")

        st.success(f"Predicted Demand: {int(prediction)} units")

        if prediction < 10:
            st.warning("⚠️ Low Demand Expected")
        elif prediction < 30:
            st.info("📊 Medium Demand")
        else:
            st.success("🔥 High Demand Expected!")

    except Exception as e:
        st.error("Prediction failed ❌")
        st.exception(e)

# -------------------------------
# Debug Panel
# -------------------------------
with st.expander("🔍 Debug Info"):
    if hasattr(model, "feature_names_in_"):
        st.write("Expected Features:", list(model.feature_names_in_))
