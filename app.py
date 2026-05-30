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
# UI STYLE
# -------------------------------
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }

    .title {
        text-align: center;
        font-size: 52px;
        font-weight: 900;
        color: #1f3b57;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: gray;
        margin-bottom: 30px;
    }

    .block {
        background: white;
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# TITLE (BIG)
# -------------------------------
st.markdown('<div class="title">📦 Demand Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered sales forecasting dashboard</div>', unsafe_allow_html=True)

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    with gzip.open("model_compressed.pkl.gz", "rb") as f:
        return pickle.load(f)

model = load_model()

# -------------------------------
# DATE BLOCK
# -------------------------------
st.markdown('<div class="block">', unsafe_allow_html=True)
st.subheader("📅 Date Information")

date = st.date_input("Select Date")

year = date.year
month = date.month
day = date.day
day_of_week = date.weekday()

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# PRODUCT BLOCK
# -------------------------------
st.markdown('<div class="block">', unsafe_allow_html=True)
st.subheader("🏪 Product Information")

product_id = st.number_input("Product ID", value=1000)
category_id = st.number_input("Category ID", value=1)
store_id = st.number_input("Store ID", value=1)

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# SALES BLOCK
# -------------------------------
st.markdown('<div class="block">', unsafe_allow_html=True)
st.subheader("📊 Sales & Market Data")

historical_sales = st.number_input("Historical Sales", value=10)
price = st.number_input("Price", value=50.0)
economic_index = st.number_input("Economic Index", value=100.0)

promotion_flag = st.selectbox("Promotion Flag", [0, 1])
holiday_flag = st.selectbox("Holiday Flag", [0, 1])

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# PREDICT BUTTON
# -------------------------------
st.markdown('<div class="block">', unsafe_allow_html=True)
st.subheader("🚀 Prediction")

if st.button("Predict Demand"):

    try:
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

        # Match training features
        if hasattr(model, "feature_names_in_"):
            input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

        prediction = model.predict(input_df)[0]

        st.success(f"📦 Predicted Demand: {int(prediction)} units")

        if prediction < 10:
            st.warning("Low Demand ⚠️")
        elif prediction < 30:
            st.info("Medium Demand 📊")
        else:
            st.success("High Demand 🔥")

    except Exception as e:
        st.error("Prediction failed ❌")
        st.exception(e)

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# DEBUG
# -------------------------------
with st.expander("🔍 Debug Info"):
    if hasattr(model, "feature_names_in_"):
        st.write(model.feature_names_in_)
