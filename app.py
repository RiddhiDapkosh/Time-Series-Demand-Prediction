import streamlit as st
import pandas as pd
import numpy as np
import gzip
import pickle

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Demand Prediction System",
    layout="wide",
    page_icon="📦"
)

# -------------------------------
# STYLE
# -------------------------------
st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 48px;
        font-weight: 900;
        color: white;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: gray;
        margin-bottom: 25px;
    }

    .card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 15px;
    }

    </style>
""", unsafe_allow_html=True)

# -------------------------------
# TITLE
# -------------------------------
st.markdown('<div class="title">📦 Demand Forecast Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Smart AI-based Sales Forecasting Dashboard</div>', unsafe_allow_html=True)

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    with gzip.open("model_compressed.pkl.gz", "rb") as f:
        return pickle.load(f)

model = load_model()

# -------------------------------
# SIDEBAR INPUTS (CLEAN UX)
# -------------------------------
st.sidebar.header("📌 Input Parameters")

st.sidebar.subheader("📅 Date Info")
date = st.sidebar.date_input("Select Date")

year = date.year
month = date.month
day = date.day
day_of_week = date.weekday()

st.sidebar.subheader("🏪 Product Info")
product_id = st.sidebar.number_input("Product ID", value=1000)
category_id = st.sidebar.number_input("Category ID", value=1)
store_id = st.sidebar.number_input("Store ID", value=1)

st.sidebar.subheader("📊 Sales Info")
historical_sales = st.sidebar.number_input("Historical Sales", value=10)
price = st.sidebar.number_input("Price", value=50.0)
economic_index = st.sidebar.number_input("Economic Index", value=100.0)

promotion_flag = st.sidebar.selectbox("Promotion Flag", [0, 1])
holiday_flag = st.sidebar.selectbox("Holiday Flag", [0, 1])

# -------------------------------
# MAIN AREA
# -------------------------------
# -------------------------------
# MAIN AREA
# -------------------------------
st.markdown("## 📊 Prediction Panel")

col1, col2, col3 = st.columns([2, 1, 2])

with col2:
    predict_btn = st.button("🚀 Predict Demand")

if predict_btn:

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

        # match model features
        if hasattr(model, "feature_names_in_"):
            input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

        prediction = model.predict(input_df)[0]

        st.success(f"📦 Predicted Demand: {int(prediction)} units")

    except Exception as e:
        st.error("Prediction failed ❌")
        st.exception(e)

        #     # -------------------------------
        #     # RESULT CARD
        #     # -------------------------------
        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")

        st.success(f"📦 Predicted Demand: **{int(prediction)} units**")

            # Insight messages
        if prediction < 10:
            st.warning("⚠️ Low Demand Expected")
        elif prediction < 30:
            st.info("📊 Moderate Demand Expected")
        else:
            st.success("🔥 High Demand Expected")

        except Exception as e:
            st.error("Prediction failed ❌")
            st.exception(e)

# -------------------------------
# DEBUG SECTION
# -------------------------------
with st.expander("🔍 Debug Info"):
    if hasattr(model, "feature_names_in_"):
        st.write("Model Features:")
        st.write(list(model.feature_names_in_))
