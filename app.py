import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Title
st.title("📦 Demand Forecasting Dashboard")

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("demand_forecasting_dataset.csv")

df = load_data()

# Show data
st.subheader("📊 Dataset Preview")
st.write(df.head())
st.write("Columns:", df.columns)

# ---------------------------
# Define target
# ---------------------------
target = 'target_demand'

# ---------------------------
# Date processing
# ---------------------------
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# ---------------------------
# Aggregate
# ---------------------------
daily_demand = df.groupby('date')[target].sum().reset_index()

# ---------------------------
# Plot
# ---------------------------
st.subheader("📈 Demand Over Time")

fig, ax = plt.subplots()
ax.plot(daily_demand['date'], daily_demand[target])
ax.set_xlabel("Date")
ax.set_ylabel("Demand")
st.pyplot(fig)

# ---------------------------
# Features
# ---------------------------
daily_demand['day'] = daily_demand['date'].dt.day
daily_demand['month'] = daily_demand['date'].dt.month
daily_demand['year'] = daily_demand['date'].dt.year

X = daily_demand[['day', 'month', 'year']]
y = daily_demand[target]

# ---------------------------
# Model
# ---------------------------
model = LinearRegression()
model.fit(X, y)

# ---------------------------
# Prediction
# ---------------------------
st.subheader("🔮 Predict Future Demand")

day = st.slider("Day", 1, 31, 15)
month = st.slider("Month", 1, 12, 6)
year = st.slider("Year", int(daily_demand['year'].min()), int(daily_demand['year'].max()))

if st.button("Predict"):
    pred = model.predict([[day, month, year]])
    st.success(f"📦 Predicted Demand: {int(pred[0])}")

# ---------------------------
# Distribution
# ---------------------------
st.subheader("📊 Demand Distribution")

fig2, ax2 = plt.subplots()
ax2.hist(daily_demand[target], bins=20)
st.pyplot(fig2)
