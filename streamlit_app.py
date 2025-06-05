import streamlit as st
import pandas as pd
import pickle
import time
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")

import base64

def set_background(image_file):
    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            color: white;
        }}
        .main > div {{
            background-color: rgba(0, 0, 0, 0.5);  /* Black with 50% opacity */
            padding: 2rem;
            border-radius: 1rem;
        }}
        .css-1cpxqw2 {{
            background-color: transparent !important;
        }}
        .css-18ni7ap {{
            background-color: transparent !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("background.png")

# --- LOAD MODEL (Pipeline) ---
with open('car_price_pipeline.pkl', 'rb') as file:
    pipeline = pickle.load(file)

# --- DROPDOWN OPTIONS ---
df = pd.read_csv('expanded_car_price_prediction.csv')
brands = sorted(df['Brand'].dropna().unique())
models = sorted(df['Model'].dropna().unique())
fuel_types = sorted(df['Fuel_Type'].dropna().unique())
transmissions = sorted(df['Transmission'].dropna().unique())

# --- Feature Lists for Chart ---
categorical_features = ['Brand', 'Model', 'Fuel_Type', 'Transmission']
numerical_features = ['Car_Age', 'Mileage', 'Engine_Size', 'Fuel_Efficiency',
                      'Previous_Owners', 'Resale_Value', 'Demand_Trend',
                      'Accident_History', 'Car_Condition_Score', 'Service_History']

# --- HEADER ---
st.markdown("""
    <style>
    @keyframes glow {
        0% { text-shadow: 0 0 5px #fff; }
        50% { text-shadow: 0 0 20px #ff4b4b, 0 0 30px #ff4b4b; }
        100% { text-shadow: 0 0 5px #fff; }
    }
    .glow-title {
        font-size: 3rem;
        color: #ff4b4b;
        animation: glow 2s infinite;
        font-weight: bold;
    }
    .glass-box {
        background: rgba(0, 0, 0, 0.55);
        border-radius: 1.5rem;
        padding: 2rem;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.4);
    }
    </style>

    <div style='text-align: center; margin-bottom: 1rem;'>
        <div class='glass-box'>
            <h1 class='glow-title'>🚘 Car Price Prediction App</h1>
            <h4 style='color: white; font-weight: normal;'>🔮 Predict resale price with machine learning magic!</h4>
        </div>
    </div>
""", unsafe_allow_html=True)
st.markdown("---")

# --- LAYOUT ---

left_col, right_col = st.columns(2)

with left_col:
        brand = st.selectbox("🏢 Car Brand", brands)
        model_name = st.selectbox("🚘 Car Model", models)
        car_age = st.number_input("📅 Car Age (years)", min_value=0, max_value=30, value=5)
        mileage = st.number_input("🛣️ Mileage (in km)", min_value=0, max_value=300000, value=50000)
        engine_size = st.number_input("⚙️ Engine Size (liters)", min_value=0.5, max_value=8.0, value=2.0, step=0.1)
        fuel_type = st.selectbox("⛽ Fuel Type", fuel_types)
        transmission = st.selectbox("🔁 Transmission Type", transmissions)

with right_col:
        fuel_efficiency = st.number_input("📏 Fuel Efficiency (km/l)", min_value=5.0, max_value=35.0, value=15.0)
        previous_owners = st.number_input("👤 Previous Owners", min_value=0, max_value=10, value=1)
        resale_value = st.number_input("💸 Resale Value (₹)", min_value=0.0, max_value=1000000.0, value=50000.0, step=1000.0)
        demand_trend = st.slider("📈 Demand Trend", min_value=1, max_value=5, value=3)
        accident_history = st.radio("🛑 Accident History", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        car_condition_score = st.slider("✅ Car Condition Score", min_value=1.0, max_value=10.0, value=7.0, step=0.1)
        service_history = st.radio("🧰 Service History", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")


# --- PREDICTION ---
input_data = pd.DataFrame([{
    'Brand': brand,
    'Model': model_name,
    'Car_Age': car_age,
    'Mileage': mileage,
    'Engine_Size': engine_size,
    'Fuel_Type': fuel_type,
    'Transmission': transmission,
    'Fuel_Efficiency': fuel_efficiency,
    'Previous_Owners': previous_owners,
    'Resale_Value': resale_value,
    'Demand_Trend': demand_trend,
    'Accident_History': accident_history,
    'Car_Condition_Score': car_condition_score,
    'Service_History': service_history
}])

# Initialize state variables (run once)
if 'show_price' not in st.session_state:
    st.session_state.show_price = False
if 'show_chart' not in st.session_state:
    st.session_state.show_chart = False

# --- PREDICT BUTTON ---
if st.button("🔮 Predict Price"):
    with st.spinner("Crunching numbers... please wait 🚗💨"):
        time.sleep(1)
        price = pipeline.predict(input_data)[0]
        st.session_state.price = int(price)
        st.session_state.show_price = True

# --- FEATURE IMPORTANCE BUTTON ---
if st.button("📊 Show Feature Importance"):
    with st.spinner("Calculating top features 🚗💨"):
        time.sleep(1)
        st.session_state.show_chart = True

# --- DISPLAY PRICE IF SET ---
if st.session_state.show_price:
    st.success(f"💰 Estimated Car Price: ₹{st.session_state.price:,}")
    

# --- DISPLAY CHART IF SET ---
if st.session_state.show_chart:
    model = pipeline.named_steps['model']
    preprocessor = pipeline.named_steps['preprocessor']

    try:
        onehot = preprocessor.named_transformers_['onehot']
        onehot_features = onehot.get_feature_names_out(categorical_features)
        all_feature_names = list(onehot_features) + numerical_features

        importances = model.feature_importances_

        st.subheader("📊 Feature Importance (What Affects Price Most)")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(all_feature_names, importances, color="#4CAF50")
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance")
        ax.invert_yaxis()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Could not display feature importance: {e}")