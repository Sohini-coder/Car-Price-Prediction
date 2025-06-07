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

# --- LOAD MODEL ---
with open("car_price_pipeline_updated.pkl", "rb") as f:
    pipeline = pickle.load(f)

with open("feature_names.pkl", "rb") as f:
    final_feature_names = pickle.load(f)

# --- INPUT FORM ---
st.title("🚗 Car Price Prediction App")

with st.form("prediction_form"):
    st.subheader("🔧 Enter Car Details:")

    brand = st.selectbox("Brand", ['BMW', 'Audi', 'Mercedes', 'Toyota', 'Jaguar', 'Skoda', 'Hyundai', 'Mahindra', 'Kia'])
    model = st.text_input("Model", "5 Series")
    fuel = st.selectbox("Fuel Type", ['Petrol', 'Diesel'])
    trans = st.selectbox("Transmission", ['Manual', 'Automatic'])

    age = st.slider("Car Age (years)", 0, 20, 2)
    mileage = st.slider("Mileage (km)", 0, 300000, 15000, step=1000)
    engine = st.slider("Engine Size (L)", 0.5, 5.0, 2.0, step=0.1)
    fuel_eff = st.slider("Fuel Efficiency (km/L)", 5.0, 25.0, 12.5, step=0.1)
    owners = st.selectbox("Previous Owners", [0, 1, 2, 3])
    demand = st.slider("Demand Trend", 1, 5, 4)
    accidents = st.selectbox("Accident History", [0, 1])
    condition = st.slider("Condition Score", 1.0, 10.0, 9.0, step=0.1)
    service = st.selectbox("Service History", [0, 1])

    submitted = st.form_submit_button("Predict Price")

# --- PREDICTION ---
if submitted:
    input_df = pd.DataFrame([{
        'Brand': brand,
        'Model': model,
        'Fuel_Type': fuel,
        'Transmission': trans,
        'Car_Age': age,
        'Mileage': mileage,
        'Engine_Size': engine,
        'Fuel_Efficiency': fuel_eff,
        'Previous_Owners': owners,
        'Demand_Trend': demand,
        'Accident_History': accidents,
        'Car_Condition_Score': condition,
        'Service_History': service
    }])

    try:
        prediction = pipeline.predict(input_df)[0]
        st.success(f"💰 Predicted Resale Price: ₹{int(prediction):,}")
        st.session_state.show_chart = True
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.session_state.show_chart = False

# --- FEATURE IMPORTANCE ---
if st.session_state.get("show_chart", False):
    try:
        model = pipeline.named_steps["model"]
        importances = model.feature_importances_

        if len(importances) != len(final_feature_names):
            st.warning("⚠️ Feature mismatch detected. Retrain model to fix.")
        else:
            importance_df = pd.DataFrame({
                "Feature": final_feature_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False)

            st.subheader("📋 Feature Importance")
            st.dataframe(importance_df.style.background_gradient(cmap="Greens"))

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(importance_df["Feature"], importance_df["Importance"], color="#4CAF50")
            ax.set_xlabel("Importance")
            ax.set_title("Feature Importance")
            ax.invert_yaxis()
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Could not display feature importance: {e}")