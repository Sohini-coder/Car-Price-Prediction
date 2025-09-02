import streamlit as st
import pandas as pd
import pickle
import time
import matplotlib.pyplot as plt
import base64

# --- PAGE CONFIG ---
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")

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
            background-color: rgba(0, 0, 0, 0.5);
            padding: 2rem;
            border-radius: 1rem;
        }}
        .css-1cpxqw2, .css-18ni7ap {{
            background-color: transparent !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("background.png")

# --- LOAD MODEL & FEATURES ---
with open("random_forest_final.pkl", "rb") as f:
    pipeline = pickle.load(f)

with open("feature_names.pkl", "rb") as f:
    final_feature_names = pickle.load(f)

with open("category_levels.pkl", "rb") as f:
    cat_levels = pickle.load(f)

brands = sorted(cat_levels[0])
models = sorted(cat_levels[1])
fuel_types = sorted(cat_levels[2])
transmissions = sorted(cat_levels[3])

# --- LOAD DATASET ---
df = pd.read_csv("dataset_1400.csv")

st.title("🚗 Car Price Prediction App")

# --- INPUT FORM ---
with st.form("prediction_form"):
    st.subheader("🔧 Enter Car Details:")

    brand = st.selectbox("Car Brand", brands)
    model = st.selectbox("Car Model", models)
    fuel = st.selectbox("Fuel Type", fuel_types)
    trans = st.selectbox("Transmission", transmissions)

    age = st.slider("Car Age (years)", 0, 20, 2)
    mileage = st.slider("Mileage (km)", 0, 300000, 15000, step=1000)
    engine_liters = st.slider("Engine Size (L)", 0.5, 5.0, 2.0, step=0.1)
    engine = int(engine_liters * 1000) 
    fuel_eff = st.slider("Fuel Efficiency (km/L)", 5.0, 25.0, 12.5, step=0.1)

    owners = st.selectbox("Previous Owners", sorted(df["Previous_Owners"].dropna().unique()))
    demand = st.selectbox("Demand Trend", sorted(df["Demand_Trend"].dropna().unique()))

    accidents_display = st.selectbox("Accident History", ["No", "Yes"])
    service_display = st.selectbox("Service History", ["No", "Yes"])
    condition = st.slider("Condition Score", 1.0, 10.0, 9.0, step=0.1)

    accidents = 1 if accidents_display == "Yes" else 0
    service = 1 if service_display == "Yes" else 0

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
        transformed_input = pipeline.named_steps["preprocessor"].transform(input_df)
        prediction = pipeline.named_steps["model"].predict(transformed_input)[0]
        st.success(f"💰 Predicted Resale Price: ₹{int(prediction):,}")
        st.session_state.show_chart = True
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.session_state.show_chart = False

# --- FEATURE IMPORTANCE SECTIONS ---
if st.session_state.get("show_chart", False):
    try:
        model = pipeline.named_steps["model"]
        importances = model.feature_importances_
        feature_df = pd.DataFrame({
            "Feature": final_feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        # 1. BRAND & MODEL IMPORTANCE
        brand_model_df = feature_df[feature_df["Feature"].str.contains("Brand_|Model_")].copy()
        brand_model_df["Feature"] = brand_model_df["Feature"].str.replace("onehot__", "")
        brand_model_df = brand_model_df.sort_values(by="Importance", ascending=False).head(15).reset_index(drop=True)

        st.subheader("🚘 Brand & Model Feature Importance")
        st.dataframe(brand_model_df.style.background_gradient(cmap="Purples"))

        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.barh(brand_model_df["Feature"], brand_model_df["Importance"], color="#9370DB")
        ax1.set_xlabel("Importance")
        ax1.set_title("Top Influential Brands & Models")
        ax1.invert_yaxis()
        st.pyplot(fig1)

        # 2. CORE ENGINEERED FEATURES
        core_keywords = [
            "Car_Age", "Mileage", "Engine_Size", "Fuel_Efficiency",
            "Previous_Owners", "Demand_Trend", "Accident_History",
            "Car_Condition_Score", "Service_History"
        ]

        core_df = feature_df[feature_df["Feature"].str.contains("|".join(core_keywords))].copy()
        core_df = core_df.reset_index(drop=True)
        core_df["Feature"] = core_df["Feature"].str.replace("scale__", "")

        st.subheader("📋 Core Feature Importance (Table)")
        st.dataframe(core_df.style.background_gradient(cmap="Oranges"))

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        bars = ax2.barh(core_df["Feature"], core_df["Importance"], color="#FFA07A")
        ax2.set_xlabel("Importance")
        ax2.set_title("Core Feature Impact on Price")
        ax2.invert_yaxis()

        # Adjust axis range to zoom into smaller values
        ax2.set_xlim(0, core_df["Importance"].max() * 1.2)

        # Add importance value as label next to each bar
        for i, bar in enumerate(bars):
            value = core_df["Importance"].iloc[i]
            ax2.text(value + 0.001, bar.get_y() + bar.get_height()/2,
                f"{value:.4f}", va='center', fontsize=9, color='black')

        plt.tight_layout()
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"Could not display feature importance: {e}")
