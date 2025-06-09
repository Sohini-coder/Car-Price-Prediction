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

# Unpack the categories in correct order
brands = sorted(cat_levels[0])
models = sorted(cat_levels[1])
fuel_types = sorted(cat_levels[2])
transmissions = sorted(cat_levels[3])

# --- LOAD DATASET FOR INPUT OPTIONS ---
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
    engine = st.slider("Engine Size (L)", 0.5, 5.0, 2.0, step=0.1)
    fuel_eff = st.slider("Fuel Efficiency (km/L)", 5.0, 25.0, 12.5, step=0.1)

    owners = st.selectbox("Previous Owners", sorted(df["Previous_Owners"].dropna().unique()))
    demand = st.selectbox("Demand Trend", sorted(df["Demand_Trend"].dropna().unique()))
    accidents = st.selectbox("Accident History", sorted(df["Accident_History"].dropna().unique()))
    condition = st.slider("Condition Score", 1.0, 10.0, 9.0, step=0.1)
    service = st.selectbox("Service History", sorted(df["Service_History"].dropna().unique()))

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
        # Force input_df to match the training feature layout
        transformed_input = pipeline.named_steps["preprocessor"].transform(input_df)
        st.write("Input shape:", transformed_input.shape)
        # Predict
        prediction = pipeline.named_steps["model"].predict(transformed_input)[0]

        st.success(f"💰 Predicted Resale Price: ₹{int(prediction):,}")
        st.session_state.show_chart = True

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.session_state.show_chart = False

# --- CORE FEATURE IMPORTANCE (Table + Graph) ---
if st.session_state.get("show_chart", False):
    try:
        model = pipeline.named_steps["model"]
        importances = model.feature_importances_

        # Feature names from preprocessor (with prefixes like 'scale__Mileage')
        feature_df = pd.DataFrame({
            "Feature": final_feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        # Only include core features that were scaled
        core_keywords = [
            "Car_Age", "Mileage", "Engine_Size", "Fuel_Efficiency",
            "Previous_Owners", "Demand_Trend", "Accident_History",
            "Car_Condition_Score", "Service_History"
        ]

        # Filter matching those core feature substrings
        core_df = feature_df[feature_df["Feature"].str.contains("|".join(core_keywords))]
        core_df = core_df.reset_index(drop=True)
        core_df["Feature"] = core_df["Feature"].str.replace("scale__", "")


        st.subheader("📋 Core Feature Importance (Table)")
        st.dataframe(core_df.style.background_gradient(cmap="Oranges"))

        # Add a horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(core_df["Feature"], core_df["Importance"], color="#FFA07A")
        ax.set_xlabel("Importance")
        ax.set_title("Core Feature Impact on Price")
        ax.invert_yaxis()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Could not display core feature importance: {e}")
