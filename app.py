# Lithium Predictor - Streamlit App (CatBoost)

# Save this as `app.py` and run with: streamlit run app.py

import streamlit as st
import numpy as np
from catboost import CatBoostRegressor

# ✅ Load trained CatBoost model
@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model("catboost_lithium_model.cbm")
    return model

model = load_model()

# ✅ App Header
st.title("🧪 Lithium Concentration Predictor")
st.markdown("Enter formation water chemistry to estimate lithium content (mg/L)")

# ✅ User Inputs
na = st.number_input("Sodium (mg/L)", value=14000.0)
k = st.number_input("Potassium (mg/L)", value=400.0)
mg = st.number_input("Magnesium (mg/L)", value=1200.0)
ca = st.number_input("Calcium (mg/L)", value=2000.0)
cl = st.number_input("Chloride (mg/L)", value=22000.0)
tds = st.number_input("Total Dissolved Solids (mg/L)", value=38000.0)

# ✅ Feature Engineering Function
def engineer_features(mg, na, k, ca, cl, tds):
    features = [
        mg,
        na,
        k,
        ca,
        cl,
        tds,
        mg / 1,                     # Placeholder for mg_li_ratio
        na / k if k else 0,         # na_k_ratio
        ca / mg if mg else 0,       # ca_mg_ratio
        cl / na if na else 0        # cl_na_ratio
    ]
    return np.array(features).reshape(1, -1)

# ✅ Predict button
if st.button("Predict Lithium"):
    X = engineer_features(mg, na, k, ca, cl, tds)
    log_pred = model.predict(X)
    lithium = np.expm1(log_pred)[0]  # Reverse log1p
    st.success(f"🔮 Estimated Lithium Concentration: **{lithium:.2f} mg/L**")
