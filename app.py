import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Used Car Price Estimator", page_icon="🚗", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif !important; }
.stApp { background-color: #0d1117; }
.header { padding: 2.5rem 0 2rem; text-align: center; border-bottom: 1px solid #21262d; margin-bottom: 2rem; }
.header h1 { color: #e6edf3; font-size: 1.8rem; font-weight: 700; margin: 0 0 0.4rem; letter-spacing: -0.5px; }
.header p  { color: #7d8590; font-size: 0.95rem; margin: 0; }
.card { background: #161b22; border: 1px solid #21262d; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.25rem; }
.card-title { color: #7d8590; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 1.2rem; padding-bottom: 0.75rem; border-bottom: 1px solid #21262d; }
.stSelectbox label, .stSlider label { color: #c9d1d9 !important; font-size: 0.875rem !important; font-weight: 500 !important; }
.stSelectbox > div > div { background: #0d1117 !important; border: 1px solid #30363d !important; border-radius: 8px !important; color: #e6edf3 !important; }
.stButton > button { background: #238636 !important; color: #fff !important; border: 1px solid #2ea043 !important; border-radius: 8px !important; padding: 0.65rem 1.5rem !important; font-size: 0.95rem !important; font-weight: 600 !important; width: 100% !important; }
.stButton > button:hover { background: #2ea043 !important; }
.result-card { background: #0d2818; border: 1px solid #238636; border-radius: 12px; padding: 2rem; text-align: center; margin: 1.5rem 0; }
.result-label { color: #7d8590; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.75rem; }
.result-price { color: #3fb950; font-size: 2.8rem; font-weight: 700; letter-spacing: -1px; line-height: 1; }
.result-range { color: #7d8590; font-size: 0.85rem; margin-top: 0.75rem; }
.result-range span { color: #c9d1d9; font-weight: 600; }
.model-badge { display: inline-block; background: #1f2d3d; border: 1px solid #388bfd; color: #388bfd; font-size: 0.78rem; font-weight: 600; padding: 0.25rem 0.75rem; border-radius: 20px; }
.stats-row { display: flex; gap: 0.75rem; margin: 1rem 0; }
.stat-box { flex: 1; background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 0.85rem; text-align: center; }
.stat-val { color: #388bfd; font-size: 1.1rem; font-weight: 700; }
.stat-lbl { color: #7d8590; font-size: 0.75rem; margin-top: 0.2rem; }
.summary-table { width: 100%; border-collapse: collapse; font-size: 0.875rem; }
.summary-table td { padding: 0.6rem 0.5rem; border-bottom: 1px solid #21262d; }
.summary-table td:first-child { color: #7d8590; }
.summary-table td:last-child  { color: #e6edf3; font-weight: 600; text-align: right; }
.summary-table tr:last-child td { border-bottom: none; }
.error-card { background: #2d1117; border: 1px solid #f85149; border-radius: 8px; padding: 1rem 1.25rem; color: #f85149; font-size: 0.875rem; }
.footer { text-align: center; color: #484f58; font-size: 0.8rem; padding: 2rem 0 1rem; border-top: 1px solid #21262d; margin-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    files = {
        "Random Forest":          "rf_model.pkl",
        "Decision Tree":          "dt_model.pkl",
        "K-Nearest Neighbors":    "knn_model.pkl",
        "Polynomial Regression":  "poly_model.pkl",
    }
    return {name: joblib.load(path) for name, path in files.items() if os.path.exists(path)}

models = load_models()

st.markdown("""
<div class="header">
    <h1>Used Car Price Estimator</h1>
    <p>Enter vehicle specifications to get an estimated market price</p>
</div>
""", unsafe_allow_html=True)

if not models:
    st.markdown('<div class="error-card">No model files found. Place rf_model.pkl, dt_model.pkl, knn_model.pkl, poly_model.pkl next to app.py and restart.</div>', unsafe_allow_html=True)

# ── Model Selection
st.markdown('<div class="card"><div class="card-title">Prediction Model</div>', unsafe_allow_html=True)
selected_model_name = st.selectbox("Model", list(models.keys()) if models else ["Random Forest","Decision Tree","K-Nearest Neighbors","Polynomial Regression"], label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

# ── Vehicle Info
st.markdown('<div class="card"><div class="card-title">Vehicle Information</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1: fuel_label  = st.selectbox("Fuel Type",     ["Petrol", "Diesel", "LPG"])
with c2: transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
c3, c4 = st.columns(2)
with c3: seller_type = st.selectbox("Seller Type",   ["Individual", "Dealer", "Trustmark Dealer"])
with c4: owner       = st.selectbox("Ownership",     ["First Owner","Second Owner","Third Owner","Fourth & Above Owner","Test Drive Car"])
seats = st.selectbox("Seats", [2,4,5,6,7,8,9,10], index=2)
st.markdown('</div>', unsafe_allow_html=True)

# ── Technical Specs
st.markdown('<div class="card"><div class="card-title">Technical Specifications</div>', unsafe_allow_html=True)
km_driven = st.slider("Kilometers Driven",       0,    500000, 60000, 5000,  format="%d km")
engine    = st.slider("Engine Displacement (cc)", 500,  5000,   1500,  50,    format="%d cc")
max_power = st.slider("Max Power (bhp)",          30,   600,    100,   5,     format="%d bhp")
mileage   = st.slider("Mileage (km/l)",           5.0,  45.0,   18.0,  0.1,   format="%.1f km/l")
age       = st.slider("Vehicle Age (years)",      0,    30,     5,     1,     format="%d yrs")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="stats-row">
    <div class="stat-box"><div class="stat-val">{km_driven:,}</div><div class="stat-lbl">KM Driven</div></div>
    <div class="stat-box"><div class="stat-val">{engine} cc</div><div class="stat-lbl">Engine</div></div>
    <div class="stat-box"><div class="stat-val">{max_power} bhp</div><div class="stat-lbl">Max Power</div></div>
    <div class="stat-box"><div class="stat-val">{age} yrs</div><div class="stat-lbl">Age</div></div>
</div>
""", unsafe_allow_html=True)

# ── Predict
if st.button("Estimate Price"):
    input_data = pd.DataFrame([{
        "km_driven":                    km_driven,
        "mileage":                      mileage,
        "engine":                       float(engine),
        "max_power":                    float(max_power),
        "seats":                        float(seats),
        "fuel_Diesel":                  fuel_label == "Diesel",
        "fuel_LPG":                     fuel_label == "LPG",
        "fuel_Petrol":                  fuel_label == "Petrol",
        "seller_type_Individual":       seller_type == "Individual",
        "seller_type_Trustmark Dealer": seller_type == "Trustmark Dealer",
        "transmission_Manual":          transmission == "Manual",
        "owner_Fourth & Above Owner":   owner == "Fourth & Above Owner",
        "owner_Second Owner":           owner == "Second Owner",
        "owner_Test Drive Car":         owner == "Test Drive Car",
        "owner_Third Owner":            owner == "Third Owner",
        "age":                          age,
    }])

    if not models or selected_model_name not in models:
        st.markdown('<div class="error-card">Model not loaded. Check .pkl files.</div>', unsafe_allow_html=True)
        st.stop()

    with st.spinner("Running prediction..."):
        price = float(models[selected_model_name].predict(input_data)[0])
        price = max(0, price)

    p_min = price * 0.93
    p_max = price * 1.07

    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">Estimated Selling Price</div>
        <div class="result-price">&#8377; {price:,.0f}</div>
        <div class="result-range">Confidence range &nbsp;<span>&#8377; {p_min:,.0f}</span> &mdash; <span>&#8377; {p_max:,.0f}</span></div>
        <br><span class="model-badge">{selected_model_name}</span>
    </div>
    <div class="card">
        <div class="card-title">Prediction Summary</div>
        <table class="summary-table">
            <tr><td>Fuel Type</td><td>{fuel_label}</td></tr>
            <tr><td>Transmission</td><td>{transmission}</td></tr>
            <tr><td>Seller Type</td><td>{seller_type}</td></tr>
            <tr><td>Ownership</td><td>{owner}</td></tr>
            <tr><td>Seats</td><td>{seats}</td></tr>
            <tr><td>Kilometers Driven</td><td>{km_driven:,} km</td></tr>
            <tr><td>Engine</td><td>{engine} cc</td></tr>
            <tr><td>Max Power</td><td>{max_power} bhp</td></tr>
            <tr><td>Mileage</td><td>{mileage} km/l</td></tr>
            <tr><td>Vehicle Age</td><td>{age} years</td></tr>
            <tr><td>Model Used</td><td>{selected_model_name}</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="footer">Predictions are estimates based on historical data. Actual prices may vary.</div>', unsafe_allow_html=True)