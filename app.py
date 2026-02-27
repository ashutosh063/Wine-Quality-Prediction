import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. Load the Model and Scaler ---
# We use try-except to handle errors just in case the files are missing
try:
    model = joblib.load('wine_model.pkl')
    scaler = joblib.load('wine_scaler.pkl')
except FileNotFoundError:
    st.error("Error: Model files not found. Please ensure 'wine_model.pkl' and 'wine_scaler.pkl' are in the same folder as this script.")
    st.stop()

# --- 2. Build the UI ---
st.title("🍷 Wine Quality Predictor")
st.write("""
This app predicts the quality of wine (Low, Medium, or High) based on its chemical properties. 
Adjust the sliders below and click 'Predict' to see the result!
""")

st.header("Input Chemical Properties")

# Create two columns to make the UI look organized
col1, col2 = st.columns(2)

# Column 1 inputs
with col1:
    fixed_acidity = st.slider('Fixed Acidity', 4.0, 16.0, 8.3)
    volatile_acidity = st.slider('Volatile Acidity', 0.1, 2.0, 0.5)
    citric_acid = st.slider('Citric Acid', 0.0, 1.0, 0.3)
    residual_sugar = st.slider('Residual Sugar', 0.9, 15.0, 2.5)
    chlorides = st.slider('Chlorides', 0.01, 0.6, 0.08)
    free_sulfur_dioxide = st.slider('Free Sulfur Dioxide', 1.0, 72.0, 14.0)

# Column 2 inputs
with col2:
    total_sulfur_dioxide = st.slider('Total Sulfur Dioxide', 6.0, 289.0, 46.0)
    density = st.slider('Density', 0.9900, 1.0040, 0.9960, step=0.0001, format="%.4f")
    pH = st.slider('pH', 2.7, 4.0, 3.3)
    sulphates = st.slider('Sulphates', 0.3, 2.0, 0.65)
    alcohol = st.slider('Alcohol (%)', 8.0, 15.0, 10.4)

# --- 3. Prediction Logic ---
if st.button("🔮 Predict Wine Quality"):
    # Gather the inputs into a dataframe (must match the order of training data!)
    input_data = pd.DataFrame([[
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
        pH, sulphates, alcohol
    ]], columns=[
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol'
    ])
    
    # Scale the user input using the saved scaler
    scaled_input = scaler.transform(input_data)
    
    # Make the prediction
    prediction = model.predict(scaled_input)[0]
    
    # Display the result with some styling
    st.subheader("Prediction Result:")
    if prediction == 'High':
        st.success(f"🌟 Excellent! This wine is predicted to be: **{prediction} Quality**")
    elif prediction == 'Medium':
        st.info(f"🍷 Not bad! This wine is predicted to be: **{prediction} Quality**")
    else:
        st.error(f"⚠️ Warning! This wine is predicted to be: **{prediction} Quality**")