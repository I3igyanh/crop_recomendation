import streamlit as st
import pandas as pd
import joblib

st.title("Crop Recommendation System")

# Load the trained model
try:
    model = joblib.load('crop_recommendation_model.pkl')
    st.write("Model loaded successfully!")
except Exception as e:
    st.write(f"Error loading model: {e}")

# Input fields
try:
    N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=140.0, step=1.0)
    P = st.number_input("Phosphorus (P)", min_value=5.0, max_value=145.0, step=1.0)
    K = st.number_input("Potassium (K)", min_value=5.0, max_value=205.0, step=1.0)
    temperature = st.number_input("Temperature (°C)", min_value=8.0, max_value=45.0, step=0.1)
    humidity = st.number_input("Humidity (%)", min_value=14.0, max_value=99.0, step=0.1)
    ph = st.number_input("Soil pH", min_value=3.5, max_value=9.0, step=0.1)
    rainfall = st.number_input("Rainfall (mm)", min_value=20.0, max_value=300.0, step=1.0)
    st.write("Inputs loaded successfully!")
except Exception as e:
    st.write(f"Error with inputs: {e}")

# Predict button
if st.button('Recommend Crop'):
    try:
        # Preprocess inputs
        input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                                  columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
        
        # Display the input data for verification
        st.write("Input Data:")
        st.write(input_data)

        # Prediction
        prediction = model.predict(input_data)
        st.write(f'The recommended crop is: {prediction[0]}')
    except Exception as e:
        st.write(f"Error during prediction: {e}")
