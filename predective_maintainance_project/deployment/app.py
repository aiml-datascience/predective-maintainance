import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="sasipriyank/predectivemodel", filename="best_predective_model.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Predective Maintainencen App")
st.write("The Predective Maintainencen App is an internal tool for customer that predicts whether Machine sensor is failed or not.")
st.write("Kindly enter the customer details to check whether they are likely to purchase or not.")

# Collect user input

LuboilPressure = st.number_input("Lub oil pressure", min_value=0.0, value=2.493592)
EngineRpm = st.number_input("Engine rpm", min_value=0, value=700)
FuelPressure = st.number_input("Fuel pressure", min_value=0.0, value=11.790927)
CoolantPressure = st.number_input("Coolant pressure", min_value=0.0, value=3.178981)
LuboilTemp = st.number_input("Lub oil temp", min_value=0.0, value=84.144163)
CoolantTemp = st.number_input("Coolant temp", min_value=0.0, value=81.632187)

# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'Lub oil pressure': LuboilPressure,
    'Engine rpm': EngineRpm,
    'Fuel pressure': FuelPressure,
    'Coolant pressure': CoolantPressure,
    'lub oil temp': LuboilTemp,
    'Coolant temp': CoolantTemp

}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "Failed" if prediction == 1 else "NotFailed"
    st.write(f"Based on the information provided, the sensor is  {result} ")
