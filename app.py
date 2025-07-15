import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Set app title
st.set_page_config(page_title="Diabetes Prediction App", layout="centered")
st.title("🩺 Diabetes Prediction Web App")
st.write("Enter your health details below to check your diabetes risk.")

# Load the trained model
model = joblib.load("diabetes_model.pkl")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=85)
blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=150, value=70)
skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=79)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=26.0, step=0.1)
diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, step=0.01)
age = st.number_input("Age", min_value=0, max_value=120, value=30)

# Prepare input for prediction
input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                        insulin, bmi, diabetes_pedigree_function, age]])

# Show input values as bar chart
input_dict = {
    'Pregnancies': pregnancies,
    'Glucose': glucose,
    'BloodPressure': blood_pressure,
    'SkinThickness': skin_thickness,
    'Insulin': insulin,
    'BMI': bmi,
    'DPF': diabetes_pedigree_function,
    'Age': age
}
input_df = pd.DataFrame(list(input_dict.items()), columns=['Feature', 'Value'])

st.subheader("📊 Input Feature Chart")
st.bar_chart(input_df.set_index('Feature'))

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)
    
    st.subheader("🔍 Prediction Result")
    if prediction[0] == 1:
        st.error("⚠️ You are **likely** to have Diabetes.")
    else:
        st.success("✅ You are **unlikely** to have Diabetes.")
