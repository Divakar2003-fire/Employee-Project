import streamlit as st
import pandas as pd 
import pickle

# Load all models
lr = pickle.load(open("logistic_model.pkl", "rb"))
rf = pickle.load(open("random_forest_model.pkl", "rb"))
xgb = pickle.load(open("xgboost_model.pkl", "rb"))
lgb = pickle.load(open("lightgbm_model.pkl", "rb"))

scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Employee Retention Prediction (All Models)")

# Model Selection
model_choice = st.selectbox(
    "Choose Model",
    ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]
)

st.write("Enter Employee Details:")

# Inputs
city_dev = st.number_input("City Development Index", 0.0, 1.0)
training_hours = st.number_input("Training Hours", 0, 500)
experience = st.number_input("Experience", 0,25)
last_job = st.number_input("Last New Job Gap", 0, 5)

# Prediction
if st.button("predict"):

    input_data = pd.DataFrame({
        'city_development_index': [city_dev],
        'training_hours': [training_hours],
        'experience': [experience],
        'last_new_job": [last_job]
    })

    # Scale
    input scaled = scaler.transform(input_data)

    # Select Model
    if model_choice == "Logistic Regression":
        model = lr
    elif model_choice == "Random Forest":
        model = rf
    elif model_choice == "XGBoost":
        model = xgb
    else:
        model = lgb

    # Prediction
    prediction = model.predict(input_scaled)

    # Output
    if prediction[0] == 1:
        st.error(f"({model_choice}) Employee likely to leave")
    else:
        st.success(f"({model_choice}) Employee likely to stay")
    