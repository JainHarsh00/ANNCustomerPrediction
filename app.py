import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# Set page config
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

# Load model and preprocessing objects
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Custom CSS
st.markdown("""
    <style>
        html, body {
            background-color: #111;
            color: #f0f0f0;
            font-family: 'Segoe UI', sans-serif;
        }
        .stButton>button {
            background-color: #f9c74f;
            color: black;
            font-size: 18px;
            border-radius: 10px;
            padding: 8px 16px;
        }
        .stButton>button:hover {
            background-color: #ffd166;
        }
        .stNumberInput input, .stSelectbox div[role="button"] {
            background-color: #222 !important;
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center; color: #f9c74f;'>🧮 Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown("#### Enter customer details below:")

# Inputs (single column)
geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('👫 Gender', label_encoder_gender.classes_)
age = st.slider('🎂 Age', 18, 92)
credit_score = st.number_input('💳 Credit Score')
balance = st.number_input('💰 Balance')
estimated_salary = st.number_input('💼 Estimated Salary')
tenure = st.slider('📆 Tenure (years with bank)', 0, 10)
num_of_products = st.slider('🛍️ Number of Products', 1, 4)
has_cr_card = st.selectbox('💳 Has Credit Card?', [0, 1])
is_active_member = st.selectbox('🟢 Is Active Member?', [0, 1])

# Submit button
if st.button("🔍 Submit & Predict"):
    # Preprocess
    input_df = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
    input_df = pd.concat([input_df.reset_index(drop=True), geo_df], axis=1)

    # Scale and predict
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0][0]

    # Display output
    st.subheader(f"📊 Churn Probability: `{prediction:.2f}`")
    if prediction > 0.5:
        st.error("⚠️ The customer is **likely to churn**.")
    else:
        st.success("✅ The customer is **not likely to churn**.")
