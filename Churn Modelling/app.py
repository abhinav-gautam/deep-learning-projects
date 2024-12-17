import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import pickle

# Load the trained model
model = load_model("model.h5")

# Load the scaler and encoder
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

with open("onehotencoder.pkl", "rb") as file:
    encoder = pickle.load(file)

# Load data
df = pd.read_csv("Churn_Modelling.csv")

# Streamlit app
st.title("Customer Churn Prediction")

# User input
geography = st.selectbox("Geography", df["Geography"].unique())
gender = st.selectbox("Gender", df["Gender"].unique())
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# Prepare the input data
input_data = pd.DataFrame(
    {
        "CreditScore": [credit_score],
        "Gender": [gender],
        "Geography": [geography],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_member],
        "EstimatedSalary": [estimated_salary],
    }
)

# One hot encoding
categorical_columns = ["Geography", "Gender"]
one_hot_encoded = encoder.transform(input_data[categorical_columns])
one_hot_encoded_columns = encoder.get_feature_names_out(categorical_columns)
one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=one_hot_encoded_columns)

# Merging one hot encoded columns with others
input_data = pd.concat([input_data, one_hot_encoded_df], axis=1)
input_data.drop(categorical_columns, axis=1, inplace=True)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f"Churn Probability: {prediction_proba:.2f}")

if prediction_proba > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is not likely to churn.")
