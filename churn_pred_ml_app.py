import pandas as pd
import streamlit as st
import datetime
import pickle
import numpy as np  # Ensure you import NumPy

churn = pd.read_csv("./churn_logistic.csv")

st.write("""
# Telco churn
""")

def model_pred(day_mins, eve_mins, night_mins, custsvc_calls, intl_plan, account_length):
    # Convert input parameters to numeric types directly in Streamlit widgets
    
    # Load the model
    with open("churn_prediction.pkl", "rb") as file:
        reg_model = pickle.load(file)
    
    # Ensure the order of features matches the model's training order
    input_features = np.array([[day_mins, eve_mins, night_mins, custsvc_calls, intl_plan, account_length]])
    
    return reg_model.predict(input_features)

col1, col2 = st.columns(2)

with col1:
    custsvc_calls = st.selectbox("Calls made to customer service: ", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

with col2:
    intl_plan = st.selectbox("Does the subscriber have an Intl. Plan?", [0, 1])

day_mins = st.slider("Day Minutes", 0.0, 350.80, step=10.0)
eve_mins = st.slider("Eve Minutes", 0.0, 59.64, step=7.5)
night_mins = st.slider("Night Minutes", 0.0, 59.64, step=7.5)
account_length = st.slider("Age of user account(in months): ", 1, 243, step=4)

if st.button("Predict Churn"):
    pr = model_pred(day_mins, eve_mins, night_mins, custsvc_calls, intl_plan, account_length)
    if pr == 0:
        st.subheader("This subscriber is unlikely to churn!")
    else:
        st.subheader("This subscriber may be considering exiting the service!")
st.dataframe(churn.head(2))
