import pandas as pd
import streamlit as st
import datetime

import pickle

churn = pd.read_csv("./churn_logistic.csv")

st.write("""
# AT&T Subscriber Churn Prediction
""")

def model_pred(day_mins, eve_mins, night_mins, custsvc_calls, intl_plan, account_length):

    ##loading the model
    with open("churn_prediction.pkl", "rb") as file:
        reg_model = pickle.load(file)

    input_features = [['Day Mins', 'Eve Mins', 'Night Mins', 'CustServ Calls', 'Intl Plan', 'Account Length']]
    return reg_model.predict(input_features)

col1, col2 = st.columns(2)

custsvc_calls = col1.selectbox("CustServ Calls",
                            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

intl_plan = col2.selectbox("Intl. Plan",
                        [0, 1])

day_mins = col1.slider("Day Mins.",
                        0.0, 350.80, step=10.0)
eve_mins = col1.slider("Eve Mins.",
                        0.0, 59.64, step=7.5)
night_mins = col2.slider("Night Mins.",
                        0.0, 59.64, step=7.5)
account_length = col2.slider("Account Length",
                        1, 243, step=4)


if(st.button("Predict Churn")):
    pr = model_pred(custsvc_calls, intl_plan, day_charge, day_mins)
    st.text("Predicted churn value is: "+ str(pr))

st.dataframe(churn.head(2))
