import pandas as pd
import streamlit as st
import datetime

import pickle

cars_df = pd.read_csv("./churn_logistic.csv")

st.write("""
# AT&T Subscriber Churn Prediction
""")

def model_pred(cust_state, custsvc_calls, intl_plan, day_charge, day_mins):

    ##loading the model
    with open("churn_prediction.pkl", "rb") as file:
        reg_model = pickle.load(file)

    input_features = [[cust_state, custsvc_calls, intl_plan, day_charge, day_mins]]
    return reg_model.predict(input_features)

col1, col2, col3 = st.columns(3)

cust_state = col1.selectbox("State",
                            [  'KS', 'OH', 'NJ', 'OK', 'AL', 'MA', 'MO', 'LA', 'WV', 'IN', 'RI',
                               'IA', 'MT', 'NY', 'ID', 'VT', 'VA', 'TX', 'FL', 'CO', 'AZ', 'SC',
                               'NE', 'WY', 'HI', 'IL', 'NH', 'GA', 'AK', 'MD', 'AR', 'WI', 'OR',
                               'MI', 'DE', 'UT', 'CA', 'MN', 'SD', 'NC', 'WA', 'NM', 'NV', 'DC',
                               'KY', 'ME', 'MS', 'TN', 'PA', 'CT', 'ND'
                            ]
                           )

custsvc_calls = col2.selectbox("CustServ Calls",
                            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

intl_plan = col3.selectbox("Intl. Plan",
                        [0, 1])

day_charge = col1.slider("Day Charge",
                        0.0, 59.64, step=7.5)
day_mins = col2.slider("Day Mins.",
                        0.0, 350.80, step=10.0)


if(st.button("Predict Churn")):
    pr = model_pred(cust_state, custsvc_calls, intl_plan, day_charge, day_mins)
    st.text("Predicted churn value is: "+ str(pr))

st.dataframe(churn.head(2))
