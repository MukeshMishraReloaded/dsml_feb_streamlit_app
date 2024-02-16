import pandas as pd
import streamlit as st
import datetime
import pickle
import numpy as np  # Ensure you import NumPy

churn = pd.read_csv("./telco_churn_data.csv")

st.write("""
# Telco churn prediction
""")

def model_pred(SeniorCitizen, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity,  OnlineBackup, TechSupport, StreamingMovies, Contract, PaperlessBilling, PaymentMethod)
):
    # Convert input parameters to numeric types directly in Streamlit widgets
    
    # Load the model
    with open("telco_churn_prediction.pkl", "rb") as file:
        pipeline = pickle.load(file)
 
    # Ensure the order of features matches the model's training order
    features = ['SeniorCitizen', 'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',  'OnlineBackup', 'TechSupport', 'StreamingMovies','Contract', 'PaperlessBilling', 'PaymentMethod']
    feature_vals = np.array([[SeniorCitizen, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity,  OnlineBackup, TechSupport, StreamingMovies, Contract, PaperlessBilling, PaymentMethod]])
    # creating the dataframe 
    X_sample = pd.DataFrame(data=feature_vals,  
                  columns = features) 
    for col in categorical_features:
        X_sample[col] = X_sample[col].astype('category')
    
    #Prediction logic
    predicted_probabilities = pipeline.predict_proba(X_sample)
    #print(predicted_probabilities[:, 1])
    predicted_classes = (predicted_probabilities[:, 1] >= best_threshold).astype(int)
    #print(best_threshold)
    return predicted_classes[0]

col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11 = st.columns(11)

with col1:
    SeniorCitizen = st.selectbox("Is the subscriber senior citizen? ", [0, 1])

with col2:
    PhoneService = st.selectbox("Does the subscriber have a Phone service?", [0, 1])

with col3:
    MultipleLines = st.selectbox("Does the subscriber have multiple phone lines?", ['No', 'No phone service', 'Yes'])

with col4:
    InternetService = st.selectbox("Does the subscriber have internet service?", ['DSL', 'Fiber optic', 'No'])

with col5:
    OnlineSecurity = st.selectbox("Does the subscriber have Online Security service?", ['No', 'No internet service', 'Yes'])

with col6:
    OnlineBackup = st.selectbox("Does the subscriber have Online backup service?", ['No', 'No internet service', 'Yes'])

with col7:
    TechSupport = st.selectbox("Does the subscriber have Tech Support service?", ['No', 'No internet service', 'Yes'])

with col8:
    StreamingMovies = st.selectbox("Does the subscriber have Streaming Movies service?", ['No', 'No internet service', 'Yes'])

with col9:
    Contract = st.selectbox("What type of contract does the subscriber have?", ['Month-to-month', 'One year', 'Two year'])
    
with col10:
    PaperlessBilling = st.selectbox("Does the subscriber have paperless billing?", [0, 1])
    
with col11:
    PaymentMethod = st.selectbox("What type of payment method does the subscriber have?", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    
tenure = st.slider("Lenght of duration of the account with the service provider(in months): ", 1, 243, step=4)

if st.button("Predict Churn"):
    pr = model_pred(SeniorCitizen, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity,  OnlineBackup, TechSupport, StreamingMovies, Contract, PaperlessBilling, PaymentMethod)
    if pr == 0:
        st.subheader("This subscriber is unlikely to churn!")
    else:
        st.subheader("This subscriber may be considering exiting the service!")
#st.dataframe(churn.head(2))
