#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__, template_folder='webapp')
gradientboost = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template("homepage.html")

def get_data():
    SeniorCitizen = request.form.get('Senior Citizen')
    Partner = request.form.get('Partner')
    Dependents = request.form.get('Dependents')
    PaperlessBilling = request.form.get('Paperless Billing')
    MonthlyCharges = request.form.get('Monthly Charges')
    TotalCharges = request.form.get('Total Charges')
    CLTV = request.form.get('CLTV')
    los_angeles_distance = request.form.get('los_angeles_distance')
    san_francisco_distance = request.form.get('san_francisco_distance')
    Loyalty = request.form.get('Loyalty')
    not_Help_Score = request.form.get('not_Help_Score')
    MultipleLines_No = request.form.get('Multiple Lines_No')
    MultipleLines_Nophoneservice = request.form.get('Multiple Lines_No phone service')
    MultipleLines_Yes = request.form.get('Multiple Lines_Yes')
    InternetService_DSL = request.form.get('Internet Service_DSL')
    InternetService_Fiberoptic = request.form.get('Internet Service_Fiber optic')
    InternetService_No = request.form.get('Internet Service_No')
    OnlineSecurity_No = request.form.get('Online Security_No')
    OnlineSecurity_Yes = request.form.get('Online Security_Yes')
    OnlineBackup_No = request.form.get('Online Backup_No')
    OnlineBackup_Yes = request.form.get('Online Backup_Yes')
    DeviceProtection_No = request.form.get('Device Protection_No')
    DeviceProtection_Yes = request.form.get('Device Protection_Yes')
    TechSupport_No = request.form.get('Tech Support_No')
    TechSupport_Yes = request.form.get('Tech Support_Yes')
    StreamingTV_No = request.form.get('Streaming TV_No')
    StreamingTV_Yes = request.form.get('Streaming TV_Yes')
    StreamingMovies_No = request.form.get('Streaming Movies_No')
    StreamingMovies_Yes = request.form.get('Streaming Movies_Yes')
    Contract_Month_to_month = request.form.get('Contract_Month-to-month')
    Contract_Oneyear = request.form.get('Contract_One year')
    Contract_Twoyear = request.form.get('Contract_Two year')
    PaymentMethod_Banktransfer = request.form.get('Payment Method_Bank transfer (automatic)')
    PaymentMethod_Creditcard = request.form.get('Payment Method_Credit card (automatic)')
    PaymentMethod_Electroniccheck = request.form.get('Payment Method_Electronic check')
    PaymentMethod_Mailedcheck = request.form.get('Payment Method_Mailed check')
    
    d_dict = {'Senior Citizen': [SeniorCitizen], 'Partner': [Partner], 'Dependents': [Dependents],
              'Paperless Billing': [PaperlessBilling], 'Monthly Charges': [MonthlyCharges], 'Total Charges': [TotalCharges],
              'CLTV': [CLTV], 'los_angeles_distance': [los_angeles_distance], 'san_francisco_distance': [san_francisco_distance],
              'Loyalty': [Loyalty], 'not_Help_Score': [not_Help_Score], 'Multiple Lines_No': [MultipleLines_No],
              'Multiple Lines_No phone service': [MultipleLines_Nophoneservice], 'Multiple Lines_Yes': [MultipleLines_Yes],
              'Internet Service_DSL': [InternetService_DSL],'Internet Service_Fiber optic': [InternetService_Fiberoptic],
              'Internet Service_No': [InternetService_No], 'Online Security_No': [OnlineSecurity_No], 'Online Security_Yes': [OnlineSecurity_Yes],
              'Online Backup_No': [OnlineBackup_No], 'Online Backup_Yes': [OnlineBackup_Yes],'Device Protection_No': [DeviceProtection_No],
              'Device Protection_Yes': [DeviceProtection_Yes], 'Tech Support_No': [TechSupport_No], 'Tech Support_Yes': [TechSupport_Yes],
              'Streaming TV_No': [StreamingTV_No], 'Streaming TV_Yes': [StreamingTV_Yes], 'Streaming Movies_No': [StreamingMovies_No],
              'Streaming Movies_Yes': [StreamingMovies_Yes], 'Contract_Month-to-month': [Contract_Month_to_month], 'Contract_One year': [Contract_Oneyear],
              'Contract_Two year': [Contract_Twoyear], 'Payment Method_Bank transfer (automatic)': [PaymentMethod_Banktransfer], 
              'Payment Method_Credit card (automatic)': [PaymentMethod_Creditcard],
              'Payment Method_Electronic check': [PaymentMethod_Electroniccheck],
              'Payment Method_Mailed check': [PaymentMethod_Mailedcheck]}
    
    return pd.DataFrame.from_dict(d_dict, orient='columns')

@app.route('/send', methods=['POST'])
def show_data():
    df = get_data()
    prediction = gradientboost.predict(df)
    outcome = 'Churner'
    if prediction == 0:
        outcome = 'Non-Churner'

    return render_template('result.html', result = outcome)



if __name__=="__main__":
    app.run(debug=True)

