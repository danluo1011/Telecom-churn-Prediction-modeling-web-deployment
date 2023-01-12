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
    Senior Citizen = request.form.get('Senior Citizen')
    Partner = request.form.get('Partner')
    Dependents = request.form.get('Dependents')
    Paperless Billing = request.form.get('Paperless Billing')
    Monthly Charges = request.form.get('Monthly Charges')
    Total Charges = request.form.get('Total Charges')
    CLTV = request.form.get('CLTV')
    los_angeles_distance = request.form.get('los_angeles_distance')
    san_francisco_distance = request.form.get('san_francisco_distance')
    Loyalty = request.form.get('Loyalty')
    not_Help_Score = request.form.get('not_Help_Score')
    Multiple Lines_No = request.form.get('Multiple Lines_No')
    Multiple Lines_No phone service = request.form.get('Multiple Lines_No phone service')
    Multiple Lines_Yes = request.form.get('Multiple Lines_Yes')
    Internet Service_DSL = request.form.get('Internet Service_DSL')
    Internet Service_Fiber optic = request.form.get('Internet Service_Fiber optic')
    Internet Service_No = request.form.get('Internet Service_No')
    Online Security_No = request.form.get('Online Security_No')
    Online Security_Yes = request.form.get('Online Security_Yes')
    Online Backup_No = request.form.get('Online Backup_No')
    Online Backup_Yes = request.form.get('Online Backup_Yes')
    Device Protection_No = request.form.get('Device Protection_No')
    Device Protection_Yes = request.form.get('Device Protection_Yes')
    Tech Support_No = request.form.get('Tech Support_No')
    Tech Support_Yes = request.form.get('Tech Support_Yes')
    Streaming TV_No = request.form.get('Streaming TV_No')
    Streaming TV_Yes = request.form.get('Streaming TV_Yes')
    Streaming Movies_No = request.form.get('Streaming Movies_No')
    Streaming Movies_Yes = request.form.get('Streaming Movies_Yes')
    Contract_Month-to-month = request.form.get('Contract_Month-to-month')
    Contract_One year = request.form.get('Contract_One year')
    Contract_Two year = request.form.get('Contract_Two year')
    Payment Method_Bank transfer (automatic) = request.form.get('Payment Method_Bank transfer (automatic)')
    Payment Method_Credit card (automatic) = request.form.get('Payment Method_Credit card (automatic)')
    Payment Method_Electronic check = request.form.get('Payment Method_Electronic check')
    Payment Method_Mailed check = request.form.get('Payment Method_Mailed check')
    
    d_dict = {'Senior Citizen': [Senior Citizen], 'Partner': [Partner], 'Dependents': [Dependents],
              'Paperless Billing': [Paperless Billing], 'Monthly Charges': [Monthly Charges], 'Total Charges': [Total Charges],
              'CLTV': [CLTV], 'los_angeles_distance': [los_angeles_distance], 'san_francisco_distance': [san_francisco_distance],
              'Loyalty': [Loyalty], 'not_Help_Score': [not_Help_Score], 'Multiple Lines_No': [Multiple Lines_No],
              'Multiple Lines_No phone service': [Multiple Lines_No phone service], 'Multiple Lines_Yes': [Multiple Lines_Yes],
              'Internet Service_DSL': [Internet Service_DSL],'Internet Service_Fiber optic': [Internet Service_Fiber optic],
              'Internet Service_No': [Internet Service_No], 'Online Security_No': [Online Security_No], 'Online Security_Yes': [Online Security_Yes],
              'Online Backup_No': [Online Backup_No], 'Online Backup_Yes': [Online Backup_Yes],'Device Protection_No': [Device Protection_No],
              'Device Protection_Yes': [Device Protection_Yes], 'Tech Support_No': [Tech Support_No], 'Tech Support_Yes': [Tech Support_Yes],
              'Streaming TV_No': [Streaming TV_No], 'Streaming TV_Yes': [Streaming TV_Yes], 'Streaming Movies_No': [Streaming Movies_No],
              'Streaming Movies_Yes': [Streaming Movies_Yes], 'Contract_Month-to-month': [Contract_Month-to-month], 'Contract_One year': [Contract_One year],
              'Contract_Two year': [Contract_Two year], 'Payment Method_Bank transfer (automatic)': [Payment Method_Bank transfer (automatic)], 
              'Payment Method_Credit card (automatic)': [Payment Method_Credit card (automatic)],
              'Payment Method_Electronic check': [Payment Method_Electronic check],
              'Payment Method_Mailed check': [Payment Method_Mailed check]}
    
    return pd.DataFrame.from_dict(d_dict, orient='columns')

@app.route('/submit', methods=['POST'])
def show_data():
    df = get_data()
    prediction = gradientboost.predict(df)
    outcome = 'Churner'
    if prediction == 0:
        outcome = 'Non-Churner'

    return render_template('results.html', result = outcome)



if __name__=="__main__":
    app.run(debug=True)

