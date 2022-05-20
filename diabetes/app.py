#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 18:37:48 2022

@author: William Engels
"""

# Imports

from flask import Flask, render_template, request
import pickle 
import pandas as pd

# Classifiers for Type Checking
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# Setup

app = Flask(__name__)
model = pickle.load(open('models/model.sav','rb')) #read mode

# Establish samples for type comparison.

rfc_sample = RandomForestClassifier()
knn_sample = KNeighborsClassifier()
xgb_sample = XGBClassifier()

def take_input() -> list:

    # Gather user data from form.
    
    pregnancies = int(request.form["pregnancies"])
    glucose = float(request.form["glucose"])
    skinthickness = float(request.form["skinthickness"])
    dpf = float(request.form["diabetespedigreefunction"])
    bloodpressure = float(request.form["bloodpressure"])
    insulin = float(request.form["insulin"])
    bmi = float(request.form["bmi"])
    age = int(request.form["age"])
    
    # Return vector with relevant data points, (1,8) for not XGB, (8) for XGB
    
    if model_type is type(xgb_sample):
        print("XGBoost Model Detected!")
        user_data = {'Pregnancies': pregnancies,
                     'Glucose': glucose,
                     'BloodPressure': bloodpressure,
                     'SkinThickness': skinthickness,
                     'Insulin': insulin,
                     'BMI': bmi,
                     'DiabetesPedigreeFunction': dpf,
                     'Age': age}
        
        input_cols = pd.DataFrame([user_data])
        return input_cols
    
    else: 
        print("Non-XGB Model Detected!")
        input_cols_xgb = [[pregnancies, glucose, bloodpressure, skinthickness,
                   insulin, bmi, dpf, age]]
        return input_cols_xgb

model_type = type(model)

@app.route("/")
def home():
    return render_template('home.html')


@app.route("/predict", methods=['GET','POST'])

def predict():
    if request.method == 'POST':
        
        input_cols = take_input()
        prediction = model.predict(input_cols)
        
        if prediction[0] == 1:
            output = 'Positive'
            
        if prediction[0] == 0:
            output = 'Negative'

        return render_template("home.html", prediction_text=
                               'Your predicted status is {}!'.format(output))

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=8080)