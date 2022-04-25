#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 18:37:48 2022

@author: William Engels
"""

from flask import Flask, render_template, request
import pickle 
app = Flask(__name__)
model = pickle.load(open('models/model.sav','rb')) #read mode

@app.route("/")
def home():
    return render_template('home.html')


@app.route("/predict", methods=['GET','POST'])


def predict():
    if request.method == 'POST':
        #access the data from form

        pregnancies = int(request.form["pregnancies"])
        glucose = float(request.form["glucose"])
        skinthickness = float(request.form["skinthickness"])
        dpf = float(request.form["diabetespedigreefunction"])
        bloodpressure = float(request.form["bloodpressure"])
        insulin = float(request.form["insulin"])
        bmi = float(request.form["BMI"])
        age = int(request.form["age"])
        
        #get prediction
        input_cols = [[pregnancies, glucose, bloodpressure, skinthickness,
                       insulin, bmi, dpf, age]]
        
        prediction = model.predict(input_cols)
        
        if prediction[0] == 1:
            output = 'Positive'
            
        if prediction[0] == 0:
            output = 'Negative'

        return render_template("home.html", prediction_text=
                               'Your predicted status is {}'.format(output))

if __name__ == "__main__":
    app.run(debug=False, port=5001)