from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application= Flask(__name__)
app= application

# initilally interact with the pickle files

ridge_model= pickle.load(open('model/ridge.pkl','rb'))
standard_scaler= pickle.load(open('model/scaler.pkl','rb'))
@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        temperature = float(request.form.get('Temperature'))
        rh = float(request.form.get('RH'))
        ws = float(request.form.get('WS'))
        rain = float(request.form.get('Rain'))
        ffmc = float(request.form.get('FFMC'))
        dmc = float(request.form.get('DMC'))
        isi = float(request.form.get('ISI'))
        classes = float(request.form.get('Classes'))
        region = float(request.form.get('Region'))

        new_data_scaled= standard_scaler.transform([[temperature, rh, ws, rain, ffmc, dmc, isi, classes, region]])
        result= ridge_model.predict(new_data_scaled)
        return render_template('home.html', results=result[0])
    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(debug=True)
