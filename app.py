from distutils.log import debug
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, render_template
import pickle

# load  model
svc = pickle.load(open('SVC.pkl', 'rb'))
dt = pickle.load(open('DT.pkl', 'rb'))
with open('stacking.pkl', 'rb') as f:
    scv = pickle.load(f)

# app
app = Flask(__name__)

# routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods = ['POST'])
def submit():
    if request.method == 'POST':

        dt1 = request.form['age']
        dt2 = request.form['sex']
        dt3 = request.form['cp']
        dt4 = request.form['trestbps']
        dt5 = request.form['chol']
        dt6 = request.form['fbs']
        dt7 = request.form['restecg']
        dt8 = request.form['thalach']
        dt9 = request.form['exang']
        dt10 = request.form['oldpeak']
        dt11 = request.form['slope']
        dt12 = request.form['ca']
        dt13 = request.form['thal']

    ar = np.array([[dt1,dt2,dt3,dt4,dt5,dt6,dt7,dt8,dt9,dt10,dt11,dt12,dt13]])
    pred = scv.predict(ar)
    return render_template('result.html', data = pred)
    

if __name__ == '__main__':
    app.run(debug = True)