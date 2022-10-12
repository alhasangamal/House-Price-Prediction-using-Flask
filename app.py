# Import Libraries
from flask import Flask, render_template, request
import numpy as np
import pickle

# Creat app
app = Flask(__name__)

# Load model
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    bed_val = request.form['bedrooms']
    bath_val = request.form['bathrooms']
    floor_val = request.form['floors']
    year_val = request.form['yr_built']
    arr = np.array([bed_val, bath_val, floor_val, year_val]).astype(np.float64)
    pred = model.predict([arr])

    return render_template('index.html', data=int(pred), bed_val=bed_val, bath_val=bath_val, floor_val=floor_val,
                           yr_built=year_val)

@app.route('/aboutdata')
def aboutdata():
    return render_template('aboutdata.html')
if __name__ == '__main__':
    app.run(debug=True)
