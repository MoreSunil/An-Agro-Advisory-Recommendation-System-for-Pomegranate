import numpy as np
import matplotlib.pylab as plt
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    if output==1:
        Result="Yes"
    else:
        Result="No"

    return render_template('index.html', prediction_text='Probable disease attack on crop as per current conditions is {}'.format(Result))


if __name__ == "__main__":
    app.run(debug=True)