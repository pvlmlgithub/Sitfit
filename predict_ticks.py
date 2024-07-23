from flask import Flask, request, jsonify
import os
import sklearn
from sklearn.ensemble import RandomForestRegressor
import pickle
import numpy as np
from PIL import Image
import pandas as pd
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def upload_file():
    data_predict = request.json
    with open('rf_models.pkl', 'rb') as f:
        models = pickle.load(f)
    rf_lower = models['rf_lower']
    rf_upper = models['rf_upper']
    
    data = pd.DataFrame(data_predict)
    predictions_lower = rf_lower.predict(data)
    predictions_upper = rf_upper.predict(data)
    return {"Predictions for Lower" : predictions_lower.tolist(),
            "Predictions for Upper": predictions_upper.tolist()}

if __name__ == '__main__':
    app.run(debug=True)
