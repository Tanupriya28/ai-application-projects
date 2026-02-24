from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import datetime
from flask_cors import CORS
import joblib
from sklearn.impute import SimpleImputer 
from scipy.fft import fft
from xgboost import XGBClassifier

# Initializing flask app
app = Flask(__name__)
CORS(app)

#-------------------------------------Severity Result---------------------------------------#

iso_chart = {
    '1': {
        "good": (0.0, 0.72),
        "satisfactory": (0.72, 1.81),
        "alert": (1.81, 4.51),
        "danger": (4.51, 45.00)
    },
    '2': {
        "good": (0.0, 1.121),
        "satisfactory": (1.121, 2.81),
        "alert": (2.81, 7.11),
        "danger": (7.11, 45.00)
    },
    '3': {
        "good": (0.0, 1.81),
        "satisfactory": (1.81, 4.51),
        "alert": (4.51, 11.21),
        "danger": (11.21, 45.00)
    }
}

def get_class(power):
    if power <= 15:
        return '1'
    elif power <= 75:
        return '2'
    else:
        return '3'

def label_severity(rms_velocity, c):
    try:
        if c not in iso_chart:
            raise ValueError("Invalid bearing size")
        thresholds = iso_chart[str(c)]
        if thresholds["good"][0] <= rms_velocity < thresholds["good"][1]:
            return "Good"
        elif thresholds["satisfactory"][0] <= rms_velocity < thresholds["satisfactory"][1]:
            return "Satisfactory"
        elif thresholds["alert"][0] <= rms_velocity < thresholds["alert"][1]:
            return "Alert"
        elif thresholds["danger"][0] <= rms_velocity < thresholds["danger"][1]:
            return "Danger"
        else:
            return "Faulty"
    except ValueError as e:
        print(f"Error: {e}")
        return "Unknown"

@app.route('/get_severity', methods=['POST'])
def get_severity():
    try:
        power = int(request.json.get('power'))
        spreadsheet_link = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vRNZq4L-n35gkzhZTF0Qcq28x_xK6hW-AuntcG2HMXLWXycUIBWAWZZGDu4FEX6S-wwYZyM89BVdjxk/pub?output=csv'
        df = pd.read_csv(spreadsheet_link)
        current_velocity = df['Velocity'].iloc[-1]
        severity = label_severity(current_velocity, get_class(power))
        response_data = {
            "current_velocity": current_velocity,
            "severity": severity
        }
        return jsonify(response_data)
    except Exception as e:
        return jsonify({"error": f"Unable to fetch value: {e}"}), 500

#--------------------------------- End of Severity Results methods -----------------------#    





##### To get fft datapoints for graph 

@app.route('/fft',  methods=['GET'])
def calculate_fft():
    spreadsheet_link = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vRNZq4L-n35gkzhZTF0Qcq28x_xK6hW-AuntcG2HMXLWXycUIBWAWZZGDu4FEX6S-wwYZyM89BVdjxk/pub?output=csv'
    df = pd.read_csv(spreadsheet_link)
    d=df[['X']]
    data=df[['X','Y','Z']]
    combined_signal=np.sum(data,axis=1)
    cs=list(combined_signal)
    fft_data = np.abs(fft(cs))
    return jsonify(fft_data.tolist())





#--------------------prediction model and results rendering ------------------#
    

def preprocess_data(data):
    imputer = SimpleImputer(strategy='mean')
    imputer.fit_transform(data)
    imputed_data = imputer.transform(data)
    fft_data = np.abs(fft(imputed_data))
    combined_data = pd.concat([pd.DataFrame(imputed_data, columns=['X', 'Y', 'Z']), pd.DataFrame(fft_data, columns=['FFT_X', 'FFT_Y', 'FFT_Z'])], axis=1)
    return combined_data

model = joblib.load('model.pkl')
xgb_model=joblib.load('xgb_model.pkl')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    spreadsheet_link = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vRNZq4L-n35gkzhZTF0Qcq28x_xK6hW-AuntcG2HMXLWXycUIBWAWZZGDu4FEX6S-wwYZyM89BVdjxk/pub?output=csv'
    df = pd.read_csv(spreadsheet_link)
    # Preprocess the input data
    preprocessed_data = preprocess_data(df[['X', 'Y', 'Z']])
    
    # Make predictions
    predictions = xgb_model.predict(preprocessed_data)
    
    # Map predictions to labels
    labels = ['Healthy' if pred == 0 else 'Faulty' for pred in predictions]
    
    # Prepare response
    response = {'predictions': labels[-1]}
    
    return jsonify(response)

#---------------end of prediction model and results rendering -------------#




# ----------sample to test connection--------------------#
# Route for seeing a data
x = datetime.datetime.now() 
@app.route('/mem')
def get_time():
 
    # Returning an api for showing in  reactjs
    return {
        'Name':"geek", 
        "Age":"22",
        "Date":x, 
        "programming":"python"
        }
#---------------end of sample------------------------# 
     
# Running app
if __name__ == '__main__':
    app.run(debug=True)