# 🔧 AssetHealth – Predictive Maintenance System (2024 Project)

> ⚠️ Note: This project was originally developed in 2024 as a collaborative academic project.  
> The original repository was private/forked. It has now been re-uploaded publicly for portfolio and documentation purposes.

---

## 📌 Overview

AssetHealth is a real-time Predictive Maintenance system built to monitor industrial rotary equipment using vibration analysis and machine learning.

The system collects vibration signals from MPU6050 sensors via ESP32, processes them using signal transformation techniques, and predicts machine health conditions using a hybrid ML model (XGBoost + SVM). A React-based dashboard visualizes machine status, FFT signals, and severity levels based on ISO vibration standards.

---

## 🏗️ System Architecture
MPU6050 Sensor → ESP32 → Google Sheets (Cloud Storage)
↓
Flask Backend (Preprocessing + ML Inference)
↓
React Frontend Dashboard (Visualization + Alerts)


---

## 🚀 Features

- 📡 Real-time vibration data ingestion
- 🔄 Automatic data fetching from cloud (Google Sheets)
- 🧮 Wavelet/Signal preprocessing & FFT feature extraction
- 🤖 Hybrid ML Model (XGBoost + SVM)
- 📊 FFT frequency visualization
- 📈 ISO vibration severity classification
- 🌐 Full-stack integration (Flask + React)

---

## 🧠 Machine Learning Pipeline

1. Raw vibration signals (X, Y, Z axes)
2. Missing value handling (SimpleImputer)
3. FFT transformation (frequency domain features)
4. Feature combination (time + frequency domain)
5. XGBoost model prediction
6. Output classification:
   - Healthy
   - Faulty
7. ISO severity categorization:
   - Good
   - Satisfactory
   - Alert
   - Danger

---

## 📊 Severity Classification (ISO Based)

The system classifies vibration levels using ISO vibration standards depending on machine power class:

- Class 1 (Small machines)
- Class 2 (Medium machines)
- Class 3 (Large machines)

Severity Levels:
- 🟢 Good
- 🟡 Satisfactory
- 🟠 Alert
- 🔴 Danger

---
## 📷 Interface Preview

The screenshots displayed in this repository represent the application interface and sample outputs.

> Note: The values shown in the interface are for demonstration purposes.  
> When connected to the MPU6050 sensor and ESP32 module, the system automatically updates in real-time based on live vibration data streamed to the backend.

## 🛠️ Tech Stack

### Backend
- Python 3.10
- Flask
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SciPy

### Frontend
- React.js
- JavaScript
- CSS

### Hardware
- ESP32
- MPU6050 Vibration Sensor

---
________________________________________
## 📈 Sample API Endpoints
•	/predict → Returns machine health status
•	/get_severity → Returns ISO severity classification
•	/fft → Returns FFT values for graph visualization
________________________________________
## 📌 Project Status
This repository represents the deployment and integration version of the system.
The original training pipeline was implemented collaboratively in Google Colab.
Future improvements may include:
•	Full retraining pipeline documentation
•	Cloud deployment (AWS/Azure)
•	Database integration instead of Google Sheets
•	Real-time streaming via MQTT
________________________________________
## 🤝 Collaboration Note
This project was developed as part of a team in 2024.
My contributions focused on:
•	Backend API integration
•	Signal preprocessing logic
•	ML inference pipeline integration
•	Full-stack system deployment
•	Dashboard connectivity

