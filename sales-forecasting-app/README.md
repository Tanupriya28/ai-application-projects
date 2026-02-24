# 📈 Sales Forecasting Dashboard (LSTM + ARIMA + Ensemble)

## 🔍 Project Overview

This project implements an end-to-end time series forecasting system to predict weekly retail sales using both deep learning and statistical models. It integrates LSTM neural networks, ARIMA models, and ensemble forecasting into an interactive web dashboard for business-friendly analysis.

---

## 🎯 Objective

- Predict weekly sales for retail stores  
- Compare deep learning and statistical forecasting models  
- Evaluate forecasting performance  
- Visualize predictions with uncertainty estimates  
- Deliver insights through an interactive dashboard  

---

## 🛠️ Tech Stack

- Python  
- Pandas, NumPy  
- TensorFlow (LSTM)  
- Statsmodels (ARIMA)  
- Flask  
- Bootstrap  
- Chart.js  

---

## 📊 Models Implemented

### 🔹 LSTM (Long Short-Term Memory)
- Sequence-based deep learning model  
- Captures long-term temporal dependencies  
- Handles nonlinear demand patterns  

### 🔹 ARIMA
- Classical statistical forecasting model  
- Effective for short-term trend prediction  
- Used as baseline comparison  

### 🔹 Ensemble Forecast
- Average of LSTM and ARIMA predictions  
- Improves stability and reduces variance  

---

## 📈 Performance Metrics

- RMSE (Root Mean Squared Error)  
- MAE (Mean Absolute Error)  

Used to evaluate accuracy across historical data.

---

## 📉 Forecast Confidence Interval

Uncertainty bands computed using:

Forecast ± 1.96 × residual standard deviation

Provides risk-aware forecasting.

---

## 💻 Dashboard Features

- Store-wise filtering  
- Model switching (LSTM / ARIMA / Ensemble)  
- Interactive time slider for historical exploration  
- Forecast vs actual visualization  
- Confidence interval bands  
- RMSE & MAE display  
- Store comparison insights  
- Business summary metrics  
- CSV export functionality  

---

## 🧠 Business Insights Generated

- Sales trend direction  
- Demand volatility measurement  
- Peak sales period identification  
- Model performance comparison  
- Forecast uncertainty awareness  

---

## 🚀 How to Run Locally

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py


Open in browser:
http://127.0.0.1:5000
________________________________________
## 📌 Key Learnings
•	Time series modeling techniques
•	Deep learning vs statistical forecasting
•	Forecast evaluation metrics
•	Model ensembling strategies
•	ML deployment using Flask
•	Interactive data visualization
________________________________________
## Conclusion
This project demonstrates full-cycle machine learning development — from data preprocessing and model building to deployment and business insight generation through an interactive web application.




