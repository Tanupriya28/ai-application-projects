# Sales Forecasting Dashboard (LSTM + ARIMA + Ensemble)

## Project Overview

This project implements an **end-to-end time series forecasting system** to predict weekly retail sales using both deep learning and statistical models.  

It integrates **LSTM neural networks, ARIMA models, and ensemble forecasting** into an interactive web dashboard for business-friendly analysis and decision-making.

The system allows users to explore sales trends, compare forecasting models, detect anomalies, and generate insights through an interactive visualization interface.

---

## Objective

- Predict weekly sales for retail stores
- Compare deep learning and statistical forecasting models
- Evaluate forecasting performance
- Detect unusual sales behavior (anomalies)
- Visualize predictions with uncertainty estimates
- Deliver insights through an interactive dashboard

---

## Dataset

The dataset contains **weekly sales data for 45 retail stores** including:

- Store ID
- Weekly Sales
- Temperature
- Fuel Price
- CPI
- Unemployment
- Holiday indicators
- Date (weekly timestamps)

Total records: **~6,400 rows**

---

## Tech Stack

- **Python**
- **Pandas, NumPy** (data processing)
- **TensorFlow / Keras** (LSTM deep learning model)
- **Statsmodels** (ARIMA time series model)
- **Flask** (backend web application)
- **Bootstrap** (UI styling)
- **Chart.js** (interactive visualizations)

---

## Models Implemented

### 1. LSTM (Long Short-Term Memory)

- Sequence-based deep learning model
- Captures long-term temporal dependencies
- Handles nonlinear demand patterns
- Requires scaled input and sliding time windows

---

### 2. ARIMA

- Classical statistical forecasting model
- Uses autoregressive and moving-average components
- Effective for smaller time series datasets
- Captures short-term trend patterns

Model format:
ARIMA(p, d, q)


Where:

- **p** → autoregressive order  
- **d** → differencing order  
- **q** → moving average order  

---

### 3. Ensemble Forecast

The ensemble model combines predictions from both models:


Ensemble = (ARIMA + LSTM) / 2


Benefits:

- Reduces variance
- Improves stability
- Combines strengths of both models

---

## Performance Metrics

Model performance is evaluated using:

### RMSE (Root Mean Squared Error)

Measures the magnitude of prediction errors.


RMSE = sqrt(mean((actual − predicted)^2))


---

### MAE (Mean Absolute Error)

Measures the average absolute difference between predictions and actual values.


MAE = mean(|actual − predicted|)


These metrics allow comparison of forecasting accuracy across models.

---

## Anomaly Detection

The system identifies **unusual sales behavior** using a statistical approach.

A **z-score method** is applied:


z = (sales − mean_sales) / standard_deviation


If:


|z| > threshold


the week is flagged as an anomaly.

This helps identify:

- Promotional spikes
- Holiday demand surges
- Supply chain disruptions
- Unexpected drops in sales

---

## Forecast Confidence Interval

Forecast uncertainty is visualized using confidence bands:


Forecast ± 1.96 × residual standard deviation


This represents a **95% confidence interval**, helping users understand prediction reliability.

---

## Dashboard Features

The interactive dashboard allows users to:

- Select **store-specific forecasts**
- Switch between **LSTM, ARIMA, and Ensemble models**
- Explore historical data using a **time slider**
- Visualize **actual vs predicted sales**
- View **forecast confidence intervals**
- Detect **anomalous sales weeks**
- Compare **store-level sales performance**
- View **RMSE and MAE metrics**
- Export dataset as **CSV**

---

## Business Insights Generated

The dashboard provides actionable insights including:

- Sales trend direction
- Demand volatility measurement
- Peak sales period identification
- Store performance comparison
- Model performance evaluation
- Forecast uncertainty awareness

These insights help support **inventory planning and demand forecasting decisions**.

---

## How to Run Locally

- Clone the repository:

git clone https://github.com/Tanupriya28/applied-ai-ml-systems.git

- Navigate to the project folder:

cd applied-ai-ml-systems/sales-forecasting-app

- Create virtual environment:

python -m venv venv

- Activate environment:

Windows:

venv\Scripts\activate

- Install dependencies:

pip install -r requirements.txt

- Run the Flask app:

python app.py

- Open in browser:

http://127.0.0.1:5000

## Key Learnings

- Time series forecasting techniques

- Deep learning vs statistical models

- Lag feature engineering

- Forecast evaluation metrics

- Model ensembling strategies

- Anomaly detection methods

- ML model deployment using Flask

- Interactive dashboard development

## Conclusion

- This project demonstrates a complete machine learning workflow:

- Data preprocessing

- Time series modeling

- Model comparison

- Anomaly detection

- Forecast evaluation

- Deployment through an interactive dashboard

The system bridges machine learning and business analytics, enabling users to explore demand patterns and make more informed operational decisions.




