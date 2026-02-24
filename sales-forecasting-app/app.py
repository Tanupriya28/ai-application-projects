from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import io

app = Flask(__name__)

# =====================
# Load data
# =====================

df = pd.read_csv("sales.csv")
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

stores = [int(s) for s in sorted(df["Store"].unique())]

# =====================
# Load models
# =====================

lstm_model = load_model("models/lstm_store1.h5", compile=False)
arima_model = pickle.load(open("models/arima_store1.pkl", "rb"))

LOOKBACK = 3


# =====================
# Helpers
# =====================

def get_store_series(store):
    store = int(store)
    s = df[df["Store"] == store].sort_values("Date")
    return s["Date"].values, s["Weekly_Sales"].values


def make_lstm_forecast(series):
    X = []
    for i in range(len(series) - LOOKBACK):
        X.append(series[i:i + LOOKBACK])

    X = np.array(X).reshape(-1, LOOKBACK, 1)
    preds = lstm_model.predict(X, verbose=0).flatten()

    return np.concatenate([np.full(LOOKBACK, np.nan), preds])


def make_arima_forecast(series):
    return arima_model.forecast(len(series))


def metrics(y, p):
    mask = ~np.isnan(p)
    y = y[mask]
    p = p[mask]
    rmse = np.sqrt(np.mean((y - p) ** 2))
    mae = np.mean(np.abs(y - p))
    return round(float(rmse), 2), round(float(mae), 2)


def find_anomalies(y, p):
    mask = ~np.isnan(p)
    res = y[mask] - p[mask]
    return np.where(np.abs(res) > 2 * np.std(res))[0].tolist()


# =====================
# Routes
# =====================

@app.route("/", methods=["GET", "POST"])
def dashboard():

    raw_store = request.form.get("store")
    selected_store = int(raw_store) if raw_store else stores[0]
    selected_model = request.form.get("model", "LSTM")

    dates, sales = get_store_series(selected_store)

    if selected_model == "LSTM":
        forecast = make_lstm_forecast(sales)

    elif selected_model == "ARIMA":
        forecast = make_arima_forecast(sales)

    else:
        forecast = np.nanmean(
            np.vstack([
                make_lstm_forecast(sales),
                make_arima_forecast(sales)
            ]),
            axis=0
        )

    rmse, mae = metrics(sales, forecast)
    anomalies = find_anomalies(sales, forecast)

    # ============ NEW FEATURES ============

    residual = sales - np.nan_to_num(forecast)
    std = np.std(residual)

    lower = forecast - 1.96 * std
    upper = forecast + 1.96 * std

    comparison = (
        df.groupby("Store")["Weekly_Sales"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
        .round(0)
        .to_dict()
    )

    trend = "Increasing" if np.polyfit(range(len(sales)), sales, 1)[0] > 0 else "Decreasing"
    volatility = round(float(np.std(sales)), 2)
    peak_week = int(np.argmax(sales))
    peak_value = int(np.max(sales))

    return render_template(
        "dashboard.html",
        stores=stores,
        selected_store=int(selected_store),
        selected_model=str(selected_model),
        dates=pd.to_datetime(dates).astype(str).tolist(),
        actual=sales.astype(float).tolist(),
        forecast=np.nan_to_num(forecast).astype(float).tolist(),
        lower=np.nan_to_num(lower).astype(float).tolist(),
        upper=np.nan_to_num(upper).astype(float).tolist(),
        rmse=rmse,
        mae=mae,
        anomalies=anomalies,
        comparison=comparison,
        trend=trend,
        volatility=volatility,
        peak_week=peak_week,
        peak_value=peak_value
    )


@app.route("/download")
def download():
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype="text/csv",
        download_name="sales_data.csv",
        as_attachment=True
    )


# =====================
# Run
# =====================

if __name__ == "__main__":
    app.run(debug=True)