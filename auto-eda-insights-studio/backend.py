#backend.py → EDA, cleaning, SQL, AutoML, forecasting, etc.
import io
import sqlite3
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from difflib import get_close_matches




# -----------------------------
# Column name fuzzy matching
# -----------------------------
def find_best_column_match(user_col: str, df_columns) -> Optional[str]:
    """
    Find the closest matching column name for a user-given label.
    Case-insensitive fuzzy match using difflib.
    """
    if not df_columns:
        return None

    user_col = user_col.lower().strip()
    df_cols_lower = [c.lower() for c in df_columns]

    best = get_close_matches(user_col, df_cols_lower, n=1, cutoff=0.45)
    if not best:
        return None

    matched_lower = best[0]
    index = df_cols_lower.index(matched_lower)
    return df_columns[index]


# -----------------------------
# Auto EDA helpers
# -----------------------------
def show_basic_summary(df: pd.DataFrame) -> None:
    st.subheader("📌 Basic Info")
    st.write(f"Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")

    st.subheader("🧬 Column Types")
    st.write(pd.DataFrame(df.dtypes, columns=["dtype"]))

    st.subheader("📊 Descriptive Statistics (Numeric)")
    numeric_df = df.select_dtypes(include=["int", "float"])
    if not numeric_df.empty:
        st.dataframe(numeric_df.describe().T)
    else:
        st.info("No numeric columns found for statistics.")


def show_missing_values(df: pd.DataFrame) -> None:
    st.subheader("🧩 Missing Values")
    missing = df.isna().sum()
    if (missing > 0).any():
        st.write(missing.to_frame("missing_count"))
        st.bar_chart(missing)
    else:
        st.info("No missing values found.")


def show_correlation_heatmap(df: pd.DataFrame) -> None:
    st.subheader("🔥 Correlation Heatmap")
    numeric_df = df.select_dtypes(include=["int", "float"])
    if numeric_df.shape[1] < 2:
        st.info("Need at least 2 numeric columns for correlation.")
        return

    corr = numeric_df.corr()
    fig, ax = plt.subplots()
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    fig.colorbar(im, ax=ax)
    st.pyplot(fig)


def show_anomaly_detection(df: pd.DataFrame, key_suffix: str = "") -> None:
    st.subheader("🚨 Anomaly Detection (Z-score based)")

    numeric_df = df.select_dtypes(include=["int", "float"])
    if numeric_df.empty:
        st.info("No numeric columns found for anomaly detection.")
        return

    # make sure suffix isn't empty (for unique keys)
    suffix = key_suffix or "default"

    col = st.selectbox(
        "Select numeric column",
        numeric_df.columns,
        key=f"anomaly_column_{suffix}",
    )

    threshold = st.slider(
        "Z-score threshold",
        1.0,
        5.0,
        3.0,
        0.1,
        key=f"anomaly_threshold_{suffix}",
    )

    series = numeric_df[col].dropna()
    mean = series.mean()
    std = series.std()

    if std == 0:
        st.warning("Standard deviation is zero; cannot detect anomalies.")
        return

    z = (series - mean) / std
    anomaly_mask = abs(z) > threshold
    anomalies = df.loc[series.index[anomaly_mask]]
    anomaly_values = series[anomaly_mask]

    st.markdown(f"### 📊 Results for **{col}**")
    st.write(f"Detected **{len(anomalies)}** anomalies with |Z| > **{threshold}**.")

    # Show anomaly table
    if not anomalies.empty:
        st.dataframe(anomalies, use_container_width=True)

        # Download Button
        csv = anomalies.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Anomalies as CSV",
            data=csv,
            file_name=f"{col}_anomalies.csv",
            mime="text/csv",
            key=f"download_anomalies_{suffix}",
        )
    else:
        st.info("No anomalies detected for this threshold.")

    st.markdown("---")
    st.markdown("### 📦 Boxplot (Anomalies Highlighted)")

    # Boxplot
    fig1, ax1 = plt.subplots(figsize=(6, 3))
    sns.boxplot(x=series, ax=ax1, color="#4b8df8")
    if len(anomaly_values) > 0:
        sns.scatterplot(
            x=anomaly_values,
            y=[0] * len(anomaly_values),
            color="red",
            s=90,
            ax=ax1,
            label="Anomalies",
        )
        ax1.legend()
    ax1.set_title(f"Boxplot of {col}")
    st.pyplot(fig1)

    st.markdown("### 📈 Histogram (With Outliers Marked)")

    # Histogram
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    sns.histplot(series, bins=25, kde=True, color="#4b8df8", ax=ax2)
    if len(anomaly_values) > 0:
        sns.scatterplot(
            x=anomaly_values,
            y=[0] * len(anomaly_values),
            color="red",
            s=90,
            ax=ax2,
            label="Anomalies",
        )
        ax2.legend()
    ax2.set_title(f"Distribution of {col} with Outliers Marked")
    st.pyplot(fig2)


def generate_eda_report_text(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    buf.write("AI Data Visualization Agent - Auto EDA Report\n")
    buf.write("=" * 60 + "\n\n")
    buf.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}\n\n")

    buf.write("Column Types:\n")
    buf.write(str(pd.DataFrame(df.dtypes, columns=["dtype"])) + "\n\n")

    numeric_df = df.select_dtypes(include=["int", "float"])
    if not numeric_df.empty:
        buf.write("Descriptive Statistics:\n")
        buf.write(str(numeric_df.describe().T) + "\n\n")

        buf.write("Correlation Matrix:\n")
        buf.write(str(numeric_df.corr()) + "\n\n")
    else:
        buf.write("No numeric columns for statistics/correlation.\n\n")

    missing = df.isna().sum()
    buf.write("Missing Values:\n")
    buf.write(str(missing) + "\n\n")

    return buf.getvalue()


# -----------------------------
# Data cleaning
# -----------------------------
def clean_dataset(df: pd.DataFrame) -> (pd.DataFrame, list):
    df_clean = df.copy()
    report = []

    # Track duplicates
    duplicates = df_clean.duplicated().sum()
    if duplicates > 0:
        df_clean = df_clean.drop_duplicates()
        report.append(f"Removed {duplicates} duplicate rows")

    # Try parsing date/time-like columns
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            try:
                before = df_clean[col].dtype
                df_clean[col] = pd.to_datetime(df_clean[col], errors="ignore")
                after = df_clean[col].dtype
                if before != after:
                    report.append(f"Parsed datetime column: {col}")
            except:
                pass

    # Fill numeric missing values
    numeric_cols = df_clean.select_dtypes(include=["int", "float"]).columns
    for col in numeric_cols:
        missing = df_clean[col].isna().sum()
        if missing > 0:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            report.append(f"Filled {missing} missing numeric values in '{col}' with median")

    # Fill categorical missing values
    cat_cols = df_clean.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        missing = df_clean[col].isna().sum()
        if missing > 0:
            mode_val = df_clean[col].mode().iloc[0]
            df_clean[col] = df_clean[col].fillna(mode_val)
            report.append(f"Filled {missing} missing categorical values in '{col}' with mode")

    if not report:
        report.append("Dataset is already clean. No changes required.")

    return df_clean, report



# -----------------------------
# AutoML + feature importance
# -----------------------------
def auto_train_model(df: pd.DataFrame, target_col: str):

    df = df.copy()

    y = df[target_col]
    X = df.drop(columns=[target_col])

    is_numeric_target = pd.api.types.is_numeric_dtype(y)

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns

    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split, cross_val_score

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    # ---------------- Safe model pool ----------------
    if is_numeric_target:
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

        models = [
            ("LinearRegression", LinearRegression()),
            ("RandomForest", RandomForestRegressor(random_state=42)),
            ("GradientBoosting", GradientBoostingRegressor())
        ]
    else:
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

        models = [
            ("LogisticRegression", LogisticRegression(max_iter=1000)),
            ("RandomForest", RandomForestClassifier(random_state=42)),
            ("GradientBoosting", GradientBoostingClassifier())
        ]

    best_pipeline = None
    best_score = -1
    best_name = ""

    # ---------------- Auto model selection ----------------
    for name, model in models:

        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", model)
        ])

        scores = cross_val_score(pipe, X, y, cv=5)
        score = scores.mean()

        if score > best_score:
            best_score = score
            best_pipeline = pipe
            best_name = name

    # ---------------- Final training ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    best_pipeline.fit(X_train, y_train)
    y_pred = best_pipeline.predict(X_test)

    # ---------------- Metrics ----------------
    if is_numeric_target:
        from sklearn.metrics import mean_squared_error, r2_score

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metric_text = (
            f"Best Model: {best_name}\n"
            f"Regression → MSE: {mse:.3f} | R2: {r2:.3f} | CV R2: {best_score:.3f}"
        )

        if abs(r2 - best_score) > 0.1:
            metric_text += "\n⚠ Possible overfitting detected"

    else:
        from sklearn.metrics import accuracy_score

        acc = accuracy_score(y_test, y_pred)

        metric_text = (
            f"Best Model: {best_name}\n"
            f"Classification → Accuracy: {acc:.3f} | CV Acc: {best_score:.3f}"
        )

        if abs(acc - best_score) > 0.1:
            metric_text += "\n⚠ Possible overfitting detected"

    # ---------------- Save best model ----------------
    try:
        import joblib
        joblib.dump(best_pipeline, "best_model.pkl")
    except:
        pass

    return best_pipeline, metric_text, None



# -----------------------------
# Smart chart recommendation
# -----------------------------
def smart_chart(df: pd.DataFrame, x_col: str, y_col: Optional[str] = None) -> None:
    fig, ax = plt.subplots()

    x_is_num = pd.api.types.is_numeric_dtype(df[x_col])
    y_is_num = pd.api.types.is_numeric_dtype(df[y_col]) if y_col else False

    if y_col is None:
        if x_is_num:
            df[x_col].plot(kind="hist", ax=ax, bins=20)
            ax.set_title(f"Histogram of {x_col}")
        else:
            df[x_col].value_counts().head(20).plot(kind="bar", ax=ax)
            ax.set_title(f"Top categories in {x_col}")
    else:
        if not x_is_num and y_is_num:
            df.groupby(x_col)[y_col].mean().sort_values().plot(kind="bar", ax=ax)
            ax.set_title(f"Average {y_col} by {x_col}")
        elif x_is_num and y_is_num:
            ax.scatter(df[x_col], df[y_col])
            ax.set_title(f"Scatter plot of {y_col} vs {x_col}")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
        elif not x_is_num and not y_is_num:
            df.groupby([x_col, y_col]).size().unstack().plot(kind="bar", ax=ax)
            ax.set_title(f"Counts of {y_col} by {x_col}")
        else:
            df[x_col].value_counts().head(20).plot(kind="bar", ax=ax)
            ax.set_title(f"Value counts of {x_col}")

    st.pyplot(fig)


# -----------------------------
# SQL helpers
# -----------------------------
def create_sqlite_connection(df: pd.DataFrame) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    df.to_sql("data", conn, index=False, if_exists="replace")
    return conn


def run_sql_query(df: pd.DataFrame, query: str) -> Optional[pd.DataFrame]:
    try:
        conn = create_sqlite_connection(df)
        result = pd.read_sql_query(query, conn)
        conn.close()
        return result
    except Exception as e:
        st.error(f"SQL Error: {e}")
        return None


# -----------------------------
# Simple forecasting
# -----------------------------
def show_forecasting(df: pd.DataFrame) -> None:
    st.subheader("📈 Simple Time-Series Forecasting")

    st.markdown(
        "This tool looks at how a value changes over time and draws a simple **trend line** "
        "to guess what future values might look like."
    )

    # Add spacing to prevent CSS overlap
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

    st.markdown("### How this works")

    st.markdown("""
This forecasting is **very simple and beginner-friendly**:

- You select a **date column** (like `Date`, `Month`, `Year`)
- You select a **numeric value** (like `Sales`, `Revenue`, `Temperature`)
- The app fits a **straight trend line** through your past values  
- It then extends that line forward for the number of **forecast steps** you choose  
- This is **not advanced forecasting** — it's a quick way to see where the trend is heading  
    """)




    # 1️⃣ Detect valid date columns by actually trying to parse them
    valid_date_cols = []
    for col in df.columns:
        # Only try on object / string / datetime-like types
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
            try:
                pd.to_datetime(df[col].dropna().iloc[:20], errors="raise")
                valid_date_cols.append(col)
            except Exception:
                continue

    if not valid_date_cols:
        st.info(
            "I couldn't find any valid date columns. "
            "Please make sure at least one column contains actual dates like '2023-01-01'."
        )
        return

    date_col = st.selectbox("🗓️ Select date column", valid_date_cols)

    # 2️⃣ Numeric columns for forecasting
    numeric_df = df.select_dtypes(include=["int", "float"])
    if numeric_df.empty:
        st.info("No numeric columns available for forecasting.")
        return

    value_col = st.selectbox("📌 Select value column", numeric_df.columns)

    horizon = st.slider(
        "⏭️ How many future steps to forecast?",
        min_value=5,
        max_value=60,
        value=15,
        help="If your data is daily → steps = days. Monthly → steps = months. Yearly → steps = years."
    )

    # 3️⃣ Prepare clean time-series data
    temp = df[[date_col, value_col]].dropna().copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp = temp.dropna(subset=[date_col])
    temp = temp.sort_values(by=date_col)

    if temp.shape[0] < 5:
        st.info("Need at least 5 valid rows of date + value data to build a forecast.")
        return

    dates = temp[date_col].values
    y = temp[value_col].values
    x = np.arange(len(y))

    # 4️⃣ Fit a simple linear trend: y = m x + b
    try:
        coeffs = np.polyfit(x, y, 1)
    except Exception:
        st.warning("I wasn't able to fit a trend line on this data.")
        return

    trend = np.poly1d(coeffs)

    # Build future x and predictions
    x_future = np.arange(len(y) + horizon)
    y_future = trend(x_future)

    # 5️⃣ Compute future dates based on average time gap
    deltas = np.diff(dates)
    if len(deltas) == 0:
        # Only one unique date
        step = pd.Timedelta(days=1)
    else:
        step = np.median(deltas)

    last_date = dates[-1]
    future_dates = [last_date + (i + 1) * step for i in range(horizon)]

    # Align future y values
    y_future_hist = y_future[: len(y)]
    y_future_forecast = y_future[len(y):]

    # 6️⃣ Plot nicely
    st.markdown("### 📊 Forecast Chart")

    fig, ax = plt.subplots(figsize=(8, 4))

    # History (actual values)
    ax.plot(temp[date_col], y, label="History (actual)", marker="o", linewidth=1.5)

    # Trend line over history
    ax.plot(
        temp[date_col],
        y_future_hist,
        label="Fitted trend (past)",
        linestyle="--",
        linewidth=1
    )

    # Forecast (future)
    all_future_dates = pd.to_datetime(future_dates)
    ax.plot(
        all_future_dates,
        y_future_forecast,
        label="Forecast (future)",
        linestyle="--",
        marker="o",
    )

    # Vertical line separating past & future
    ax.axvline(x=temp[date_col].iloc[-1], color="grey", linestyle=":", linewidth=1)
    ax.annotate(
        "Forecast starts here",
        xy=(temp[date_col].iloc[-1], y[-1]),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=8,
    )

    ax.set_xlabel("Date")
    ax.set_ylabel(value_col)
    ax.legend()
    ax.grid(True, alpha=0.2)

    st.pyplot(fig)

    # 7️⃣ Forecast table
    st.markdown("### 📋 Forecast Table")

    forecast_df = pd.DataFrame({
        "Date": all_future_dates,
        f"Predicted {value_col}": y_future_forecast
    })

    st.dataframe(forecast_df, use_container_width=True)

    # 8️⃣ Download button
    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download forecast as CSV",
        data=csv,
        file_name=f"forecast_{value_col.lower()}.csv",
        mime="text/csv",
    )

    # 9️⃣ Friendly explanation footer
    st.caption(
        "This is a very simple linear trend forecast. "
        "It assumes the future will continue in roughly the same direction as the past. "
        "It is useful for **quick intuition**, not for precision."
    )


# -----------------------------
# PDF report
# -----------------------------
def create_pdf_report(text: str, filename: str = "eda_report.pdf") -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    lines = text.split("\n")
    x = 40
    y = height - 40

    for line in lines:
        if y < 40:
            c.showPage()
            y = height - 40
        c.drawString(x, y, line[:110])
        y -= 14

    c.save()
    pdf = buffer.getvalue()
    buffer.close()
    return pdf
