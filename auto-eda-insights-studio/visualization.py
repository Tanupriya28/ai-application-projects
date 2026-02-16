import os
import re
import sys
import io
import contextlib
import warnings
from typing import Optional, List, Any, Tuple

import base64
from io import BytesIO
import sqlite3

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from groq import Groq
from e2b_code_interpreter import Sandbox

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from difflib import get_close_matches


def find_best_column_match(user_col: str, df_columns):
    """
    Finds the closest matching column name in the dataset
    for the user-specified name. Case-insensitive fuzzy match.
    """
    user_col = user_col.lower().strip()
    df_cols_lower = [c.lower() for c in df_columns]

    # get_close_matches returns a list of best matches
    best = get_close_matches(user_col, df_cols_lower, n=1, cutoff=0.45)

    if not best:
        return None

    matched_lower = best[0]                    # e.g. "gender"
    index = df_cols_lower.index(matched_lower)  # find its original index
    return df_columns[index]                   # return original column name


warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

pattern = re.compile(r"```python\n(.*?)\n```", re.DOTALL)


# --------------------------------
# Execute Python code in E2B sandbox
# --------------------------------
def code_interpret(e2b_code_interpreter: Sandbox,
                   code: str) -> Optional[List[Any]]:
    with st.spinner("Executing code in E2B sandbox..."):
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(
            stderr_capture
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                exec = e2b_code_interpreter.run_code(code)

        if stderr_capture.getvalue():
            print("[Code Interpreter Warning/Error]", file=sys.stderr)
            print(stderr_capture.getvalue(), file=sys.stderr)

        if exec.error:
            print(f"[Code Interpreter ERROR] {exec.error}", file=sys.stderr)
            return None

        return exec.results


# --------------------------------
# Extract Python code from LLM response
# --------------------------------
def match_code_blocks(llm_response: str) -> str:
    match = pattern.search(llm_response)
    return match.group(1) if match else ""


# --------------------------------
# Ask Groq LLM to generate Python code
# --------------------------------
def chat_with_llm(
    e2b_code_interpreter: Sandbox,
    user_message: str,
    dataset_path: str,
    chart_type: str,
) -> Tuple[Optional[List[Any]], str]:

    system_prompt = f"""
You are a senior Python data scientist and visualization expert.

You MUST follow these rules:

────────────────────────────
📌 1. COLUMN NAME HANDLING
────────────────────────────
- Always detect column names dynamically.
- Match user-specified names to the closest column (case-insensitive, fuzzy matching).
- Use the backend helper function: find_best_column_match(user_col, df.columns)
- Do NOT write your own fuzzy-matching algorithm.
- If no close match exists, reply with a friendly error listing available columns.

Example:
User says "gender"
Dataset has ["Gender", "age", "bmi"]
→ You MUST use "Gender"

────────────────────────────
📌 2. CODE STYLE RULES
────────────────────────────
- Return ONLY one clean Python code block.
- No fallback code.
- No repeated alternative solutions.
- Do NOT print df.head().
- Do NOT generate boolean-based grouping hacks.
- Do NOT produce explanatory text inside the code.

Your output MUST be ONLY:

```python
# final clean working code here
```

────────────────────────────
📌 3. CHART SELECTION (AUTO MODE)
────────────────────────────
If user selects "auto", choose the BEST chart type:

categorical vs numeric → bar chart

numeric vs numeric → scatter plot

categorical vs categorical → heatmap or grouped bar

one numeric only → histogram

one categorical only → bar chart

time series → line chart

────────────────────────────
📌 4. VALIDATION RULES
────────────────────────────
For ANY chart:

Ensure columns exist (after name correction).

Ensure numeric required columns are numeric.

If needed, convert categories using df[col].astype(str).


────────────────────────────
📌 4A. COUNTING RULES (IMPORTANT)
────────────────────────────
For ANY question involving “count”, “number of”, or “how many”:

- You MUST use:
      df.groupby(category_column)[target_column].count().reset_index()

- NEVER plot raw columns directly.
- NEVER pass the full dataframe into px.bar() for counts.
- ALWAYS group the data first.

────────────────────────────
📌 5. OUTPUT REQUIREMENTS
────────────────────────────

Use pandas + matplotlib OR plotly.express.

Never call plt.show().

Always return a fig object for Streamlit.

Keep code clean, short, and reliable.

────────────────────────────
📌 6. PLOT REQUIREMENTS
────────────────────────────

Always label axes.

Always add a title.

Keep charts clean and minimal.

────────────────────────────
Dataset path: '{dataset_path}'
Preferred chart type: {chart_type}
"""
    full_user_message = (
        user_message +
        f"\n\n(Preferred chart type from the user: {chart_type}. Use it when it makes sense.)")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": full_user_message},
    ]

    with st.spinner("💬 Querying Groq AI model..."):
        client = Groq(api_key=st.session_state.groq_api_key)
        response = client.chat.completions.create(
            model=st.session_state.model_name,
            messages=messages,
        )

        response_message = response.choices[0].message.content
        python_code = match_code_blocks(response_message)

        if python_code:
            results = code_interpret(e2b_code_interpreter, python_code)
            return results, response_message

        st.warning("⚠ No Python code found in LLM response.")
        return None, response_message


# --------------------------------
# Upload dataset to E2B sandbox
# --------------------------------
def upload_dataset(code_interpreter: Sandbox, uploaded_file) -> str:
    dataset_path = f"./{uploaded_file.name}"
    code_interpreter.files.write(dataset_path, uploaded_file)
    return dataset_path


# --------------------------------
# AUTO EDA HELPERS
# --------------------------------
def show_basic_summary(df: pd.DataFrame):
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


def show_missing_values(df: pd.DataFrame):
    st.subheader("🧩 Missing Values")
    missing = df.isna().sum()
    if (missing > 0).any():
        st.write(missing.to_frame("missing_count"))
        st.bar_chart(missing)
    else:
        st.info("No missing values found.")


def show_correlation_heatmap(df: pd.DataFrame):
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


def show_outlier_view(df: pd.DataFrame):
    st.subheader("📦 Outlier View (Boxplot)")
    numeric_df = df.select_dtypes(include=["int", "float"])  # ✅ FIXED

    if numeric_df.empty:
        st.info("No numeric columns found.")
        return

    col = st.selectbox("Select a column", numeric_df.columns)
    fig, ax = plt.subplots()
    ax.boxplot(numeric_df[col].dropna(), vert=True)
    ax.set_title(f"Boxplot of {col}")
    st.pyplot(fig)


def show_anomaly_detection(df: pd.DataFrame):
    st.subheader("🚨 Anomaly Detection (Z-score based)")
    numeric_df = df.select_dtypes(include=["int", "float"])
    if numeric_df.empty:
        st.info("No numeric columns found for anomaly detection.")
        return

    col = st.selectbox(
        "Select numeric column for anomaly detection",
        numeric_df.columns)
    threshold = st.slider("Z-score threshold", 2.0, 5.0, 3.0, 0.1)
    series = numeric_df[col].dropna()
    mean = series.mean()
    std = series.std()

    if std == 0:
        st.info("Standard deviation is zero; cannot detect anomalies for this column.")
        return

    z_scores = (series - mean) / std
    anomalies = series[abs(z_scores) > threshold]

    st.write(
        f"Detected {
            anomalies.shape[0]} anomalies with |z| > {threshold}.")
    if not anomalies.empty:
        st.dataframe(df.loc[anomalies.index])


def generate_eda_report_text(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    buf.write("AI Data Visualization Agent - Auto EDA Report\n")
    buf.write("=" * 60 + "\n\n")
    buf.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}\n\n")

    buf.write("Column Types:\n")
    buf.write(str(pd.DataFrame(df.dtypes, columns=["dtype"])) + "\n\n")

    numeric_df = df.select_dtypes(include=["int", "float"])   # ✅ FIXED LINE

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


def generate_llm_insights(df: pd.DataFrame) -> str:
    if not st.session_state.groq_api_key:
        return "Groq API key missing."

    numeric_df = df.select_dtypes(include=["int", "float"])  # ✅ FIXED

    summary_text = ""

    if not numeric_df.empty:
        summary_text += "Summary statistics:\n"
        summary_text += numeric_df.describe().to_string() + "\n\n"
        summary_text += "Correlation matrix:\n"
        summary_text += numeric_df.corr().to_string() + "\n\n"

    summary_text += "Missing values:\n"
    summary_text += df.isna().sum().to_string() + "\n\n"

    prompt = f"""
You are a data analyst. Based on this dataset summary, give 5–7 clear, bullet-point insights:

{summary_text}
"""

    client = Groq(api_key=st.session_state.groq_api_key)
    res = client.chat.completions.create(
        model=st.session_state.model_name,
        messages=[
            {"role": "system", "content": "You are a helpful data analyst."},
            {"role": "user", "content": prompt},
        ],
    )
    return res.choices[0].message.content


# --------------------------------
# Data cleaning
# --------------------------------
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates()

    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors="ignore")
            except Exception:
                pass

    numeric_cols = df.select_dtypes(include=["int", "float"]).columns
    for col in numeric_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        if df[col].isna().any():
            mode_val = df[col].mode().iloc[0]
            df[col] = df[col].fillna(mode_val)

    return df


# --------------------------------
# AutoML + Feature Importance
# --------------------------------
def auto_train_model(df: pd.DataFrame, target_col: str):
    df = df.copy().dropna(subset=[target_col])
    y = df[target_col]
    is_numeric_target = pd.api.types.is_numeric_dtype(y)

    X = df.drop(columns=[target_col])
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if is_numeric_target:
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        metric_text = f"Regression task detected. MSE = {mse:.3f}"
    else:
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        metric_text = f"Classification task detected. Accuracy = {acc:.3f}\n\n"
        metric_text += classification_report(y_test, y_pred)

    importances = pd.Series(
        model.feature_importances_,
        index=X.columns).sort_values(
        ascending=False)

    return model, metric_text, importances


# --------------------------------
# Smart chart recommendation
# --------------------------------
def smart_chart(df: pd.DataFrame, x_col: str, y_col: Optional[str] = None):
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
            df.groupby(x_col)[y_col].mean(
            ).sort_values().plot(kind="bar", ax=ax)
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


# --------------------------------
# Chart storytelling
# --------------------------------
def narrate_chart(
        df: pd.DataFrame,
        x_col: str,
        y_col: Optional[str] = None) -> str:
    if not st.session_state.groq_api_key:
        return "Groq API key missing. Cannot generate chart narrative."

    cols = [x_col] + ([y_col] if y_col else [])
    sample = df[cols].head(50)

    prompt = f"""
You are a data storyteller. You are given a chart based on columns:

X: {x_col}
Y: {y_col if y_col else "None (only one column)"}

Here is a sample of the data behind the chart:
{sample.to_string()}

Write 5-6 bullet points explaining:
- what the main pattern/trend is,
- which values are high/low,
- any anomalies or interesting points,
- what a business person should conclude.
"""

    client = Groq(api_key=st.session_state.groq_api_key)
    resp = client.chat.completions.create(
        model=st.session_state.model_name,
        messages=[
            {"role": "system", "content": "You are a senior data storyteller."},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content


# --------------------------------
# SQL helpers
# --------------------------------
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


# --------------------------------
# Simple forecasting
# --------------------------------
def show_forecasting(df: pd.DataFrame):
    st.subheader("📈 Simple Time-Series Forecasting (Trend-based)")

    date_cols = [
        c
        for c in df.columns
        if pd.api.types.is_datetime64_any_dtype(df[c])
        or pd.api.types.is_object_dtype(df[c])
    ]
    if not date_cols:
        st.info(
            "No obvious date/time columns. If you have a date column, make sure it is in a parseable format."
        )
        return

    date_col = st.selectbox("Select date column", date_cols)

    numeric_df = df.select_dtypes(include=["int", "float"])
    if numeric_df.empty:
        st.info("No numeric columns available for forecasting.")
        return

    value_col = st.selectbox("Select value column", numeric_df.columns)
    horizon = st.slider("Forecast steps", 5, 50, 10)

    temp = df[[date_col, value_col]].dropna().copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp = temp.dropna(subset=[date_col])
    temp = temp.sort_values(by=date_col)

    if temp.shape[0] < 10:
        st.info("Need at least 10 data points for a meaningful forecast.")
        return

    y = temp[value_col].values
    x = np.arange(len(y))

    coeffs = np.polyfit(x, y, 1)
    trend = np.poly1d(coeffs)

    x_future = np.arange(len(y) + horizon)
    y_future = trend(x_future)

    date_index = temp[date_col].values
    deltas = np.diff(date_index)
    if len(deltas) == 0:
        st.info("Not enough distinct dates to compute frequency.")
        return
    step = np.median(deltas)
    last_date = date_index[-1]
    future_dates = [last_date + (i + 1) * step for i in range(horizon)]

    fig, ax = plt.subplots()
    ax.plot(temp[date_col], y, label="History")
    ax.plot(
        list(temp[date_col]) + list(future_dates),
        y_future,
        linestyle="--",
        label="Trend + Forecast",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel(value_col)
    ax.legend()
    st.pyplot(fig)


# --------------------------------
# PDF report
# --------------------------------
def create_pdf_report(text: str, filename: str = "eda_report.pdf") -> bytes:
    buffer = BytesIO()
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


# --------------------------------
# MAIN APP
# --------------------------------
def main():
    st.title("📊 AI Data Visualization Agent (Groq Powered)")
    st.write(
        "Upload a CSV file, ask questions in natural language, run Auto-EDA, SQL queries, anomaly detection, forecasting, cleaning, AutoML, smart charts, AI storytelling, and chat with your dataset."
    )

    if "groq_api_key" not in st.session_state:
        st.session_state.groq_api_key = ""
    if "e2b_api_key" not in st.session_state:
        st.session_state.e2b_api_key = ""
    if "model_name" not in st.session_state:
        st.session_state.model_name = ""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        st.header("🔑 API Keys Required")

        st.session_state.groq_api_key = st.text_input(
            "Groq API Key", type="password"
        )
        st.markdown("[Get Groq API Key](https://console.groq.com/keys)")

        st.session_state.e2b_api_key = st.text_input(
            "E2B Code Interpreter API Key", type="password"
        )
        st.markdown("[Get E2B API Key](https://e2b.dev)")

        model_options = {
            "Llama 3.1 8B Instant": "llama-3.1-8b-instant",
            "Llama 3.1 70B Versatile": "llama-3.1-70b-versatile",
            "Mixtral 8x7B Instruct": "mixtral-8x7b-instruct",
        }
        selected = st.selectbox("Select Model", list(model_options.keys()))
        st.session_state.model_name = model_options[selected]

    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### 📄 Dataset Preview")
        st.dataframe(df.head())

        # ---------------- Ask AI section ----------------
        st.markdown("## 🤖 Ask AI About Your Data")

        chart_type = st.selectbox(
            "Preferred chart type (for AI-generated code)",
            ["auto", "bar", "line", "scatter", "hist", "box", "heatmap", "pie"],
            index=0,
        )

        query = st.text_area(
            "Ask a question about the data:",
            "Plot the average of numeric columns grouped by a category column.",
        )

        if st.button("Analyze with AI"):
            if not st.session_state.groq_api_key or not st.session_state.e2b_api_key:
                st.error("Please enter BOTH Groq and E2B API keys.")
            else:
                with Sandbox(api_key=st.session_state.e2b_api_key) as code_interpreter:
                     dataset_path = upload_dataset(code_interpreter, uploaded_file)
                     results, llm_response = chat_with_llm(
                        code_interpreter, query, dataset_path, chart_type
                )

                st.write("### 🧠 LLM Response")
                st.write(llm_response)

                # SHOW RESULTS (INSIDE THE BUTTON BLOCK)
                if results:
                    for result in results:

                        # Plotly figure
                        if hasattr(result, "to_plotly_json"):
                            st.plotly_chart(result, use_container_width=True)

                        # Matplotlib figure
                        elif hasattr(result, "figure"):
                            st.pyplot(result.figure)

                        # DataFrame or Series
                        elif isinstance(result, (pd.DataFrame, pd.Series)):
                            st.dataframe(result)

                         # Other objects
                        else:
                            st.write(result)

        # ---------------- Auto-EDA ----------------
        st.markdown("---")
        st.markdown("## 📈 Automatic Exploratory Data Analysis")

        if st.button("Run Full Auto-EDA"):
            show_basic_summary(df)
            st.markdown("---")
            show_missing_values(df)
            st.markdown("---")
            show_correlation_heatmap(df)
            st.markdown("---")
            show_outlier_view(df)

        # ---------------- Cleaning ----------------
        st.markdown("---")
        st.markdown("## 🧹 Data Cleaning Agent")

        if st.button("Clean My Dataset"):
            cleaned_df = clean_dataset(df)
            st.session_state.cleaned_df = cleaned_df
            st.success("Dataset cleaned! Preview of cleaned data:")
            st.dataframe(cleaned_df.head())
        else:
            st.info("Click the button to clean your dataset using simple rules.")

        # ---------------- Anomaly detection ----------------
        st.markdown("---")
        st.markdown("## 🚨 Anomaly Detection")
        show_anomaly_detection(df)

        # ---------------- SQL mode ----------------
        st.markdown("---")
        st.markdown("## 🧮 SQL Mode (table name: data)")

        default_sql = "SELECT * FROM data LIMIT 5;"
        sql_query = st.text_area(
            "Write a SQL query on table `data`:",
            default_sql)

        if st.button("Run SQL Query"):
            result = run_sql_query(df, sql_query)
            if result is not None:
                st.dataframe(result)

        # ---------------- Forecasting ----------------
        st.markdown("---")
        st.markdown("## 📈 Simple Forecasting")
        show_forecasting(df)

        # ---------------- Auto ML ----------------
        st.markdown("---")
        st.markdown("## 🤖 Auto ML Model Training + Feature Importance")

        target_options = [c for c in df.columns if df[c].nunique() > 1]
        if not target_options:
            st.info("No suitable target columns found.")
        else:
            target_col = st.selectbox("Select target column", target_options)

            if st.button("Train Model"):
                model, metric_text, importances = auto_train_model(
                    df, target_col)

                st.write("### 📊 Model Performance")
                st.text(metric_text)

                st.write("### ⭐ Feature Importance")
                st.dataframe(importances.to_frame("importance").head(20))

                fig, ax = plt.subplots()
                importances.head(20).plot(kind="bar", ax=ax)
                ax.set_title("Top 20 Feature Importances")
                st.pyplot(fig)

        # ---------------- Smart chart + storytelling ----------------
        st.markdown("---")
        st.markdown("## 🧠 Smart Chart Recommendation + Storytelling")

        x_col = st.selectbox("X-axis column", df.columns, key="smart_x")
        y_col = st.selectbox(
            "Y-axis column (optional)",
            ["<None>"] +
            list(
                df.columns),
            key="smart_y")

        if st.button("Generate Smart Chart"):
            if y_col == "<None>":
                smart_chart(df, x_col)
                story = narrate_chart(df, x_col, None)
            else:
                smart_chart(df, x_col, y_col)
                story = narrate_chart(df, x_col, y_col)

            st.markdown("### 📝 AI Narrative for This Chart")
            st.write(story)

        # ---------------- AI insights + reports ----------------
        st.markdown("---")
        st.subheader("🧠 AI Insights on This Dataset")
        if st.button("Generate AI Insights"):
            insights = generate_llm_insights(df)
            st.write(insights)

        report_text = generate_eda_report_text(df)
        st.download_button(
            label="📄 Download EDA Report (Text)",
            data=report_text,
            file_name="eda_report.txt",
            mime="text/plain",
        )

        pdf_bytes = create_pdf_report(report_text)
        st.download_button(
            label="📕 Download EDA Report (PDF)",
            data=pdf_bytes,
            file_name="eda_report.pdf",
            mime="application/pdf",
        )

        # ---------------- Chat with dataset ----------------
        st.markdown("---")
        st.markdown("## 💬 Chat With Your Dataset")

        user_msg = st.text_input(
            "Ask anything about this dataset:",
            key="chat_dataset")

        if st.button("Send Question"):
            if not user_msg.strip():
                st.warning("Please enter a question.")
            elif not st.session_state.groq_api_key:
                st.error("Groq API key missing.")
            else:
                schema = pd.DataFrame(df.dtypes, columns=["dtype"])
                head = df.head(5)

                prompt = f"""
You are a data analyst chatbot. The user will ask questions about a dataset.

Dataset schema:
{schema.to_string()}

First 5 rows:
{head.to_string()}

Conversation so far:
{st.session_state.chat_history}

User question: {user_msg}

Answer in a friendly, clear way. If you need exact stats/plots that require code,
explain what should be done conceptually, based on the sample.
"""

                client = Groq(api_key=st.session_state.groq_api_key)
                resp = client.chat.completions.create(
                    model=st.session_state.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful data analyst.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                )
                answer = resp.choices[0].message.content

                st.session_state.chat_history.append(
                    {"user": user_msg, "assistant": answer}
                )

        if st.session_state.chat_history:
            st.write("### 🧠 Conversation History")
            for turn in st.session_state.chat_history:
                st.markdown(f"**You:** {turn['user']}")
                st.markdown(f"**AI:** {turn['assistant']}")
                st.markdown("---")


if __name__ == "__main__":
    main()
