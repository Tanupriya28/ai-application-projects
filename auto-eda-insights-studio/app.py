

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from e2b_code_interpreter import Sandbox

from backend import (
    show_basic_summary,
    show_missing_values,
    show_correlation_heatmap,
    show_anomaly_detection,
    clean_dataset,
    auto_train_model,
    smart_chart,
    run_sql_query,
    show_forecasting,
    generate_eda_report_text,
    create_pdf_report,
)
from llm_agent import chat_with_llm, upload_dataset


# -------------------------------------------------------
# 🍏 VISION PRO DARK MODE — PREMIUM UI STYLING
# -------------------------------------------------------
visionpro_dark_css = """
<style>

@import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600;700;800;900&display=swap');

* {
    font-family: 'SF Pro Display', sans-serif !important;
}

/* ---------- APP BACKGROUND ---------- */
body, .stApp {
    background: radial-gradient(circle at 10% 10%, #050816 0%, #050b18 40%, #020617 80%) !important;
    color: #E5E7EB !important; /* light grey text */
}

/* ---------- MAIN TITLE ---------- */
h1 {
    font-size: 54px !important;
    font-weight: 800 !important;

    padding: 10px 0;
    margin-bottom: 25px !important;
    margin-top: 10px !important;

    color: #F9FAFB !important;
    letter-spacing: -0.02em;

    text-align: left !important;

    text-shadow: 0 2px 18px rgba(59,130,246,0.45);
}

/* ---------- SECTION HEADINGS ---------- */
h2, h3 {
    color: #F9FAFB !important;
    font-weight: 700 !important;
    margin-top: 20px !important;
}

/* ---------- LABELS / SMALL TEXT ---------- */
label, .stMarkdown p {
    color: #E5E7EB !important;
}

/* ---------- MAIN CONTENT PANELS ---------- */
.block-container {
    padding-top: 1.2rem !important;
}

.stDataFrame, .stPlotlyChart {
    border-radius: 18px !important;
    background: rgba(15,23,42,0.92) !important;
    border: 1px solid rgba(148,163,184,0.35);
    backdrop-filter: blur(20px) saturate(150%);
    padding: 14px !important;
    box-shadow: 0 10px 40px rgba(0,0,0,0.7);
}

/* ---------- SIDEBAR ---------- */
section[data-testid="stSidebar"] {
    background: radial-gradient(circle at 0% 0%, #020617 0%, #020617 35%, #020617 100%) !important;
    border-right: 1px solid rgba(148,163,184,0.35);
    box-shadow: 6px 0 30px rgba(0,0,0,0.5);
}

section[data-testid="stSidebar"] * {
    color: #E5E7EB !important;
}

section[data-testid="stSidebar"] a {
    color: #60A5FA !important;
}

/* ---------- INPUTS ---------- */
.stTextInput input,
.stTextArea textarea,
textarea,
input[type="text"] {
    background-color: #111827 !important;
    color: #F9FAFB !important;
    border-radius: 10px !important;
    border: 1px solid #374151 !important;
    padding: 10px 12px !important;
    font-size: 15px !important;
}

.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #9CA3AF !important;
}

/* ---------- SELECT BOX ---------- */
.stSelectbox div[data-baseweb="select"] > div {
    background-color: #111827 !important;
    color: #F9FAFB !important;
    border-radius: 10px !important;
    border: 1px solid #374151 !important;
    min-height: 40px;
}

/* ---------- BUTTONS ---------- */
.stButton > button {
    background: linear-gradient(135deg, #2563EB, #4F46E5) !important;
    color: #F9FAFB !important;
    border: none !important;
    border-radius: 999px !important;
    padding: 10px 26px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    box-shadow: 0 10px 30px rgba(37,99,235,0.45);
    transition: all 0.2s ease-out;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 16px 40px rgba(37,99,235,0.7);
}

/* ---------- TABS ---------- */
div[data-testid="stTabs"] button {
    border-radius: 999px !important;
    background: #020617 !important;
    color: #9CA3AF !important;
    border: 1px solid #1F2937 !important;
    padding: 6px 18px !important;
    font-weight: 500 !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    background: linear-gradient(135deg, #1D4ED8, #4F46E5) !important;
    color: #F9FAFB !important;
    border-color: transparent !important;
}

/* ---------- FILE UPLOADER ---------- */
[data-testid="stFileUploader"] section {
    background-color: #111827 !important;
    border-radius: 14px !important;
    border: 1px solid #374151 !important;
}
[data-testid="stFileUploader"] label {
    color: #F9FAFB !important;
}
[data-testid="stFileUploader"] button {
    background-color: #1E3A8A !important;
    color: #F9FAFB !important;
}
[data-testid="stFileUploader"] button:hover {
    background-color: #3B82F6 !important;
    color: #FFFFFF !important;
}



/* ---------- FIX EXPANDER HEADER (remove ugly white bar) ---------- */
.streamlit-expanderHeader {
    background-color: #0f172a !important;  /* dark slate */
    color: #F9FAFB !important;
    border-radius: 10px !important;
    border: 1px solid rgba(148,163,184,0.25) !important;
    padding: 8px 12px !important;
}

.streamlit-expanderHeader p {
    color: #F9FAFB !important;
}

/* Hide Streamlit expander shortcut hint */
.streamlit-expanderHeader .keyboard-shortcut, 
.streamlit-expanderHeader [title="Keyboard shortcuts"] {
    display: none !important;
}
/* FIX: Hide the expander arrow that is overlapping */
details > summary::-webkit-details-marker {
    display: none !important;
}

details > summary::marker {
    display: none !important;
}

/* Additional Streamlit internal arrow hiding */
.streamlit-expanderHeader svg {
    display: none !important;
}

.streamlit-expanderHeader [data-testid="stExpanderChevron"] {
    display: none !important;
}

fig.update_layout(height=500)

</style>

"""
st.markdown(visionpro_dark_css, unsafe_allow_html=True)

st.markdown("""
<style>

/* REMOVE ALL BACKGROUND BUBBLES */
.js-plotly-plot .modebar, 
.js-plotly-plot .modebar-group {
    background: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* MAKE ICONS WHITE & CLEAR */
.js-plotly-plot .modebar-btn svg {
    filter: invert(1) brightness(2) !important;  /* white icons */
    opacity: 1 !important;
}

/* HOVER LOOK — LIGHT GREY */
.js-plotly-plot .modebar-btn:hover {
    background-color: rgba(255,255,255,0.25) !important;
    border-radius: 6px !important;
}

/* ALWAYS SHOW TOOLBAR */
.js-plotly-plot .modebar {
    opacity: 1 !important;
}

/* PREVENT CLIPPING */
.stPlotlyChart > div {
    overflow: visible !important;
}

</style>
""", unsafe_allow_html=True)





# -------------------------------------------------------
# APP START
# -------------------------------------------------------
def main():
    st.set_page_config(page_title="AI Data Insights Studio", layout="wide")

    st.title("AI Data Insights Studio")
    st.write(
        "A unified workspace to explore, visualize, analyze, clean, forecast, and model your datasets using natural language."
    )

    # ---------------- Session State ----------------
    if "groq_api_key" not in st.session_state:
        st.session_state.groq_api_key = ""
    if "e2b_api_key" not in st.session_state:
        st.session_state.e2b_api_key = ""
    if "model_name" not in st.session_state:
        st.session_state.model_name = ""

    if "eda_ran" not in st.session_state:
        st.session_state.eda_ran = False
    if "clean_df" not in st.session_state:
        st.session_state.clean_df = None
    if "cleaning_report" not in st.session_state:
        st.session_state.cleaning_report = None

    # SQL history
    if "sql_query" not in st.session_state:
        st.session_state.sql_query = "SELECT * FROM data LIMIT 5;"
    if "sql_history" not in st.session_state:
        st.session_state.sql_history = []

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.header("API Keys")

        st.session_state.groq_api_key = st.text_input("Groq API Key", type="password")
        st.session_state.e2b_api_key = st.text_input("E2B Interpreter Key", type="password")

        model_options = {
            "Llama 3.3 70B Versatile": "llama-3.3-70b-versatile",
            "Llama 3.1 70B Turbo": "llama-3.1-70b-turbo",
            "Llama 3.1 8B Turbo": "llama-3.1-8b-turbo",
            "Mixtral 8x7B Instruct": "mixtral-8x7b-instruct",
            "Gemma2 9B": "gemma2-9b-it",
        }
        selected = st.selectbox("Model", list(model_options.keys()))
        st.session_state.model_name = model_options[selected]

    # ---------------- File Upload ----------------
    uploaded_file = st.file_uploader("📂 Upload a CSV file", type="csv")

    if uploaded_file is not None:
        st.session_state["uploaded_file"] = uploaded_file

    if "uploaded_file" not in st.session_state:
        st.info("Please upload a CSV file to continue.")
        st.stop()

    df = pd.read_csv(st.session_state["uploaded_file"])

    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    # ---------------- Tabs ----------------
    tab_ai, tab_eda, tab_clean, tab_sql, tab_ml, tab_chat = st.tabs(
        ["Ask AI", "Auto EDA", "Cleaning", "SQL", "AutoML", "Chat"]
    )
    
   
    
       
    # -----------------------------------------------
# TAB 1 — ASK AI
# -----------------------------------------------
    with tab_ai:
        st.subheader("Ask AI to Analyze & Visualize Data")

        chart_type = st.selectbox(
            "Preferred Chart Type",
            ["auto", "bar", "line", "scatter", "hist", "box", "heatmap", "pie"],
        )

        query = st.text_area("Ask a question:", "Count of diabetes by gender")

        if st.button("Analyze"):
            if not st.session_state.groq_api_key or not st.session_state.e2b_api_key:
                st.error("Enter both API keys.")
            else:
                with Sandbox(api_key=st.session_state.e2b_api_key) as code_interpreter:

                    dataset_path = upload_dataset(
                        code_interpreter, st.session_state["uploaded_file"]
                    )

                    results, llm_response = chat_with_llm(
                        code_interpreter, query, dataset_path, chart_type, list(df.columns)
                    )

                    # Show generated code
                    st.code(llm_response, language="python")

                    # -------------------------
                    # FINAL WORKING RESULT HANDLER
                    # -------------------------
                    if results:
                        for result in results:

                            extracted = None

                            # 1) Sometimes the figure is here
                            if hasattr(result, "value") and result.value is not None:
                                extracted = result.value

                            # 2) Sometimes here
                            elif hasattr(result, "data") and result.data is not None:
                                extracted = result.data

                            # 3) YOUR PLOTLY FIGURE IS HERE (confirmed by debug)
                            if extracted is None and hasattr(result, "extra"):
                                if isinstance(result.extra, dict):
                                    if "application/vnd.plotly.v1+json" in result.extra:
                                        extracted = result.extra["application/vnd.plotly.v1+json"]

                            # nothing found → skip
                            if extracted is None:
                                continue

                            # ---------------------------------------
                            # CASE A — Actual Plotly Figure object
                            # ---------------------------------------
                            try:
                                import plotly.graph_objs as go
                                if isinstance(extracted, go.Figure):
                                    st.plotly_chart(extracted, use_container_width=True)
                                    continue
                            except:
                                pass

                            # ---------------------------------------
                            # CASE B — Plotly Dict (data + layout)
                            # ---------------------------------------
                            if isinstance(extracted, dict) and "data" in extracted and "layout" in extracted:

                                import plotly.graph_objects as go

                                # Convert dict → real Plotly figure
                                fig = go.Figure(extracted)

                                # FIX chart size & overflow
                                fig.update_layout(
                                autosize=True,
                                height=600,
                                margin=dict(l=40, r=40, t=60, b=40)
                                )

                                st.plotly_chart(fig, use_container_width=True)
                                continue
                            # ---------------------------------------
                            # CASE C — DataFrame / Series
                            # ---------------------------------------
                            if isinstance(extracted, (pd.DataFrame, pd.Series)):
                                st.dataframe(extracted, use_container_width=True)
                                continue

                            # ---------------------------------------
                            # CASE D — Text output
                            # ---------------------------------------
                            if isinstance(extracted, str) and extracted.strip():
                                st.write(extracted)
                                continue

                    else:
                        st.info("No visual output or data returned.")



    # -----------------------------------------------
    # TAB 2 — EDA
    # -----------------------------------------------
    with tab_eda:
        st.subheader("Automatic EDA")

        if st.button("Run EDA"):
            st.session_state.eda_ran = True

        if st.session_state.eda_ran:
            show_basic_summary(df)
            show_missing_values(df)
            show_correlation_heatmap(df)
            show_anomaly_detection(df, key_suffix="eda")

        report_text = generate_eda_report_text(df)
        st.download_button("Download Report (.txt)", report_text, "eda_report.txt")
        st.download_button(
            "Download Report (.pdf)",
            create_pdf_report(report_text),
            "eda_report.pdf",
        )

    # -----------------------------------------------
    # TAB 3 — CLEAN DATA
    # -----------------------------------------------
    with tab_clean:
        st.subheader("Cleaning Tools")

        cleaned_df = st.session_state.clean_df
        cleaning_report = st.session_state.cleaning_report

        if st.button("Clean Dataset"):
            cleaned_df, cleaning_report = clean_dataset(df)
            st.session_state.clean_df = cleaned_df
            st.session_state.cleaning_report = cleaning_report

        if cleaned_df is not None:
            st.success("Dataset Cleaned Successfully ✓")

            if cleaning_report:
                st.subheader("🧼 Cleaning Report")
                for item in cleaning_report:
                    st.write("• " + item)

            st.subheader("📘 Cleaned Dataset")
            st.dataframe(cleaned_df, use_container_width=True)

        else:
            st.info("Click 'Clean Dataset' to clean your data.")
            st.dataframe(df, use_container_width=True)

      
        # -----------------------------------------------
    # TAB 4 — SQL (clean + history + delete + clear, NO USE BUTTON)
    # -----------------------------------------------
    with tab_sql:
        st.subheader("SQL Workspace")

        # Schema viewer (same UX)
        st.markdown("#### 📚 Table: `data` schema")
        schema_df = pd.DataFrame(
            {"Column": df.columns, "Type": df.dtypes.astype(str)}
        )
        st.dataframe(schema_df, use_container_width=True)

        # SQL Editor
        st.markdown("### 🧾 SQL Editor")
        sql_query = st.text_area(
            "Write SQL query",
            value=st.session_state.sql_query,
            height=150,
            key="sql_editor",
        )
        st.session_state.sql_query = sql_query

        # Execute
        if st.button("Execute SQL"):
            if sql_query.strip():

                # Save to history (max 8)
                if sql_query not in st.session_state.sql_history:
                    st.session_state.sql_history.insert(0, sql_query)
                    st.session_state.sql_history = st.session_state.sql_history[:8]

                result = run_sql_query(df, sql_query)
                if result is not None:
                    st.subheader("📊 Query Result")
                    st.dataframe(result, use_container_width=True)
            else:
                st.warning("Please enter a SQL query.")

        # History
        st.markdown("### 🕘 Query History")
        history = st.session_state.sql_history

        if history:

            # Clear all
            if st.button("🧹 Clear History"):
                st.session_state.sql_history = []
                st.rerun()

            for i, q in enumerate(history):
                cols = st.columns([9, 1])

                # Show query
                with cols[0]:
                    st.code(q, language="sql")

                # Delete button
                with cols[1]:
                    if st.button("✖", key=f"delete_{i}"):
                        st.session_state.sql_history.pop(i)
                        st.rerun()

        else:
            st.info("No previous queries yet.")

        st.markdown("---")
        show_forecasting(df)



    # -----------------------------------------------
    # TAB 5 — AutoML
    # -----------------------------------------------
    with tab_ml:
        st.subheader("AutoML Model Training")

        candidates = [c for c in df.columns if df[c].nunique() > 1]

        if candidates:
            target = st.selectbox("Select target column:", candidates)

            if st.button("Train"):
                data_for_ml = st.session_state.clean_df if st.session_state.clean_df is not None else df
                model, metrics,  _ = auto_train_model(data_for_ml, target)
                st.write(metrics)


        st.subheader("Smart Chart Recommendation")
        x_col = st.selectbox("X-axis", df.columns)
        y_col = st.selectbox("Y-axis", ["<None>"] + list(df.columns))

        if st.button("Generate Chart"):
            smart_chart(df, x_col, None if y_col == "<None>" else y_col)
    
    # -----------------------------------------------
    # TAB 6 — Chat (Fully Fixed + Stable)
    # -----------------------------------------------
     # -------------------------
    # Custom Chat With Dataset
    # -------------------------
    with tab_chat:
        st.subheader("Chat With Dataset")
        st.write("Ask natural language questions about your dataset.")

        user_msg = st.text_input("Ask something:")

        if st.button("Send"):
            if not st.session_state.groq_api_key:
                st.error("Missing Groq API key.")
            else:
                 # Prepare dataset schema context
                schema_info = pd.DataFrame(df.dtypes, columns=["dtype"]).to_string()
                sample_info = df.sample(min(10, len(df))).to_string()
                summary_info = df.describe(include="all").to_string()
            # --- NEW PROMPT: No return, no print, no assignments ---
            prompt = f"""
You are an intelligent pandas code generator. Convert ANY natural-language question into ONE valid pandas expression that answers the question using the dataframe df.

====================
STRICT RULES
====================
- Use ONLY the dataframe named df.
- NEVER recreate df or generate sample data.
- NEVER use print().
- NEVER use return.
- NEVER assign variables (NO "=").
- Output EXACTLY ONE pandas expression. No explanation. No text. No backticks.
- Use EXACT column names as they appear in the dataset.
- Use ONLY values that actually appear in the dataset (using schema/sample rows).
- NEVER guess categories.
- NEVER invert logic or reinterpret user intent.
- ALWAYS use df.loc[...] for filtering.
- NEVER use df[df[...] == ...] syntax.
- When filtering, ALWAYS write: df.loc[(condition1) & (condition2) & ...]
- For OR logic: df.loc[(condition1) | (condition2)]
- If question is ambiguous, choose the simplest literal interpretation using existing columns.

====================
MISSING VALUE RULES
====================
- "rows with missing values" → df[df.isna().any(axis=1)].shape[0]
- "all rows with missing data" → df[df.isna().any(axis=1)]
- "count missing in column X" → df['X'].isna().sum()
- "which columns have missing values" → df.isna().sum()

====================
AGGREGATION RULES
====================
- average / mean → .mean()
- total / sum → .sum()
- count → .shape[0]
- median → .median()
- max/min / highest/lowest → .max() / .min()
- percentage of rows where condition is true → (condition).mean()

====================
GROUP BY RULES
====================
- "by X" or "per X" → df.groupby('X')['Y'].agg('function')
- If multiple aggregations needed, return a DataFrame expression.

====================
RANKING / TOP-N RULES
====================
- top N → df.nlargest(N, 'column')
- bottom N → df.nsmallest(N, 'column')
- "highest X" → .max()
- "lowest X" → .min()

====================
TEXT RULES
====================
- "contains 'abc'" → df[df['column'].str.contains('abc', case=False, na=False)]
- "starts with" → .str.startswith()
- "ends with" → .str.endswith()
- unique values → df['column'].unique()
- value counts / distribution → df['column'].value_counts()

====================
DATE RULES
====================
If df contains datetime-like columns (detected from schema/sample):
- "after YYYY-MM-DD" → df[df['date'] > 'YYYY-MM-DD']
- "before YYYY-MM-DD" → df[df['date'] < 'YYYY-MM-DD']
- "between A and B" → df[(df['date'] >= A) & (df['date'] <= B)]
- "month-wise" → df['date'].dt.month.value_counts()

====================
CORRELATION RULES
====================
- correlation between X and Y → df[['X','Y']].corr()
- correlation matrix → df.corr()

====================
OUTLIER RULES (IQR)
====================
- outliers in column X →
    Q1 = df['X'].quantile(0.25)
    Q3 = df['X'].quantile(0.75)
    IQR = Q3 - Q1
    df.loc[(df['X'] < Q1 - 1.5*IQR) | (df['X'] > Q3 + 1.5*IQR)]

Express as a single expression without intermediate variables.

====================
ADVANCED INTERPRETATION
====================
- Handle multi-step reasoning.
- Understand synonyms:
    • “greater than” → >
    • “less than” → <
    • “not equal” → !=
    • “most common” → value_counts().idxmax()
    • “distribution of” → value_counts()
- If question refers to a concept, map it to the closest column name from schema.

====================
OUTPUT FORMAT
====================
- Output ONLY the final pandas expression.
- NO comments, NO explanation, NO markdown.

====================
Context:
Schema:
{schema_info}

Samples:
{sample_info}

Summary:
{summary_info}

User Question:
{user_msg}


"""

            # Call Groq
            from groq import Groq
            client = Groq(api_key=st.session_state.groq_api_key)

            llm_resp = client.chat.completions.create(
                model=st.session_state.model_name,
                messages=[
                    {"role": "system", "content": "You output ONLY Python pandas expressions."},
                    {"role": "user", "content": prompt},
                ],
            )

            generated_code = llm_resp.choices[0].message.content.strip()

            # Display code
            st.markdown("### 🤖 LLM Generated Code")
            st.code(generated_code, language="python")

            # -------------- EXECUTION SECTION --------------
            try:
                # Evaluate safely — only df allowed
                safe_env = {"df": df, "pd": pd}
                final_result = eval(generated_code, safe_env)

                st.markdown("### 📊 Result")

                if isinstance(final_result, (pd.DataFrame, pd.Series)):
                    st.dataframe(final_result, use_container_width=True)
                else:
                    st.markdown(
        f"<h1 style='font-size:70px; color:#F9FAFB; font-weight:800; margin-top:-10px;'>{final_result}</h1>",
        unsafe_allow_html=True,
    )

            except Exception as e:
                st.error(f"Execution error: {e}")

               



if __name__ == "__main__":
    main()
