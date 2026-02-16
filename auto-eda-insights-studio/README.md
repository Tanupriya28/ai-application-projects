
# 🚀 AI Data Insights Studio

An end-to-end AI-powered data analysis platform that allows users to upload datasets and instantly explore, clean, query, visualize, model, and interact with data using natural language.

This system combines:
- Automated data analytics
- Machine learning pipelines
- SQL querying
- Large Language Models (LLMs)
- Interactive dashboards

into a single unified workspace.

---

## 🎯 Project Objective

Real-world data is messy and requires multiple tools to analyze effectively.

AI Data Insights Studio removes this complexity by providing:

> A complete automated data analysis environment that works directly on raw uploaded CSV files — without manual coding.

From exploration to modeling to querying — everything is handled in one platform.

---

# 🧠 Core Features

## 📊 Automatic Exploratory Data Analysis (EDA)

- Dataset overview & schema detection  
- Descriptive statistics  
- Missing value visualization  
- Correlation heatmaps  
- Anomaly detection using Z-score  
- Auto-generated downloadable reports (TXT & PDF)

---

## 🧹 Intelligent Data Cleaning

Automatically performs:

- Duplicate removal  
- Datetime parsing  
- Median imputation for numerical columns  
- Mode imputation for categorical columns  

Provides a transparent cleaning report for users.

---

## 🤖 LLM-Powered Data Chatbot

Users can ask natural language questions about their dataset.

The system:

- Converts queries into **valid pandas expressions**
- Executes them safely
- Displays the result instantly
- Shows the **generated pandas code** for learning & transparency

Example:
> "Show average sales by region"  
→ Auto-generates pandas groupby code

This makes data analysis accessible without manual coding.

---

## 📈 AI-Generated Visualizations

Using LLM + Plotly:

- Natural language → Python visualization code  
- Automatically selects chart types  
- Ensures correct column usage (no hallucinations)  
- Produces interactive charts

---

## 🗄 SQL Analytics Workspace

Includes a built-in SQL engine:

- Upload dataset → instantly becomes a SQL table  
- Run standard SQL queries (SELECT, GROUP BY, WHERE, JOIN-style logic)  
- Schema viewer  
- Query history management  
- Result visualization

Bridges traditional SQL analytics with modern AI tools.

---

## 🧠 AutoML Pipeline (Production-Style)

### Automatic task detection
- Classification vs Regression

### Automated preprocessing
- Missing value handling
- Feature scaling
- One-hot encoding
- ColumnTransformer pipelines

### Model pool
Regression:
- Linear Regression  
- Random Forest  
- Gradient Boosting (scikit-learn)

Classification:
- Logistic Regression  
- Random Forest  
- Gradient Boosting (scikit-learn)

### Model selection
- 5-Fold Cross Validation
- Best model selected automatically

### Overfitting detection
- Compares test performance vs cross-validation score

### Model persistence
- Automatically saves trained model

---

## 📉 Forecasting Module

- Detects date-based columns
- Fits trend-based forecasting model
- Projects future values
- Visual forecast plots
- Downloadable forecast table

---

# 🏗 System Architecture
CSV Upload
↓
EDA + Cleaning
↓
SQL Query Layer
↓
LLM Chat & Visualization
↓
AutoML Pipeline
↓
Forecasting

LLM code runs in an isolated execution environment for safety.

---

# 🛠 Tech Stack

| Layer | Technology |
|------|-----------|
| UI | Streamlit |
| Data | Pandas, NumPy |
| ML | Scikit-learn |
| Visualization | Matplotlib, Seaborn, Plotly |
| SQL | SQLite (in-memory) |
| LLM | Groq API |
| Execution | E2B Sandbox |
| Reports | ReportLab |
| Model Saving | Joblib |

---

# 🚀 How to Run
git clone <repo-url>
cd project-folder
pip install -r requirements.txt
streamlit run app.py
________________________________________
# 📂 Project Structure
app.py
backend.py
llm_agent.py
visualization.py
requirements.txt
screenshots
README.md

# 📈 Future Enhancements
•	SHAP explainability
•	Advanced forecasting models
•	Cloud deployment
•	Dataset versioning
•	Dashboard performance tracking
________________________________________
# 👨‍💻 Author
Built as a full-stack AI data analytics system to demonstrate:
•	Machine learning pipelines
•	Data engineering workflows
•	LLM integration
•	Automated analytics
•	Real-world system design
________________________________________
# 🏁 Final Note
AI Data Insights Studio demonstrates how raw data can be transformed into insights using automation, machine learning, SQL, and AI — all within a single interactive platform.
