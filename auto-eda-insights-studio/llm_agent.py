# llm_agent.py → Groq + E2B + code generation
import io
import re
import sys
import contextlib
import warnings
from typing import Optional, List, Any, Tuple

import pandas as pd
import streamlit as st
from groq import Groq
from e2b_code_interpreter import Sandbox

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

pattern = re.compile(r"```python\n(.*?)\n```", re.DOTALL)


# -----------------------------
# Execute Python inside E2B sandbox
# -----------------------------
def code_interpret(e2b_code_interpreter: Sandbox, code: str) -> Optional[List[Any]]:
    with st.spinner("⚙️ Executing code in E2B sandbox..."):
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(
            stderr_capture
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                exec_result = e2b_code_interpreter.run_code(code)

        if stderr_capture.getvalue():
            print("[Code Interpreter Warning/Error]", file=sys.stderr)
            print(stderr_capture.getvalue(), file=sys.stderr)

        if exec_result.error:
            print(f"[Code Interpreter ERROR] {exec_result.error}", file=sys.stderr)
            return None

        return exec_result.results


# -----------------------------
# Extract Python code block from LLM message
# -----------------------------
def match_code_blocks(llm_response: str) -> str:
    match = pattern.search(llm_response)
    if match:
        return match.group(1).strip()

    # fallback: treat raw text as code
    lines = llm_response.strip().splitlines()
    cleaned = []
    for line in lines:
        if line.strip().startswith("```"):
            continue
        cleaned.append(line)

    return "\n".join(cleaned).strip()


# -----------------------------
# Upload dataset into sandbox
# -----------------------------
def upload_dataset(code_interpreter: Sandbox, uploaded_file) -> str:
    dataset_path = f"./{uploaded_file.name}"
    code_interpreter.files.write(dataset_path, uploaded_file)
    return dataset_path


# -----------------------------
# LLM: generate Python visualization code
# -----------------------------
def chat_with_llm(
    e2b_code_interpreter: Sandbox,
    user_message: str,
    dataset_path: str,
    chart_type: str,
    column_list: list,
):
    system_prompt = f"""
You are an expert Python data analyst.
You must generate ONE clean Python code block ONLY.

RULES YOU MUST FOLLOW:

1. Load the dataset EXACTLY like this:
   df = pd.read_csv("{dataset_path}")

2. NEVER:
   - load any other dataset
   - recreate df
   - rename columns
   - modify column spelling
   - pluralize/singularize column names
   - define functions
   - add extra imports
   - print anything
   - use df.head()
   - use fig.show()
   - write explanations
   - NEVER invent new column names.
   - You MUST use exact column names from the dataset.
   - If the user writes a typo like "claims" but actual column is "claim",
     automatically correct it to the closest real column name.


3. Only allowed imports:
   import pandas as pd
   import plotly.express as px

4. Use ONLY column names that already exist in df.columns.
   If needed, inspect df.columns directly inside the generated code.

5. Your code MUST:
   - answer the user question using pandas
   - create a Plotly figure when visualization is appropriate
   - store the figure in a variable named fig
   - END WITH: fig

6. Chart rules when user selects a chart:
   bar -> px.bar
   line -> px.line
   scatter -> px.scatter
   hist -> px.histogram
   box -> px.box
   heatmap -> px.imshow
   pie -> px.pie

IMPORTANT:
Here are all valid column names in the dataset:
{column_list}

You MUST use ONLY these exact column names.
If the user writes a wrong column, map it to the closest correct one.

7. If chart_type == "auto":
   Choose the best chart based on columns:
      numeric vs numeric        -> scatter
      categorical vs numeric    -> bar
      datetime vs numeric       -> line
      one numeric               -> histogram
      one categorical           -> bar
      categorical vs categorical -> grouped bar
      
You MUST use ONLY these exact names.
If user asks using a wrong name (like 'claims'), fix it to the correct one ('claim').


8. Your output MUST be ONE Python code block ONLY:

import pandas as pd
import plotly.express as px

df = pd.read_csv("{dataset_path}")

# analysis code here
# plotting code here

fig = <a plotly figure>
fig

NO explanation text.
NO markdown outside the code block.
"""


    full_user_message = (
        user_message
        + f"\n\nUser-selected chart type: {chart_type}. Use it unless auto is selected."
    )

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
    else:
        st.warning("⚠ No Python code found in LLM response.")
        return None, response_message
