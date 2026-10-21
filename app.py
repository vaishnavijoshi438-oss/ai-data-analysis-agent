
import os
import re
import json
import time

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from dotenv import load_dotenv
from google import genai


# =========================================================
# 1. CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Smart AI Data Analysis Agent",
    page_icon="📊",
    layout="wide",
)

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
MODEL_NAME = "gemini-3.8-flash"
client = genai.Client(api_key=API_KEY) if API_KEY else None


# =========================================================
# 2. DATA PROFILE
# =========================================================

def data_profile(df):
    numeric = df.select_dtypes(include=np.number)

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": [str(c) for c in df.columns],
        "data_types": {
            str(c): str(t) for c, t in df.dtypes.items()
        },
        "missing_values": {
            str(c): int(v) for c, v in df.isna().sum().items()
        },
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_summary": (
            numeric.describe().round(2).to_dict()
            if not numeric.empty else {}
        ),
    }


# =========================================================
# 3. GEMINI REQUEST WITH RETRIES
# =========================================================

def gemini_request(prompt, retries=3):
    if client is None:
        raise RuntimeError(
            "Gemini API key missing. Check your .env file."
        )

    last_error = None

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )

            if not response.text:
                raise RuntimeError("Gemini returned an empty response.")

            return response.text

        except Exception as error:
            last_error = error
            message = str(error).upper()

            temporary = any(
                item in message
                for item in [
                    "503",
                    "UNAVAILABLE",
                    "429",
                    "RESOURCE_EXHAUSTED",
                    "500",
                    "INTERNAL",
                ]
            )

            if temporary and attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue

            raise last_error

    raise RuntimeError(str(last_error))


# =========================================================
# 4. SMART AI QUESTION PLANNER
# =========================================================

def create_analysis_plan(question, df):
    columns = [str(c) for c in df.columns]

    numeric_columns = [
        str(c)
        for c in df.select_dtypes(include=np.number).columns
    ]

    column_details = {
        str(c): {
            "type": str(df[c].dtype),
            "sample_values": (
                df[c].dropna().astype(str).head(5).tolist()
            ),
            "unique_count": int(df[c].nunique(dropna=True)),
            "missing_count": int(df[c].isna().sum()),
        }
        for c in df.columns
    }

    prompt = f"""
You are a careful general-purpose data analysis planner.

Convert the user question into one JSON object only.
Never return markdown or code.

Supported operations:
count_rows, count_missing, count_duplicates, count_unique,
describe, date_summary, sum, mean, median, min, max,
value_counts, group_aggregate, top_rows, bottom_rows,
correlation, preview, explain

JSON format:
{{
  "operation": "supported operation",
  "column": "exact column name or null",
  "group_by": "exact column name or null",
  "metric": "sum, mean, median, min, max, or count",
  "top_n": 5,
  "reason": "brief reason"
}}

Rules:
- Use only exact column names provided below.
- Never invent columns, values, or statistics.
- Use column names, data types and sample values to identify meaning.
- For total of a numeric column use sum.
- For average use mean.
- For median, minimum or maximum use median, min or max.
- For highest or lowest records use top_rows or bottom_rows.
- For category frequencies use value_counts.
- For unique/distinct counts use count_unique.
- For group comparisons use group_aggregate.
- For earliest/latest date use date_summary.
- For overall statistics use describe.
- For row count use count_rows.
- For missing values use count_missing.
- For duplicate rows use count_duplicates.
- For correlation use correlation.
- If a column is ambiguous, use explain.
- Group aggregation needs a numeric column and a group column.
- Return valid JSON only.

COLUMNS:
{json.dumps(columns)}

NUMERIC COLUMNS:
{json.dumps(numeric_columns)}

COLUMN DETAILS:
{json.dumps(column_details, default=str)}

QUESTION:
{question}
"""

    raw = gemini_request(prompt)
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)

    start = raw.find("{")
    end = raw.rfind("}")

    if start < 0 or end < 0:
        raise ValueError("Gemini did not return valid JSON.")

    plan = json.loads(raw[start:end + 1])

    allowed = {
        "count_rows",
        "count_missing",
        "count_duplicates",
        "count_unique",
        "describe",
        "date_summary",
        "sum",
        "mean",
        "median",
        "min",
        "max",
        "value_counts",
        "group_aggregate",
        "top_rows",
        "bottom_rows",
        "correlation",
        "preview",
        "explain",
    }

    if plan.get("operation") not in allowed:
        raise ValueError("Unsupported analysis operation.")

    return plan


# =========================================================
# 5. EXECUTE REAL DATA CALCULATIONS
# =========================================================

def execute_analysis_plan(plan, df):
    operation = plan["operation"]
    column = plan.get("column")
    group_by = plan.get("group_by")
    metric = plan.get("metric", "sum")

    if operation == "count_rows":
        return f"Total rows: {len(df):,}"

    if operation == "count_missing":
        result = df.isna().sum()
        result = result[result > 0].sort_values(ascending=False)

        if result.empty:
            return "No missing values found."

        return result.to_string()

    if operation == "count_duplicates":
        return f"Duplicate rows: {int(df.duplicated().sum()):,}"

    if operation == "count_unique":
        if column not in df.columns:
            return "Please specify a valid column."

        return (
            f"Unique values in {column}: "
            f"{df[column].nunique(dropna=True):,}"
        )

    if operation == "describe":
        return df.describe(include="all").transpose().to_string()

    if operation == "preview":
        return df.head(10).to_string(index=False)

    if operation == "explain":
        return (
            "I could not confidently identify the required calculation. "
            "Try using an exact column name from your dataset."
        )

    if operation == "correlation":
        numeric = df.select_dtypes(include=np.number)

        if numeric.shape[1] < 2:
            return "At least two numeric columns are required."

        return numeric.corr().round(3).to_string()

    if operation == "date_summary":
        if column not in df.columns:
            return "Please specify a valid date column."

        dates = pd.to_datetime(df[column], errors="coerce")
        valid = dates.dropna()

        if valid.empty:
            return f"No valid dates found in {column}."

        return (
            f"Column: {column}\n"
            f"Earliest: {valid.min()}\n"
            f"Latest: {valid.max()}\n"
            f"Valid dates: {len(valid):,}\n"
            f"Missing or invalid dates: {int(dates.isna().sum()):,}"
        )

    if operation in {"top_rows", "bottom_rows"}:
        if column not in df.columns:
            return "Please specify a valid column."

        if not pd.api.types.is_numeric_dtype(df[column]):
            return "Top/bottom rows require a numeric column."

        n = max(1, min(int(plan.get("top_n", 5)), 50))

        if operation == "top_rows":
            result = df.nlargest(n, column)
        else:
            result = df.nsmallest(n, column)

        return result.to_string(index=False)

    if operation == "value_counts":
        if column not in df.columns:
            return "Please specify a valid column."

        return (
            df[column]
            .value_counts(dropna=False)
            .head(20)
            .to_string()
        )

    if operation in {"sum", "mean", "median", "min", "max"}:
        if column not in df.columns:
            return "Please specify a valid column."

        if not pd.api.types.is_numeric_dtype(df[column]):
            return f"{column} is not a numeric column."

        series = df[column].dropna()

        if series.empty:
            return f"No usable numeric values found in {column}."

        functions = {
            "sum": series.sum,
            "mean": series.mean,
            "median": series.median,
            "min": series.min,
            "max": series.max,
        }

        result = functions[operation]()

        return (
            f"Operation: {operation}\n"
            f"Column: {column}\n"
            f"Result: {result:,.4f}"
        )

    if operation == "group_aggregate":
        if column not in df.columns or group_by not in df.columns:
            return (
                "Could not identify the metric or group column. "
                "Please clarify the column names."
            )

        if not pd.api.types.is_numeric_dtype(df[column]):
            return f"{column} must be numeric."

        if metric not in {"sum", "mean", "median", "min", "max", "count"}:
            return "Unsupported aggregation metric."

        working = df[[group_by, column]].copy()

        if metric == "count":
            result = (
                working.groupby(group_by, dropna=False)[column]
                .count()
                .sort_values(ascending=False)
                .head(20)
            )
        else:
            result = (
                working.groupby(group_by, dropna=False)[column]
                .agg(metric)
                .sort_values(ascending=False)
                .head(20)
            )

        return (
            f"{metric} of {column}, grouped by {group_by}\n\n"
            f"{result.to_string()}"
        )

    return "Unsupported analysis operation."


# =========================================================
# 6. SQL VALIDATION
# =========================================================

def validate_sql(query):
    cleaned = query.strip()

    if not cleaned:
        return False, "Enter a SQL query."

    if cleaned.endswith(";"):
        cleaned = cleaned[:-1].strip()

    if ";" in cleaned:
        return False, "Only one SQL statement is allowed."

    if not re.match(r"^(SELECT|WITH)\b", cleaned, re.I):
        return False, "Only SELECT queries are allowed."

    forbidden = re.compile(
        r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|"
        r"TRUNCATE|ATTACH|DETACH|COPY|INSTALL|LOAD|EXPORT|"
        r"IMPORT|PRAGMA|CALL|EXECUTE|SET|RESET|VACUUM|GRANT|REVOKE)\b",
        re.I,
    )

    if forbidden.search(cleaned):
        return False, "The query contains a restricted keyword."

    return True, cleaned


# =========================================================
# 7. APP HEADER AND FILE UPLOAD
# =========================================================

st.title("📊 Smart AI Data Analysis Agent")

st.write(
    "Upload CSV or Excel data, explore it, create charts, "
    "run SQL queries and ask AI questions about your dataset."
)

with st.sidebar:
    st.header("⚙️ Settings")

    if API_KEY:
        st.success("Gemini key detected")
    else:
        st.warning("Gemini key missing in .env")

uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "xls"],
)

if uploaded_file is None:
    st.info("Upload a CSV or Excel file to begin.")

    st.markdown(
        """
        **Features**
        - CSV and Excel upload
        - Data preview and quality checks
        - Statistics and interactive charts
        - SQL query lab
        - Smart AI analysis
        - Downloadable reports
        """
    )

    st.stop()


# =========================================================
# 8. READ THE UPLOADED DATASET
# =========================================================

try:
    filename = uploaded_file.name.lower()

    if filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        try:
            df = pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding="latin-1")

except Exception as error:
    st.error(f"Could not read the uploaded dataset: {error}")
    st.info(
        "Check the file format. For Excel support, install openpyxl."
    )
    st.stop()

if df.empty:
    st.warning("The dataset contains no data rows.")
    st.stop()

if len(df.columns) == 0:
    st.warning("No columns were found.")
    st.stop()

if not df.columns.is_unique:
    st.error("Duplicate column names found. Please rename them.")
    st.stop()

df = df.copy()

with st.sidebar:
    st.write(f"File: {uploaded_file.name}")
    st.write(f"Rows: {len(df):,}")
    st.write(f"Columns: {len(df.columns):,}")

    remove_duplicates = st.checkbox("Remove duplicate rows")

    if remove_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)


# =========================================================
# 9. OVERVIEW METRICS
# =========================================================

m1, m2, m3, m4 = st.columns(4)

m1.metric("Rows", f"{len(df):,}")
m2.metric("Columns", f"{len(df.columns):,}")
m3.metric("Missing Cells", f"{int(df.isna().sum().sum()):,}")
m4.metric("Duplicate Rows", f"{int(df.duplicated().sum()):,}")


# =========================================================
# 10. CREATE ALL TABS BEFORE USING THEM
# =========================================================

(
    tab_preview,
    tab_quality,
    tab_stats,
    tab_charts,
    tab_sql,
    tab_ai,
    tab_report,
) = st.tabs([
    "👀 Preview",
    "🧹 Data Quality",
    "📊 Statistics",
    "📈 Charts",
    "🧮 SQL Lab",
    "🤖 Smart AI Analysis",
    "📄 Report",
])


# =========================================================
# TAB 1: PREVIEW
# =========================================================

with tab_preview:
    st.subheader("Dataset Preview")

    st.dataframe(df.head(10), use_container_width=True)

    info = pd.DataFrame({
        "Column": [str(c) for c in df.columns],
        "Data Type": [str(t) for t in df.dtypes],
        "Missing": df.isna().sum().values,
        "Unique": df.nunique(dropna=True).values,
    })

    st.subheader("Column Information")
    st.dataframe(info, use_container_width=True)

    st.download_button(
        "Download Working Dataset",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="working_dataset.csv",
        mime="text/csv",
    )


# =========================================================
# TAB 2: DATA QUALITY
# =========================================================

with tab_quality:
    missing = pd.DataFrame({
        "Column": [str(c) for c in df.columns],
        "Missing Count": df.isna().sum().values,
        "Missing %": (df.isna().mean() * 100).round(2).values,
    })

    st.subheader("Missing Values")
    st.dataframe(missing, use_container_width=True)

    missing_plot = missing[missing["Missing Count"] > 0]

    if not missing_plot.empty:
        fig = px.bar(
            missing_plot,
            x="Column",
            y="Missing Count",
            title="Missing Values by Column",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No missing values found.")

    duplicates = int(df.duplicated().sum())

    if duplicates:
        st.warning(f"{duplicates:,} duplicate rows found.")
    else:
        st.success("No duplicate rows found.")


# =========================================================
# TAB 3: STATISTICS
# =========================================================

with tab_stats:
    numeric = df.select_dtypes(include=np.number)

    if numeric.empty:
        st.info("No numeric columns found.")
    else:
        st.subheader("Statistical Summary")
        st.dataframe(numeric.describe().T, use_container_width=True)

        if numeric.shape[1] >= 2:
            fig = px.imshow(
                numeric.corr(),
                text_auto=".2f",
                aspect="auto",
                title="Correlation Matrix",
            )
            st.plotly_chart(fig, use_container_width=True)

        chosen = st.selectbox(
            "Choose numeric column",
            numeric.columns.tolist(),
        )

        series = numeric[chosen].dropna()

        a, b, c = st.columns(3)

        a.metric(
            "Mean",
            f"{series.mean():,.2f}" if not series.empty else "N/A",
        )
        b.metric(
            "Median",
            f"{series.median():,.2f}" if not series.empty else "N/A",
        )
        c.metric(
            "Maximum",
            f"{series.max():,.2f}" if not series.empty else "N/A",
        )


# =========================================================
# TAB 4: CHARTS
# =========================================================

with tab_charts:
    columns = df.columns.tolist()
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

    chart_type = st.selectbox(
        "Chart type",
        ["Bar", "Line", "Scatter", "Histogram", "Box", "Pie"],
    )

    if chart_type in {"Bar", "Line", "Scatter"}:
        if len(columns) < 2:
            st.info("At least two columns are required.")
        else:
            x = st.selectbox("X-axis", columns, key="chart_x")

            y_options = [c for c in columns if c != x]

            y = st.selectbox(
                "Y-axis",
                y_options,
                key="chart_y",
            )

            color = st.selectbox(
                "Color grouping",
                ["None"] + columns,
                key="chart_color",
            )

            color = None if color == "None" else color

            try:
                if chart_type == "Bar":
                    fig = px.bar(df, x=x, y=y, color=color)
                elif chart_type == "Line":
                    fig = px.line(df, x=x, y=y, color=color)
                else:
                    fig = px.scatter(df, x=x, y=y, color=color)

                st.plotly_chart(fig, use_container_width=True)

            except Exception as error:
                st.error(f"Could not create chart: {error}")

    elif chart_type in {"Histogram", "Box"}:
        if not numeric_columns:
            st.info("No numeric columns available.")
        else:
            column = st.selectbox(
                "Numeric column",
                numeric_columns,
            )

            if chart_type == "Histogram":
                fig = px.histogram(df, x=column)
            else:
                fig = px.box(df, y=column)

            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Pie":
        category = st.selectbox("Category column", columns)

        pie_data = (
            df[category]
            .fillna("Missing")
            .astype(str)
            .value_counts()
            .head(15)
            .rename_axis("Category")
            .reset_index(name="Count")
        )

        fig = px.pie(
            pie_data,
            names="Category",
            values="Count",
        )

        st.plotly_chart(fig, use_container_width=True)


# =========================================================
# TAB 5: SQL LAB
# =========================================================

with tab_sql:
    st.subheader("DuckDB SQL Lab")
    st.caption("Use the table name: data")

    query = st.text_area(
        "SQL query",
        value="SELECT * FROM data LIMIT 10;",
        height=120,
    )

    if st.button("Run SQL"):
        valid, result = validate_sql(query)

        if not valid:
            st.error(result)
        else:
            connection = None

            try:
                connection = duckdb.connect(":memory:")
                connection.register("data", df)

                output = connection.execute(result).fetchdf()

                st.dataframe(output, use_container_width=True)

                st.download_button(
                    "Download SQL Results",
                    data=output.to_csv(index=False).encode("utf-8"),
                    file_name="sql_results.csv",
                    mime="text/csv",
                )

            except Exception as error:
                st.error(f"SQL error: {error}")

            finally:
                if connection is not None:
                    connection.close()


# =========================================================
# TAB 6: SMART AI ANALYSIS
# =========================================================

with tab_ai:
    st.subheader("🤖 Ask Questions About Your Data")

    st.write(
        "Ask questions in natural English. The AI creates an analysis "
        "plan, and supported calculations run on your actual dataset."
    )

    with st.expander("Example questions"):
        st.markdown(
            """
            - How many rows are in this dataset?
            - Which category appears most often?
            - What is the average of Revenue?
            - Show the top 5 records by Score.
            - Which department has the highest total?
            - How many unique values are in Region?
            - What are the earliest and latest dates?
            - Are there missing or duplicate values?
            - Show the statistical summary.
            - Show the correlation between numeric columns.
            """
        )

    question = st.text_area(
        "Ask a data question",
        placeholder=(
            "Example: What is the average of Revenue?\n"
            "Use exact column names where possible."
        ),
        height=130,
        key="smart_question",
    )

    if st.button("Analyze My Data"):
        if not question.strip():
            st.warning("Please enter a question.")

        elif client is None:
            st.error(
                "Gemini API key missing. Check GEMINI_API_KEY "
                "in your .env file."
            )

        else:
            try:
                with st.spinner("Understanding your question..."):
                    plan = create_analysis_plan(question, df)

                with st.expander("Analysis plan"):
                    st.json(plan)

                with st.spinner("Calculating from your dataset..."):
                    result = execute_analysis_plan(plan, df)

                st.subheader("Calculated Result")
                st.code(str(result))

                st.caption(
                    "Supported calculations are executed locally "
                    "using the uploaded dataset."
                )

            except Exception as error:
                st.error(f"Analysis failed: {error}")

                st.info(
                    "Try using exact column names. If Gemini is "
                    "temporarily unavailable, wait and retry."
                )


# =========================================================
# TAB 7: REPORT
# =========================================================

with tab_report:
    st.subheader("Data Analysis Report")

    lines = [
        "SMART AI DATA ANALYSIS REPORT",
        "=" * 40,
        f"File: {uploaded_file.name}",
        f"Rows: {len(df)}",
        f"Columns: {len(df.columns)}",
        f"Missing cells: {int(df.isna().sum().sum())}",
        f"Duplicate rows: {int(df.duplicated().sum())}",
        "",
        "COLUMN DETAILS",
        "-" * 40,
    ]

    for column in df.columns:
        lines.append(
            f"{column}: type={df[column].dtype}, "
            f"missing={int(df[column].isna().sum())}, "
            f"unique={int(df[column].nunique(dropna=True))}"
        )

    lines.extend(["", "NUMERIC SUMMARY", "-" * 40])

    numeric = df.select_dtypes(include=np.number)

    if numeric.empty:
        lines.append("No numeric columns found.")
    else:
        lines.append(numeric.describe().round(2).to_string())

    report = "\n".join(lines)

    st.text_area(
        "Report preview",
        report,
        height=350,
    )

    st.download_button(
        "Download Report",
        data=report,
        file_name="data_analysis_report.txt",
        mime="text/plain",
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Smart AI Data Analysis Agent | "
    "Python • Pandas • NumPy • Plotly • DuckDB • Gemini"
)
