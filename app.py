import streamlit as st
import pandas as pd
import os

from graph.graph_builder import build_graph
from utils.dataframe_io import extract_schema

# --------------------------------------------------
# App Config
# --------------------------------------------------
st.set_page_config(layout="wide")
st.title("LangGraph Role-Based Data Application")

graph = build_graph()

# --------------------------------------------------
# Session State
# --------------------------------------------------
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None

if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None

if "dataframe_schema" not in st.session_state:
    st.session_state.dataframe_schema = None

if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

# --------------------------------------------------
# File Upload
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Data File",
    type=["csv", "xlsx", "json", "parquet"]
)

def load_dataframe(file):
    ext = os.path.splitext(file.name)[1].lower()
    if ext == ".csv":
        return pd.read_csv(file)
    elif ext == ".xlsx":
        return pd.read_excel(file)
    elif ext == ".json":
        return pd.read_json(file)
    elif ext == ".parquet":
        return pd.read_parquet(file)
    return None

# --------------------------------------------------
# Handle Upload
# --------------------------------------------------
if uploaded_file is not None:
    if uploaded_file.name != st.session_state.uploaded_filename:
        try:
            st.session_state.raw_df = load_dataframe(uploaded_file)
            st.session_state.cleaned_df = None
            st.session_state.dataframe_schema = None
            st.session_state.uploaded_filename = uploaded_file.name
            st.success(f"✅ Loaded raw dataset: {uploaded_file.name}")
        except Exception as e:
            st.error(f"❌ Failed to load file: {e}")
            st.stop()

# --------------------------------------------------
# Role Selector
# --------------------------------------------------
role = st.selectbox(
    "Select Role",
    ["Data Engineer", "Data Analyst", "SQL", "Visualization", "Questionnaire"]
)

roles_requiring_input = {
    "Data Analyst", "SQL", "Visualization", "Questionnaire"
}

# --------------------------------------------------
# User Input
# --------------------------------------------------
if role in roles_requiring_input:
    user_input = st.text_area("Enter your request")
else:
    user_input = ""
    st.info("ℹ️ Data Engineer runs automatically on raw data.")

show_debug = st.checkbox("Show debug output", value=False)

# --------------------------------------------------
# Run
# --------------------------------------------------
if uploaded_file and st.button("Run"):

    if role in roles_requiring_input and not user_input.strip():
        st.warning("Please enter a request.")
        st.stop()

    # -----------------------------
    # Data Routing
    # -----------------------------
    if role in ["Data Engineer", "Questionnaire"]:
        df = st.session_state.raw_df
        schema = extract_schema(df)  # raw schema OK here
    else:
        if st.session_state.cleaned_df is None:
            st.warning("⚠ Please run Data Engineer first to clean the data.")
            st.stop()

        df = st.session_state.cleaned_df
        schema = st.session_state.dataframe_schema  # ✅ TRUST ENGINEER

    result = graph.invoke({
        "role": role,
        "user_input": user_input,
        "dataframe": df,
        "raw_df": st.session_state.raw_df,
        "cleaned_df": st.session_state.cleaned_df,
        "dataframe_schema": schema
    })

    st.subheader("Output")

    # --------------------------------------------------
    # SQL
    # --------------------------------------------------
    if role == "SQL":
        st.code(result.get("output", ""), language="sql")

    # --------------------------------------------------
    # Visualization
    # --------------------------------------------------
    elif role == "Visualization":
        plot_paths = result.get("plot_paths", [])
        plot_explanations = result.get("plot_explanations", [])

        if not plot_paths:
            st.warning("No visualization generated.")
        else:
            for i, path in enumerate(plot_paths):
                col1, col2 = st.columns([3, 2])
                with col1:
                    st.image(os.path.abspath(path), use_container_width=True)
                with col2:
                    st.markdown("### Explanation")
                    st.write(plot_explanations[i])

    # --------------------------------------------------
    # Data Engineer
    # --------------------------------------------------
    elif role == "Data Engineer":
        st.markdown("## 📌 Dataset Overview")
        st.json(result.get("dataset_summary", {}))

        st.markdown("## 🔄 Raw → Cleaned Change Summary")
        st.json(result.get("change_summary", {}))

        st.markdown("## 🧱 Column-Level Details")
        column_report = result.get("column_report", [])
        if column_report:
            st.dataframe(pd.DataFrame(column_report), use_container_width=True)

        st.markdown("## 🧾 Cleaning Audit Log")
        for msg in result.get("audit_log", []):
            st.write(msg)

        st.markdown("## 🧪 Reproducible Data Engineering Code")
        st.code(result.get("generated_code", ""), language="python")

        if result.get("cleaned_df") is not None:
            st.session_state.cleaned_df = result["cleaned_df"]
            st.session_state.dataframe_schema = result["dataframe_schema"]

            st.markdown("## 🔍 Cleaned Data Preview")
            st.dataframe(result["cleaned_df"].head(20), use_container_width=True)

    # --------------------------------------------------
    # Data Analyst (FINAL FIX)
    # --------------------------------------------------
    elif role == "Data Analyst":
        st.markdown("## 📊 Analysis Summary")

        summary = result.get("analysis_summary")

        if summary:
            st.write(summary)
        else:
            st.error("❌ Analyst did not generate output.")
            st.json(result)

        if show_debug:
            st.markdown("## 🧪 Analysis Metadata")
            st.json(result.get("analysis_metadata", {}))

    # --------------------------------------------------
    # Questionnaire
    # --------------------------------------------------
    elif role == "Questionnaire":
        st.write(result.get("output", ""))
