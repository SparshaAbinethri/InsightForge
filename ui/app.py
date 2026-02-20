# --------------------------------------------------
# Fix Python Path (VERY IMPORTANT)
# --------------------------------------------------
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --------------------------------------------------
# Imports
# --------------------------------------------------
import streamlit as st
import pandas as pd
from graph.graph_builder import build_graph
from utils.dataframe_io import extract_schema

# --------------------------------------------------
# App Config
# --------------------------------------------------
st.set_page_config(
    page_title="InsightForge AI",
    layout="wide"
)

st.title("🚀 InsightForge – Role-Based AI Data Application")

# --------------------------------------------------
# Build LangGraph
# --------------------------------------------------
graph = build_graph()

# --------------------------------------------------
# Session State Initialization
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
# Sidebar – Role Selector
# --------------------------------------------------
st.sidebar.header("⚙️ Controls")

role = st.sidebar.selectbox(
    "Select Role",
    ["Data Engineer", "Data Analyst", "SQL", "Visualization", "Questionnaire"]
)

show_debug = st.sidebar.checkbox("Show Debug Output", value=False)

roles_requiring_input = {
    "Data Analyst",
    "SQL",
    "Visualization",
    "Questionnaire"
}

# --------------------------------------------------
# File Upload
# --------------------------------------------------
st.header("📂 Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV / Excel / JSON / Parquet",
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
            df = load_dataframe(uploaded_file)

            st.session_state.raw_df = df
            st.session_state.cleaned_df = None
            st.session_state.dataframe_schema = None
            st.session_state.uploaded_filename = uploaded_file.name

            st.success(f"✅ Loaded dataset: {uploaded_file.name}")
            st.dataframe(df.head(), use_container_width=True)

        except Exception as e:
            st.error(f"❌ Failed to load file: {e}")
            st.stop()

# --------------------------------------------------
# User Input
# --------------------------------------------------
st.header("📝 Request")

if role in roles_requiring_input:
    user_input = st.text_area("Enter your request")
else:
    user_input = ""
    st.info("ℹ️ Data Engineer runs automatically on raw data.")

# --------------------------------------------------
# Run Button
# --------------------------------------------------
if uploaded_file and st.button("▶ Run"):

    if role in roles_requiring_input and not user_input.strip():
        st.warning("Please enter a request.")
        st.stop()

    # -----------------------------
    # Data Routing Logic
    # -----------------------------
    if role in ["Data Engineer", "Questionnaire"]:
        df = st.session_state.raw_df
        schema = extract_schema(df)

    else:
        if st.session_state.cleaned_df is None:
            st.warning("⚠ Please run Data Engineer first.")
            st.stop()

        df = st.session_state.cleaned_df
        schema = st.session_state.dataframe_schema

    # -----------------------------
    # Invoke Graph
    # -----------------------------
    with st.spinner("Running AI Workflow..."):
        result = graph.invoke({
            "role": role,
            "user_input": user_input,
            "dataframe": df,
            "raw_df": st.session_state.raw_df,
            "cleaned_df": st.session_state.cleaned_df,
            "dataframe_schema": schema
        })

    st.success("✅ Execution Complete")

    st.divider()
    st.subheader("📤 Output")

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
                    if i < len(plot_explanations):
                        st.write(plot_explanations[i])

    # --------------------------------------------------
    # Data Engineer
    # --------------------------------------------------
    elif role == "Data Engineer":

        st.markdown("## 📌 Dataset Summary")
        st.json(result.get("dataset_summary", {}))

        st.markdown("## 🔄 Change Summary")
        st.json(result.get("change_summary", {}))

        st.markdown("## 🧱 Column Report")
        column_report = result.get("column_report", [])
        if column_report:
            st.dataframe(pd.DataFrame(column_report), use_container_width=True)

        st.markdown("## 🧾 Audit Log")
        for msg in result.get("audit_log", []):
            st.write(msg)

        st.markdown("## 🧪 Generated Cleaning Code")
        st.code(result.get("generated_code", ""), language="python")

        if result.get("cleaned_df") is not None:
            st.session_state.cleaned_df = result["cleaned_df"]
            st.session_state.dataframe_schema = result["dataframe_schema"]

            st.markdown("## 🔍 Cleaned Data Preview")
            st.dataframe(result["cleaned_df"].head(20), use_container_width=True)

    # --------------------------------------------------
    # Data Analyst
    # --------------------------------------------------
    elif role == "Data Analyst":

        summary = result.get("analysis_summary")
        if summary:
            st.markdown("## 📊 Analysis Summary")
            st.write(summary)
        else:
            st.error("❌ No analysis generated.")

        if show_debug:
            st.markdown("## 🧪 Analysis Metadata")
            st.json(result.get("analysis_metadata", {}))

    # --------------------------------------------------
    # Questionnaire
    # --------------------------------------------------
    elif role == "Questionnaire":
        st.write(result.get("output", ""))

    # --------------------------------------------------
    # Debug Mode
    # --------------------------------------------------
    if show_debug:
        st.markdown("## 🐞 Full Debug Output")
        st.json(result)
