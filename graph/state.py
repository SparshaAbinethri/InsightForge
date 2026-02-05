from typing import Optional, Dict, Any, List
import pandas as pd

class AppState(dict):
    # -----------------------------
    # Routing
    # -----------------------------
    role: str
    user_input: str

    # -----------------------------
    # Dataframes
    # -----------------------------
    raw_df: Optional[pd.DataFrame]
    cleaned_df: Optional[pd.DataFrame]
    dataframe: Optional[pd.DataFrame]

    # -----------------------------
    # Schema (CRITICAL)
    # -----------------------------
    dataframe_schema: Dict[str, Any]

    # -----------------------------
    # Data Engineer outputs
    # -----------------------------
    profile_report: Dict[str, Any]
    audit_log: List[str]
    dataset_summary: Dict[str, Any]
    column_report: List[Dict[str, Any]]
    change_summary: Dict[str, Any]
    generated_code: str

    # -----------------------------
    # Data Analyst outputs 
    # -----------------------------
    analysis_summary: str
    analysis_metadata: Dict[str, Any]

    # -----------------------------
    # SQL / Questionnaire / Viz
    # -----------------------------
    output: str
    plot_paths: List[str]
    plot_explanations: List[str]
