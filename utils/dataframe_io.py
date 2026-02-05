import pandas as pd
from typing import Dict, Any


def extract_schema(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Returns a structured schema dict used by agents (Analyst/SQL/Visualization).
    Format:
    {
      "col1": {"dtype": "float64"},
      "col2": {"dtype": "datetime64[ns]"},
      ...
    }
    """
    schema: Dict[str, Dict[str, Any]] = {}
    for col, dtype in df.dtypes.items():
        schema[str(col)] = {"dtype": str(dtype)}
    return schema


def schema_to_text(schema: Dict[str, Dict[str, Any]], table_name: str = "data_table") -> str:
    """
    Optional helper for UI / prompts if you want a readable schema string.
    """
    lines = [f"Table name: {table_name}", "Columns:"]
    for col, meta in schema.items():
        lines.append(f"- {col} ({meta.get('dtype')})")
    return "\n".join(lines)
