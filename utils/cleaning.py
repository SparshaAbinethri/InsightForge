import pandas as pd
import re
from typing import Dict, Any


# -------------------------------------------------
# 1. Column Standardization (WITH RENAME TRACKING)
# -------------------------------------------------
def standardize_column_names(df: pd.DataFrame):
    out = df.copy()
    rename_map = {}

    for col in out.columns:
        new_col = re.sub(
            r"[^a-zA-Z0-9_]",
            "",
            col.strip().lower().replace(" ", "_")
        )
        rename_map[col] = new_col

    out = out.rename(columns=rename_map)
    return out, rename_map


# -------------------------------------------------
# 2. Normalize Missing Tokens
# -------------------------------------------------
def normalize_missing_tokens(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    missing_tokens = ["", "na", "n/a", "null", "none", "nan", "-", "--", "unknown"]
    out = out.replace(to_replace=missing_tokens, value=pd.NA)
    out = out.replace(r"^\s*$", pd.NA, regex=True)
    return out


# -------------------------------------------------
# 3. Numeric Coercion
# -------------------------------------------------
def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in out.columns:
        if out[col].dtype == "object":
            cleaned = (
                out[col]
                .astype(str)
                .str.replace(r"[,$₹€]", "", regex=True)
                .str.replace(r"\s+", "", regex=True)
            )
            numeric = pd.to_numeric(cleaned, errors="coerce")

            # Convert only if majority values are numeric
            if numeric.notna().mean() >= 0.7:
                out[col] = numeric

    return out


# -------------------------------------------------
# 4. Date Coercion (Explicit Only)
# -------------------------------------------------
def coerce_date_columns(df: pd.DataFrame, date_cols=None, dayfirst=False) -> pd.DataFrame:
    out = df.copy()
    if not date_cols:
        return out

    for col in date_cols:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce", dayfirst=dayfirst)

    return out


# -------------------------------------------------
# 5. Text Standardization
# -------------------------------------------------
def standardize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include="object").columns:
        out[col] = (
            out[col]
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )
    return out


# -------------------------------------------------
# 6. Missing Value Handling
# -------------------------------------------------
def fill_missing_values(df: pd.DataFrame, cat_fill="Unknown") -> pd.DataFrame:
    out = df.copy()

    num_cols = out.select_dtypes(include="number").columns
    obj_cols = out.select_dtypes(exclude="number").columns

    if len(num_cols):
        out[num_cols] = out[num_cols].fillna(out[num_cols].median())

    if len(obj_cols):
        out[obj_cols] = out[obj_cols].fillna(cat_fill)

    return out


# -------------------------------------------------
# 7. Duplicate Detection
# -------------------------------------------------
def duplicate_report(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "exact_duplicates": int(df.duplicated().sum()),
        "duplicate_ratio": float(df.duplicated().mean())
    }


# -------------------------------------------------
# 8. Outlier Detection (IQR)
# -------------------------------------------------
def outlier_report(df: pd.DataFrame) -> Dict[str, Any]:
    outliers = {}
    for col in df.select_dtypes(include="number").columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers[col] = int(((df[col] < lower) | (df[col] > upper)).sum())
    return outliers


# -------------------------------------------------
# 9. AUDIT LOG (RENAME-SAFE — FIXED)
# -------------------------------------------------
def audit_log(before: pd.DataFrame, after: pd.DataFrame, rename_map: Dict[str, str]):
    dtype_changes = {}

    for old_col, new_col in rename_map.items():
        if old_col in before.columns and new_col in after.columns:
            before_dtype = str(before[old_col].dtype)
            after_dtype = str(after[new_col].dtype)

            if before_dtype != after_dtype:
                dtype_changes[old_col] = {
                    "renamed_to": new_col,
                    "before_dtype": before_dtype,
                    "after_dtype": after_dtype
                }

    return {
        "rows_before": len(before),
        "rows_after": len(after),
        "columns_before": len(before.columns),
        "columns_after": len(after.columns),
        "nulls_before": int(before.isna().sum().sum()),
        "nulls_after": int(after.isna().sum().sum()),
        "column_renames": rename_map,
        "dtype_changes": dtype_changes
    }


# -------------------------------------------------
# 10. ORCHESTRATOR
# -------------------------------------------------
def clean_dataframe(
    df: pd.DataFrame,
    cat_fill="Unknown",
    date_cols=None,
    dayfirst=False
) -> Dict[str, Any]:

    before = df.copy()

    out, rename_map = standardize_column_names(df)
    out = normalize_missing_tokens(out)
    out = coerce_numeric_columns(out)
    out = coerce_date_columns(out, date_cols, dayfirst)
    out = standardize_text_columns(out)
    out = fill_missing_values(out, cat_fill)

    return {
        "cleaned_df": out,
        "rename_map": rename_map,
        "duplicate_report": duplicate_report(out),
        "outlier_report": outlier_report(out),
        "audit_log": audit_log(before, out, rename_map)
    }
