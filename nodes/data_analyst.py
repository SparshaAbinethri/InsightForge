import re
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple


# ============================================================
# Custom Exception
# ============================================================
class AnalystRefusal(Exception):
    pass


# ============================================================
# KPI CATALOG
# ============================================================
KPI_CATALOG = {
    "revenue": {
        "keywords": ["revenue", "sales", "amount", "gmv", "turnover"],
        "aggregation": "sum",
        "preferred_cols": ["revenue", "sales", "amount", "total", "gmv", "price"],
    },
    "average_value": {
        "keywords": ["average", "avg", "mean", "aov"],
        "aggregation": "mean",
        "preferred_cols": ["amount", "sales", "revenue", "value", "price"],
    },
    "count": {
        "keywords": ["count", "how many", "number of", "total records"],
        "aggregation": "count",
        "preferred_cols": [],
    },
}


# ============================================================
# Helpers
# ============================================================
def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def success(meta: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": "success", **meta}


def build_schema_from_df(df: pd.DataFrame) -> Dict[str, Any]:
    return {c: {"dtype": str(df[c].dtype)} for c in df.columns}


def map_columns(schema: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
    numeric, categorical, datetime = [], [], []
    for col, meta in schema.items():
        t = meta.get("dtype", "").lower()
        if any(x in t for x in ["int", "float", "double", "decimal"]):
            numeric.append(col)
        elif "datetime" in t or "date" in t:
            datetime.append(col)
        else:
            categorical.append(col)
    return numeric, categorical, datetime


# ============================================================
# NLP Detection
# ============================================================
def detect_kpi(user_input: str) -> Optional[Dict[str, Any]]:
    text = normalize(user_input)

    if any(k in text for k in ["average", "avg", "mean"]) and any(
        k in text for k in ["revenue", "sales", "amount"]
    ):
        return {"kpi": "average_value", **KPI_CATALOG["average_value"]}

    for kpi, cfg in KPI_CATALOG.items():
        for kw in cfg["keywords"]:
            if kw in text:
                return {"kpi": kpi, **cfg}
    return None


def extract_topn(user_input: str) -> Optional[int]:
    text = normalize(user_input)
    m = re.search(r"\btop\s*(\d+)\b", text)
    if m:
        return int(m.group(1))
    if any(k in text for k in ["most", "highest", "largest", "top contributor"]):
        return 1
    return None


def detect_groupby_cols(user_input: str, categorical_cols: List[str]) -> List[str]:
    text = normalize(user_input)
    hits = []

    synonyms = {
        "sales person": "sales_person",
        "sales people": "sales_person",
        "salesperson": "sales_person",
    }

    for phrase, target in synonyms.items():
        if phrase in text:
            for c in categorical_cols:
                if normalize(c) == target:
                    hits.append(c)

    for c in categorical_cols:
        if normalize(c) in text:
            hits.append(c)

    if " by " in text:
        after = text.split(" by ", 1)[1]
        for c in categorical_cols:
            if normalize(c) in after:
                hits.append(c)

    return list(dict.fromkeys(hits))


def choose_metric_column(df, numeric_cols, preferred_cols):
    def is_id(col):
        return any(x in normalize(col) for x in ["id", "_id", "code", "number", "no"])

    ranked = []
    for col in numeric_cols:
        if is_id(col):
            continue
        score = sum(10 for p in preferred_cols if p in normalize(col))
        score += df[col].nunique(dropna=True) > 1
        ranked.append((score, col))

    ranked.sort(reverse=True)
    return ranked[0][1] if ranked and ranked[0][0] > 0 else None


def build_kpi_insight(metric, finding):
    return (
        f"Based on the available data, the analysis of {metric} shows that "
        f"{finding}. This is computed directly from the dataset."
    )


# ============================================================
# Core Analyst Logic
# ============================================================
def run_data_analyst(state: Dict[str, Any]) -> Dict[str, Any]:
    df = state.get("cleaned_df")
    user_input = state.get("user_input", "")

    if df is None or df.empty:
        raise AnalystRefusal("Data is not available for analysis.")

    schema = state.get("dataframe_schema") or build_schema_from_df(df)
    numeric_cols, categorical_cols, datetime_cols = map_columns(schema)
    text = normalize(user_input)

    # --------------------------------------------------
    # Trend guard
    # --------------------------------------------------
    if any(k in text for k in ["trend", "monthly", "over time"]):
        if not datetime_cols:
            raise AnalystRefusal(
                "Trend analysis requires a date/time column, but none was found in the dataset."
            )

    # --------------------------------------------------
    # INSIGHTS / DISTRIBUTION (🔥 missing earlier)
    # --------------------------------------------------
    if any(k in text for k in ["insight", "concentration", "skew", "outlier", "extreme"]):
        insights = []

        if categorical_cols:
            c = categorical_cols[0]
            top_share = df[c].value_counts(normalize=True).iloc[0] * 100
            insights.append(
                f"Most records are concentrated in '{df[c].value_counts().idxmax()}' under '{c}', "
                f"accounting for ~{round(top_share,1)}% of rows."
            )

        if numeric_cols:
            n = numeric_cols[0]
            skew = df[n].skew()
            if abs(skew) > 1:
                insights.append(f"The '{n}' distribution is highly skewed.")
            else:
                insights.append(f"The '{n}' distribution is fairly balanced.")

            q1, q3 = df[n].quantile([0.25, 0.75])
            iqr = q3 - q1
            outliers = df[(df[n] < q1 - 1.5 * iqr) | (df[n] > q3 + 1.5 * iqr)]
            insights.append(f"Detected {len(outliers)} potential outliers in '{n}'.")

        return {
            "analysis_summary": " ".join(insights),
            "analysis_metadata": success({"analysis_type": "insight"}),
        }

    # --------------------------------------------------
    # Data quality
    # --------------------------------------------------
    if any(k in text for k in ["null", "missing", "duplicate", "quality"]):
        return {
            "analysis_summary": (
                f"Missing values per column: {df.isna().sum().to_dict()}. "
                f"Duplicate rows: {int(df.duplicated().sum())}."
            ),
            "analysis_metadata": success({"analysis_type": "data_quality"}),
        }

    # --------------------------------------------------
    # Dataset summary
    # --------------------------------------------------
    if any(k in text for k in ["summary", "summarize", "overview"]):
        return {
            "analysis_summary": (
                f"Dataset summary: rows={len(df)}, columns={len(df.columns)}. "
                f"Columns include {list(df.columns)}."
            ),
            "analysis_metadata": success({"analysis_type": "dataset_summary"}),
        }

    # --------------------------------------------------
    # Distinct
    # --------------------------------------------------
    if "distinct" in text or "unique" in text:
        for c in categorical_cols:
            if normalize(c) in text:
                return {
                    "analysis_summary": f"There are {df[c].nunique()} unique values in {c}.",
                    "analysis_metadata": success({"analysis_type": "distinct"}),
                }

    # --------------------------------------------------
    # KPI
    # --------------------------------------------------
    kpi = detect_kpi(user_input)
    if not kpi:
        raise AnalystRefusal("I couldn't detect a KPI request (e.g., revenue, profit, average).")

    metric_col = choose_metric_column(df, numeric_cols, kpi["preferred_cols"])
    if not metric_col:
        raise AnalystRefusal("A valid business metric column could not be identified.")

    s = pd.to_numeric(df[metric_col], errors="coerce")
    topn = extract_topn(user_input)
    groupby_cols = detect_groupby_cols(user_input, categorical_cols)

    if topn and not groupby_cols and categorical_cols:
        groupby_cols = [categorical_cols[0]]

    if groupby_cols:
        grouped = (
            df.assign(_metric=s)
            .groupby(groupby_cols)["_metric"]
            .agg(kpi["aggregation"])
            .sort_values(ascending=False)
        )
        if topn:
            grouped = grouped.head(topn)

        preview = "; ".join(f"{k}={round(float(v),2)}" for k, v in grouped.items())
        return {
            "analysis_summary": build_kpi_insight(
                kpi["kpi"],
                f"'{metric_col}' {kpi['aggregation']} by {groupby_cols}: {preview}",
            ),
            "analysis_metadata": success({"analysis_type": "kpi_grouped"}),
        }

    value = round(float(getattr(s, kpi["aggregation"])()), 2)
    return {
        "analysis_summary": build_kpi_insight(
            kpi["kpi"], f"the computed value is {value} (using '{metric_col}')"
        ),
        "analysis_metadata": success({"analysis_type": "kpi"}),
    }


def data_analyst_node(state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return run_data_analyst(state)
    except AnalystRefusal as e:
        return {
            "analysis_summary": str(e),
            "analysis_metadata": {"status": "refused"},
        }


__all__ = ["data_analyst_node"]
