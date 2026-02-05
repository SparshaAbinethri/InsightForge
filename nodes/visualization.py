import env_setup
from langchain_openai import ChatOpenAI
import json
import pandas as pd
import re

from utils.plot_helpers import generate_plot

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --------------------------------------------------
# AGGREGATION NORMALIZER
# --------------------------------------------------
def normalize_aggregation(agg):
    if not isinstance(agg, str):
        return None

    agg = agg.lower().strip()

    VALID = {"sum", "mean", "count", "min", "max", "median"}
    ALIAS = {
        "avg": "mean",
        "average": "mean",
        "total": "sum",
    }

    if agg in ALIAS:
        return ALIAS[agg]
    if agg in VALID:
        return agg

    return None


def safe_json_load(text: str):
    if "{" in text and "}" in text:
        text = text[text.find("{"): text.rfind("}") + 1]
    return json.loads(text)


# --------------------------------------------------
# DATE COLUMN DETECTION
# --------------------------------------------------
COMMON_DATE_COLUMNS = [
    "date", "order_date", "order date",
    "created_at", "created at",
    "transaction_date", "transaction date",
    "purchase_date", "purchase date",
    "invoice_date", "invoice date",
]


def detect_date_column(df: pd.DataFrame):
    normalized = {c.lower().strip(): c for c in df.columns}
    for c in COMMON_DATE_COLUMNS:
        if c in normalized:
            return normalized[c]

    for c in df.columns:
        parsed = pd.to_datetime(df[c], errors="coerce")
        if parsed.notna().sum() > 0:
            return c
    return None


# --------------------------------------------------
# MAIN VISUALIZATION NODE
# --------------------------------------------------
def visualization_node(state):
    user_question = state["user_input"]
    df = state["dataframe"]

    if df is None or df.empty:
        return {"output": "Dataset is empty. Cannot generate visualization."}

    df = df.copy()
    df.columns = df.columns.str.strip()
    q = user_question.lower()

    # --------------------------------------------------
    # GENERIC NUMERIC CLEANUP
    # --------------------------------------------------
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.strip()
            )

    # --------------------------------------------------
    # DATE NORMALIZATION
    # --------------------------------------------------
    date_col = detect_date_column(df)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # --------------------------------------------------
    # 1️⃣ INTENT PARSING
    # --------------------------------------------------
    intent_prompt = f"""
Return ONLY valid JSON:
- x_axis
- y_axis
- aggregation (optional)

Question:
{user_question}
"""

    try:
        intent = safe_json_load(llm.invoke(intent_prompt).content)
    except Exception:
        intent = {}

    x = intent.get("x_axis")
    y = intent.get("y_axis")

    # --------------------------------------------------
    # 2️⃣ SCHEMA NORMALIZATION
    # --------------------------------------------------
    col_map = {c.lower(): c for c in df.columns}

    metric_aliases = {
        "sales": "Amount",
        "revenue": "Amount",
        "total_sales": "Amount",
        "amount": "Amount",
        "boxes": "Boxes Shipped",
        "boxes_shipped": "Boxes Shipped",
    }

    if isinstance(x, str) and x.lower() in col_map:
        x = col_map[x.lower()]

    if isinstance(y, str):
        yl = y.lower()
        if yl in metric_aliases and metric_aliases[yl] in df.columns:
            y = metric_aliases[yl]
        elif yl in col_map:
            y = col_map[yl]

    intent["x_axis"] = x
    intent["y_axis"] = y

    # --------------------------------------------------
    # 3️⃣ AGGREGATION NORMALIZATION
    # --------------------------------------------------
    intent["aggregation"] = normalize_aggregation(intent.get("aggregation"))

    # --------------------------------------------------
    # 4️⃣ TIME BUCKETING + FORCED AGGREGATION
    # --------------------------------------------------
    if date_col and "month" in q:
        df["YearMonth"] = df[date_col].dt.to_period("M").astype(str)
        intent["x_axis"] = "YearMonth"
        intent["aggregation"] = intent["aggregation"] or "sum"

    elif date_col and ("year" in q or "annual" in q):
        df["Year"] = df[date_col].dt.year
        intent["x_axis"] = "Year"
        intent["aggregation"] = intent["aggregation"] or "sum"

    # --------------------------------------------------
    # 5️⃣ CHART TYPE
    # --------------------------------------------------
    if any(k in q for k in ["trend", "over time", "monthly", "yearly", "annual"]):
        intent["chart_type"] = "line"
    elif any(k in q for k in ["distribution", "histogram"]):
        intent["chart_type"] = "histogram"
    elif any(k in q for k in ["vs", "correlation", "relationship"]):
        intent["chart_type"] = "scatter"
    else:
        intent["chart_type"] = "bar"

    # --------------------------------------------------
    # 6️⃣ GROUPING
    # --------------------------------------------------
    color_by = None

    if "country" in q and "Country" in df.columns:
        color_by = "Country"
    elif "product" in q and "Product" in df.columns:
        color_by = "Product"
    elif "sales person" in q and "Sales Person" in df.columns:
        color_by = "Sales Person"

    intent["color_by"] = color_by
    intent["facet_by"] = None

    # --------------------------------------------------
    # 7️⃣ AUTO INDEX MODE (COMPARE / RELATIVE)
    # --------------------------------------------------
    index_mode = False
    if any(k in q for k in ["compare", "relative", "growth"]):
        index_mode = True

    intent["index_mode"] = index_mode

    # --------------------------------------------------
    # 8️⃣ FINAL VALIDATION
    # --------------------------------------------------
    if intent["x_axis"] not in df.columns or intent["y_axis"] not in df.columns:
        return {
            "output": f"Columns '{intent['x_axis']}' or '{intent['y_axis']}' not found."
        }

    # --------------------------------------------------
    # 9️⃣ GENERATE PLOT
    # --------------------------------------------------
    image_path = generate_plot(intent, df)

    # --------------------------------------------------
    # 🔟 INSIGHTS
    # --------------------------------------------------
    insight_prompt = f"""
Give 3 concise business insights.

Question: {user_question}
Chart specs: {intent}
"""
    insights = llm.invoke(insight_prompt).content.strip()

    return {
        "plot_paths": [image_path],
        "plot_explanations": [insights],
        "plot_specs": [intent],
        "output": "Visualization generated successfully."
    }
