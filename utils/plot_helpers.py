import plotly.express as px
import uuid
import os
import pandas as pd

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
OUTPUT_DIR = "workspace/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLOR_SEQ = px.colors.qualitative.Set2


# --------------------------------------------------
# NUMERIC NORMALIZATION
# --------------------------------------------------
def normalize_numeric_column(df: pd.DataFrame, col: str):
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# --------------------------------------------------
# VALIDATION
# --------------------------------------------------
def validate_columns(df: pd.DataFrame, cols: list[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


# --------------------------------------------------
# INDEXING (COMPARE MODE)
# --------------------------------------------------
def apply_indexing(df: pd.DataFrame, x: str, y: str, group_col: str):
    df = df.sort_values(by=x)

    def index_series(s):
        base = s.iloc[0]
        return (s / base) * 100 if base not in (0, None) else s

    df[y] = df.groupby(group_col)[y].transform(index_series)
    return df


# --------------------------------------------------
# MAIN PLOT ENGINE
# --------------------------------------------------
def generate_plot(intent: dict, df: pd.DataFrame):
    """
    FINAL, HARD-SAFE plot generator
    Guarantees:
    - Monthly trends are aggregated
    - Facets do NOT break grouping
    - No raw transaction plotting
    - Compare mode uses indexed trends
    """

    df = df.copy()
    df.columns = df.columns.str.strip()

    chart_type = intent.get("chart_type")
    x = intent.get("x_axis")
    y = intent.get("y_axis")
    agg = intent.get("aggregation") or "sum"
    color_by = intent.get("color_by")
    facet_by = intent.get("facet_by")
    index_mode = intent.get("index_mode", False)
    annotations = intent.get("annotations", [])
    y_axis_label = intent.get("y_axis_label")

    if not chart_type or not x or not y:
        raise ValueError("chart_type, x_axis and y_axis are required")

    # --------------------------------------------------
    # NUMERIC CLEANUP
    # --------------------------------------------------
    df = normalize_numeric_column(df, y)

    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------
    required_cols = [x, y]
    if color_by:
        required_cols.append(color_by)
    if facet_by:
        required_cols.append(facet_by)

    validate_columns(df, required_cols)

    # --------------------------------------------------
    # 🔒 HARD ENFORCED AGGREGATION (CRITICAL)
    # --------------------------------------------------
    if chart_type in {"line", "bar"}:
        group_cols = [x]

        if color_by:
            group_cols.append(color_by)

        if facet_by:
            group_cols.append(facet_by)

        df = (
            df.groupby(group_cols, as_index=False)[y]
            .agg(agg)
            .sort_values(by=x)
        )

    # --------------------------------------------------
    # 🔒 DATA GRAIN ASSERTION (BUG PREVENTION)
    # --------------------------------------------------
    if chart_type == "line":
        grain_cols = [x]
        if color_by:
            grain_cols.append(color_by)
        if facet_by:
            grain_cols.append(facet_by)

        dupes = df.duplicated(subset=grain_cols).sum()
        if dupes > 0:
            raise RuntimeError(
                f"Grain violation detected: {dupes} duplicate rows for {grain_cols}. "
                "Trend charts must be aggregated before plotting."
            )

    # --------------------------------------------------
    # 📈 INDEX MODE (COMPARE / RELATIVE)
    # --------------------------------------------------
    if chart_type == "line" and index_mode and (color_by or facet_by):
        group_col = color_by or facet_by
        df = apply_indexing(df, x, y, group_col)
        y_axis_label = "Indexed Sales (Base = 100)"

    # --------------------------------------------------
    # FIGURE
    # --------------------------------------------------
    if chart_type == "line":
        fig = px.line(
            df,
            x=x,
            y=y,
            color=color_by,
            facet_col=facet_by,
            line_group=color_by or facet_by,
            color_discrete_sequence=COLOR_SEQ,
        )

        # 🔒 FORCE LINE RENDERING
        fig.update_traces(
            mode="lines",
            line=dict(width=3),
        )

    elif chart_type == "bar":
        fig = px.bar(
            df,
            x=x,
            y=y,
            color=color_by,
            facet_col=facet_by,
            barmode="group",
            color_discrete_sequence=COLOR_SEQ,
        )

    elif chart_type == "scatter":
        fig = px.scatter(
            df,
            x=x,
            y=y,
            color=color_by,
            facet_col=facet_by,
            color_discrete_sequence=COLOR_SEQ,
        )

    elif chart_type == "histogram":
        fig = px.histogram(
            df,
            x=y,
            color=color_by,
            nbins=30,
            color_discrete_sequence=COLOR_SEQ,
        )

    else:
        raise ValueError(f"Unsupported chart type: {chart_type}")

    # --------------------------------------------------
    # CLEAN FACET TITLES
    # --------------------------------------------------
    if facet_by:
        fig.for_each_annotation(
            lambda a: a.update(text=a.text.split("=")[-1])
        )

    # --------------------------------------------------
    # LAYOUT
    # --------------------------------------------------
    fig.update_layout(
        template="plotly_white",
        height=520,
        margin=dict(t=70, l=55, r=40, b=60),
        legend_title=color_by,
        title=dict(text=f"{chart_type.title()} Chart", x=0.5),
    )

    fig.update_xaxes(tickangle=-45)
    fig.update_yaxes(title=y_axis_label or y)

    # --------------------------------------------------
    # ANNOTATIONS
    # --------------------------------------------------
    for ann in annotations:
        if ann.get("type") == "vline":
            fig.add_vline(x=ann["x"], line_dash="dash")
            fig.add_annotation(
                x=ann["x"],
                y=1.02,
                xref="x",
                yref="paper",
                text=ann.get("text", ""),
                showarrow=False,
            )

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------
    file_name = f"{uuid.uuid4()}.png"
    path = os.path.join(OUTPUT_DIR, file_name)
    fig.write_image(path)

    return path
