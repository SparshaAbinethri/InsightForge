import env_setup  # must be first
from langchain_openai import ChatOpenAI
import re
from typing import Any, Dict, List, Optional, Tuple

# ============================================================
# LLM
# ============================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ============================================================
# Schema Normalization + Parsing
# ============================================================

def _normalize_schema(schema: Any) -> str:
    """
    Always convert schema into safe string format (prevents regex crashes).
    Supports:
    - str
    - dict: {table_name, columns}
    - list/tuple
    - fallback: str(schema)
    """
    if schema is None:
        return ""
    if isinstance(schema, str):
        return schema
    if isinstance(schema, dict):
        table = schema.get("table_name") or schema.get("table") or schema.get("name") or "data_table"
        cols = schema.get("columns") or schema.get("cols") or schema.get("fields") or []
        lines: List[str] = []
        if isinstance(cols, list):
            for c in cols:
                if isinstance(c, str):
                    lines.append(f"- {c}")
                elif isinstance(c, dict):
                    name = c.get("name") or c.get("column") or c.get("field")
                    ctype = c.get("type") or c.get("dtype")
                    if name and ctype:
                        lines.append(f"- {name} ({ctype})")
                    elif name:
                        lines.append(f"- {name}")
        col_text = "\n".join(lines)
        return f"Table name: {table}\nColumns:\n{col_text}".strip()
    if isinstance(schema, (list, tuple)):
        return "\n".join(str(x) for x in schema)
    return str(schema)


def _extract_table_name(schema_text: str) -> str:
    m = re.search(r"Table name:\s*([A-Za-z_][A-Za-z0-9_]*)", schema_text or "")
    return m.group(1) if m else "data_table"


def _extract_columns(schema_text: str) -> List[str]:
    """
    Best-effort column extraction from schema text.
    Supports:
      - ColName
      - ColName (TYPE)
    """
    schema_text = schema_text or ""
    cols = set()

    # bullet list: - Amount (DECIMAL)
    for m in re.finditer(r"^\s*-\s*([A-Za-z_][A-Za-z0-9_]*)\b", schema_text, re.M):
        cols.add(m.group(1))

    # inline columns: Columns: A, B, C
    m = re.search(r"Columns:\s*(.+)", schema_text, re.I)
    if m:
        tail = m.group(1)
        for token in re.split(r"[,\|]", tail):
            token = token.strip()
            token = re.sub(r"\(.*?\)", "", token).strip()
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", token):
                cols.add(token)

    return sorted(cols)


# ============================================================
# Question detectors (semantic triggers)
# ============================================================

def _q_lower(q: str) -> str:
    return (q or "").strip().lower()

def _is_global_best_question(q: str) -> bool:
    t = _q_lower(q)
    return any(k in t for k in [
        "highest", "maximum", "max", "best", "largest", "greatest"
    ]) and not any(k in t for k in [
        "top 3", "top 5", "top 10", "for each", "per ", "within each"
    ])


def _is_growth_question(q: str) -> bool:
    t = _q_lower(q)
    return any(k in t for k in [
        "growth", "growth rate", "percentage growth", "percent increase",
        "mom", "month over month", "month-over-month", "month to month",
        "yoy", "year over year", "year-over-year", "previous year", "last year",
        "compare to last month", "compared to last month", "compared to last year"
    ])


def _is_ranking_question(q: str) -> bool:
    t = _q_lower(q)
    return any(k in t for k in [
        "top", "highest", "lowest", "best", "rank", "ranking",
        "for each", "per ", "within each",
        "by country", "by product", "by category", "by year", "by month"
    ])


def _is_streak_question(q: str) -> bool:
    t = _q_lower(q)
    return any(k in t for k in [
        "consecutive", "in a row", "streak", "back to back", "back-to-back",
        "continuous", "months of decline", "months of decrease", "months of increase",
        "consecutive months"
    ])


def _wants_monthly(q: str) -> bool:
    t = _q_lower(q)
    return any(k in t for k in ["month", "monthly", "mom", "month over month", "month-to-month", "%y-%m", "%Y-%m"])


def _wants_yearly(q: str) -> bool:
    t = _q_lower(q)
    return any(k in t for k in ["year", "yearly", "annual", "yoy", "year over year", "previous year", "last year"])


# ============================================================
# SQL text helpers
# ============================================================
def _has_window_alias_filtered_same_level(sql: str) -> bool:
    """
    Detect illegal MySQL pattern:
    Window alias filtered in same SELECT level.

    Example (INVALID):

    SELECT ...,
           DENSE_RANK() OVER (...) AS rnk
    FROM data_table
    HAVING rnk <= 3
    """

    s = sql or ""

    alias_match = re.search(
        r"(DENSE_RANK|ROW_NUMBER|RANK)\s*\(.*?\)\s*OVER\s*\(.*?\)\s+AS\s+([A-Za-z_][A-Za-z0-9_]*)",
        s,
        re.I | re.S
    )

    if not alias_match:
        return False

    alias = alias_match.group(2)

    return bool(re.search(
        rf"\b(WHERE|HAVING)\b.*\b{alias}\b",
        s,
        re.I | re.S
    ))
    
def _strip_markdown(sql: str) -> str:
    s = (sql or "").strip()
    fence = re.match(r"^```(?:sql)?\s*(.*?)\s*```$", s, re.I | re.S)
    if fence:
        return fence.group(1).strip()
    s = re.sub(r"```(?:sql)?", "", s, flags=re.I)
    return s.replace("```", "").strip()


def _force_table_name(sql: str, table_name: str) -> str:
    """
    Force FIRST FROM clause to use correct table name.
    """
    if not sql:
        return sql
    return re.sub(
        r"\bFROM\b\s+[`\"]?[A-Za-z_][A-Za-z0-9_]*[`\"]?",
        f"FROM {table_name}",
        sql,
        count=1,
        flags=re.I
    )


def _needs_amount_cast(schema_text: str) -> bool:
    return bool(re.search(r"\bAmount\b", schema_text or "", re.I))


def _inject_amount_cast(sql: str) -> str:
    """
    Conservative amount cast:
    - Only patches SUM(Amount) (most common)
    - Does NOT globally replace Amount (avoids double-cast)
    """
    if not sql:
        return sql
    return re.sub(
        r"\bSUM\s*\(\s*Amount\s*\)",
        "SUM(CAST(REPLACE(REPLACE(Amount,'$',''),',','') AS DECIMAL(10,2)))",
        sql,
        flags=re.I
    )


def _has_any(sql: str, patterns: List[str]) -> bool:
    return any(re.search(p, sql or "", re.I | re.S) for p in patterns)


def _count_from_tables(sql: str) -> int:
    s = sql or ""
    n_from = len(re.findall(r"\bFROM\b", s, re.I))
    n_join = len(re.findall(r"\bJOIN\b", s, re.I))
    return n_from + n_join


def _has_window_in_where_or_having(sql: str) -> bool:
    return bool(re.search(
        r"\b(WHERE|HAVING)\b.*\b(LAG|LEAD|ROW_NUMBER|RANK|DENSE_RANK)\s*\(",
        sql or "",
        re.I | re.S
    ))


def _has_window_over_aggregate_same_level(sql: str) -> bool:
    """
    catches: LAG(SUM(x)) OVER ...
    """
    return bool(re.search(r"\b(LAG|LEAD)\s*\(\s*(SUM|AVG|COUNT|MIN|MAX)\s*\(", sql or "", re.I))


def _has_qualify_or_filter_distincton(sql: str) -> bool:
    return _has_any(sql, [
        r"\bQUALIFY\b",
        r"\bDISTINCT\s+ON\b",
        r"\bFILTER\s*\(",
        r"\bWINDOW\b\s+\w+\s+AS\b"  # named WINDOW clause
    ])


def _has_cte(sql: str) -> bool:
    return bool(re.match(r"^\s*WITH\b", sql or "", re.I))


def _has_limit(sql: str) -> bool:
    return bool(re.search(r"\bLIMIT\b", sql or "", re.I))


def _has_ranking_window(sql: str) -> bool:
    return bool(re.search(r"\b(DENSE_RANK|ROW_NUMBER|RANK)\s*\(", sql or "", re.I))


def _has_partition_by(sql: str) -> bool:
    return bool(re.search(r"\bPARTITION\s+BY\b", sql or "", re.I))


def _has_nullif(sql: str) -> bool:
    return "NULLIF" in (sql or "").upper()


def _has_division(sql: str) -> bool:
    # rough: any a / b
    return bool(re.search(r"[A-Za-z0-9_\)\]]\s*/\s*[A-Za-z0-9_\(]", sql or "", re.I))


def _looks_like_date_column(schema_text: str) -> Optional[str]:
    schema_text = schema_text or ""
    for cand in ["Date", "date", "OrderDate", "order_date", "TransactionDate", "transaction_date"]:
        if re.search(rf"\b{re.escape(cand)}\b", schema_text):
            return cand
    return None


def _mentions_str_to_date(sql: str) -> bool:
    return "STR_TO_DATE" in (sql or "").upper()


def _mentions_date_format(sql: str) -> bool:
    return "DATE_FORMAT" in (sql or "").upper()


def _mentions_gap_island(sql: str) -> bool:
    return bool(re.search(r"ROW_NUMBER\s*\(.*?\)\s*-\s*ROW_NUMBER\s*\(", sql or "", re.I | re.S))


def _has_alias_in_having(sql: str) -> bool:
    """
    Completely ban HAVING usage for ranking/window queries.
    """
    s = sql or ""

    # If HAVING exists at all AND ranking window exists → invalid
    if "HAVING" in s.upper() and re.search(r"\b(DENSE_RANK|ROW_NUMBER|RANK)\s*\(", s, re.I):
        return True

    # If HAVING references ranking aliases
    return bool(re.search(
        r"\bHAVING\b\s+.*\b(rank|rnk|rn)\b",
        s,
        re.I | re.S
    ))

def _has_alias_in_where(sql: str) -> bool:
    """
    Only block illegal same-level window misuse.
    Alias filtering in OUTER SELECT is valid.
    So we do not flag aliases here anymore.
    """
    return False

def _unbalanced_parentheses(sql: str) -> bool:
    s = sql or ""
    return s.count("(") != s.count(")")


def _has_multiple_statements(sql: str) -> bool:
    """
    For safety: reject multiple statements split by semicolon.
    Allow final trailing semicolon.
    """
    s = (sql or "").strip()
    if not s:
        return False
    # remove a single trailing semicolon
    s2 = s[:-1] if s.endswith(";") else s
    return ";" in s2


def _has_dangerous_sql(sql: str) -> bool:
    """
    Ban any DDL/DML statements (we only want SELECT queries).
    """
    return _has_any(sql, [
        r"\bINSERT\b", r"\bUPDATE\b", r"\bDELETE\b", r"\bDROP\b", r"\bALTER\b",
        r"\bTRUNCATE\b", r"\bCREATE\b", r"\bREPLACE\b", r"\bMERGE\b",
        r"\bGRANT\b", r"\bREVOKE\b", r"\bCALL\b", r"\bEXEC\b", r"\bEXECUTE\b"
    ])


def _ranking_window_missing_order_by(sql: str) -> bool:
    """
    Window ranking requires ORDER BY inside OVER(...).
    Very rough check: if ranking exists but no ORDER BY anywhere inside OVER blocks.
    """
    s = sql or ""
    if not _has_ranking_window(s):
        return False
    # ensure at least one OVER(...) contains ORDER BY
    overs = re.findall(r"OVER\s*\((.*?)\)", s, flags=re.I | re.S)
    if not overs:
        return True
    return all("ORDER BY" not in o.upper() for o in overs)


def _has_outer_rank_filter(sql: str) -> bool:
    """
    Ensure rank filtering happens AFTER window computation.

    Accept valid patterns like:

    WITH ...
    SELECT ...
    FROM (
        SELECT ..., DENSE_RANK() OVER(...) AS rnk
        FROM ...
    ) ranked
    WHERE rnk <= 3;
    """

    s = sql or ""

    # Must use CTE layering
    if not re.match(r"^\s*WITH\b", s, re.I):
        return False

    # Ensure rank filter exists in outermost WHERE
    # We look for final WHERE rnk <= N (or rank/rn)
    return bool(re.search(
        r"\bWHERE\s+(rank|rnk|rn)\s*<=\s*\d+",
        s,
        re.I
    ))


def _group_by_mismatch(sql: str) -> bool:
    """
    Conservative check for obvious GROUP BY mismatch.
    Not a SQL parser. Catches many common LLM mistakes.
    """
    s = sql or ""
    if "GROUP BY" not in s.upper():
        return False

    m_sel = re.search(r"\bSELECT\b(.*?)\bFROM\b", s, re.I | re.S)
    m_grp = re.search(r"\bGROUP\s+BY\b(.*?)(\bHAVING\b|\bORDER\b|\bLIMIT\b|$)", s, re.I | re.S)
    if not m_sel or not m_grp:
        return False

    sel = m_sel.group(1)
    grp = m_grp.group(1)

    # Replace function calls with FUNC( to reduce false positives
    sel_clean = re.sub(
        r"\b(SUM|AVG|COUNT|MIN|MAX|CAST|REPLACE|NULLIF|STR_TO_DATE|DATE_FORMAT|YEAR|MONTH)\s*\(",
        "FUNC(",
        sel,
        flags=re.I
    )

    ids = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", sel_clean)
    ignore = {
        "select","distinct","as","func","over","partition","by","order","desc","asc",
        "dense_rank","rank","row_number","lag","lead","case","when","then","else","end"
    }
    grp_ids = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", grp))

    for ident in ids:
        if ident.lower() in ignore:
            continue
        # If SELECT contains identifier and it's not aggregated (we tried to remove funcs),
        # and it's not in GROUP BY, likely mismatch.
        if ident not in grp_ids:
            return True

    return False


# ============================================================
# Validator: returns list of violations (rules)
# ============================================================

def _validate_sql(
    sql: str,
    schema_text: str,
    table_name: str,
    question: str,
    columns: List[str]
) -> List[str]:
    v: List[str] = []
    s = sql or ""
    su = s.upper()

    # --------------------------------------------------------
    # 0) Basic safety: only SELECT, only single statement
    # --------------------------------------------------------
    if not re.search(r"^\s*(WITH\b|SELECT\b)", s, re.I):
        v.append("Query must start with WITH or SELECT (SELECT-only engine).")

    if _has_multiple_statements(s):
        v.append("Multiple SQL statements detected. Only ONE statement is allowed.")

    if _has_dangerous_sql(s):
        v.append("Non-SELECT statement detected (DDL/DML). Only SELECT queries are allowed.")

    if _unbalanced_parentheses(s):
        v.append("Unbalanced parentheses detected.")

    # --------------------------------------------------------
    # 1) Table enforcement
    # --------------------------------------------------------
    if not re.search(rf"\bFROM\b\s+{re.escape(table_name)}\b", s, re.I):
        v.append(f"FROM clause must use table name exactly '{table_name}'.")

    # single-table enforcement
    if _count_from_tables(s) > 1:
        v.append("Only single-table queries allowed. Do NOT use JOINs or additional tables.")

    # Avoid SELECT *
    if re.search(r"\bSELECT\s+\*\b", s, re.I):
        v.append("SELECT * is not allowed. Select only required columns explicitly.")

    # --------------------------------------------------------
    # 2) MySQL unsupported features
    # --------------------------------------------------------
    if _has_qualify_or_filter_distincton(s):
        v.append("MySQL 8+ does NOT support QUALIFY, FILTER(), DISTINCT ON, or named WINDOW clauses.")

    # --------------------------------------------------------
    # 3) Window safety rules
    # --------------------------------------------------------
    if _has_window_in_where_or_having(s):
        v.append("MySQL does NOT allow filtering window function outputs in WHERE/HAVING. Compute in CTE and filter in outer SELECT.")
    
    if _has_window_alias_filtered_same_level(s):
        v.append("Window function alias cannot be filtered in same SELECT. Compute ranking in CTE and filter in outer SELECT.")
    
    if _has_alias_in_having(s) or _has_alias_in_where(s):
        v.append("Do NOT filter on derived/window aliases in WHERE/HAVING. Use an outer SELECT/CTE to filter.")

    if _has_window_over_aggregate_same_level(s):
        v.append("Do NOT use window functions over aggregates in the same SELECT (e.g., LAG(SUM(x))). Aggregate first in a CTE, then window.")

    # If any OVER(...) exists, require CTE layering (prevents many MySQL illegal patterns)
    if re.search(r"\bOVER\s*\(", s, re.I) and not _has_cte(s):
        v.append("Query uses window functions; enforce CTE layering (aggregate first, then window in next CTE).")

    # ranking windows must have ORDER BY inside OVER
    if _ranking_window_missing_order_by(s):
        v.append("Ranking window must include ORDER BY inside OVER(...).")

    # --------------------------------------------------------
# --------------------------------------------------------
#     # --------------------------------------------------------
    # 4) Ranking rules
    # --------------------------------------------------------
    if _is_ranking_question(question):

        if "HAVING" in su:
            v.append("HAVING is forbidden for ranking queries.")

        if _has_limit(s):
            v.append("Do NOT use LIMIT for per-group ranking.")

        if not _has_ranking_window(s):
            v.append("Ranking requires ROW_NUMBER/RANK/DENSE_RANK.")

        if _has_ranking_window(s) and not _has_partition_by(s):
            v.append("Per-group ranking requires PARTITION BY.")

        if _is_global_best_question(question) and re.search(r"\bROW_NUMBER\s*\(", s, re.I):
            v.append("Use DENSE_RANK() for tie-safe ranking.")

        if _has_ranking_window(s) and not _has_outer_rank_filter(s):
            v.append("Rank filtering must happen in outer SELECT.")

        # Product enforcement
        if "product" in question.lower():
            if not re.search(r"\bProduct\b", s, re.I):
                v.append("Ranking products requires Product column in SELECT and GROUP BY.")

            if "GROUP BY" in su and not re.search(r"\bGROUP\s+BY\b.*\bProduct\b", s, re.I | re.S):
                v.append("GROUP BY must include Product when ranking products.")

        # Partition enforcement
        if "for each year and country" in question.lower():
            if not re.search(r"PARTITION\s+BY\s+.*year.*country", s, re.I | re.S):
                v.append("PARTITION BY must include both year and Country.")

        # Undefined N check
        if re.search(r"\brnk\s*<=\s*N\b", s, re.I):
            v.append("Replace N with a concrete numeric value (e.g., 3, 5).")

        # Ranking column in final output
        if _has_ranking_window(s):
            if not re.search(r"\brnk\b|\brank\b|\brn\b", s, re.I):
                v.append("Ranking column must be included in final SELECT.")
    # --------------------------------------------------------
    # 5) Growth rules
    # --------------------------------------------------------
    if _is_growth_question(question):
        if not re.search(r"\bLAG\s*\(", s, re.I):
            v.append("Growth questions require LAG() for previous-period comparison (after aggregation).")
        if _has_division(s) and not _has_nullif(s):
            v.append("Growth calculations must use NULLIF to prevent division-by-zero.")
        # If growth question + ranking question, strongly encourage 2-step CTE
        if _is_ranking_question(question) and not _has_cte(s):
            v.append("Growth + ranking requires CTE layering: aggregate -> lag -> compute growth -> rank -> outer filter.")

    # --------------------------------------------------------
    # 6) Streak rules
    # --------------------------------------------------------
    if _is_streak_question(question) and not _mentions_gap_island(s):
        v.append("Streak/consecutive questions require gap-and-island logic (ROW_NUMBER difference) to prove consecutiveness.")

    # --------------------------------------------------------
    # 7) Amount rules
    # --------------------------------------------------------
    if _needs_amount_cast(schema_text) and re.search(r"\bAmount\b", s):
        if "REPLACE(REPLACE(Amount" not in s:
            v.append("Amount appears to be string currency; must CAST(REPLACE(REPLACE(Amount,'$',''),',','') AS DECIMAL(10,2)).")

    # --------------------------------------------------------
    # 8) Date rules (heuristic)
    # --------------------------------------------------------
    date_col = _looks_like_date_column(schema_text)
    if date_col and (_wants_monthly(question) or _wants_yearly(question) or _is_growth_question(question) or _is_streak_question(question)):
        if not _mentions_str_to_date(s) and date_col.lower() == "date":
            v.append("If Date is stored as text, parse using STR_TO_DATE(Date, '%m/%d/%Y') before YEAR()/DATE_FORMAT().")
        if _wants_monthly(question) and not _mentions_date_format(s):
            v.append("Monthly analysis should use DATE_FORMAT(parsed_date, '%Y-%m') for consistent month buckets.")

    # --------------------------------------------------------
    # 9) GROUP BY sanity
    # --------------------------------------------------------
    if _group_by_mismatch(s):
        v.append("Possible GROUP BY mismatch: non-aggregated SELECT columns should appear in GROUP BY (or move logic into CTE).")

    # --------------------------------------------------------
    # 10) Column sanity (best-effort)
    # --------------------------------------------------------
    if columns:
        known = set(c.lower() for c in columns)

        # strip strings to reduce false positives
        s_no_str = re.sub(r"'[^']*'", "''", s)
        tokens = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", s_no_str)

        ignore = {
            "select","from","where","group","by","order","limit","with","as","and","or","on","join",
            "inner","left","right","full","cross","distinct","union","all","having","case","when","then","else","end",
            "sum","avg","min","max","count","dense_rank","rank","row_number","lag","lead","over","partition",
            "year","month","date_format","str_to_date","cast","replace","decimal","nullif","desc","asc",
            # common aliases that should not be flagged
            "cte","t","x","y","a","b","c","d","e","rnk","rn","ranked","base","agg"
        }

        suspicious: List[str] = []
        for t in tokens:
            tl = t.lower()
            if tl in ignore:
                continue
            if tl == table_name.lower():
                continue
            # allow numeric-ish tokens already excluded by regex
            if known and (tl not in known):
                suspicious.append(t)

        if suspicious:
            v.append(f"Possible unknown columns used: {', '.join(sorted(set(suspicious))[:10])}. Use ONLY schema columns.")



# ============================================================
# Prompts (generation + rewrite)
# ============================================================

def _build_generation_prompt(schema_text: str, question: str, table_name: str) -> str:
    return f"""
You are a senior data engineer and MySQL 8+ SQL expert.

Schema:
{schema_text}

HARD RULES (NO EXCEPTIONS):
- Output ONLY SQL (no markdown, no comments, no explanations).
- MySQL 8+ ONLY.
- ONLY SELECT queries (no INSERT/UPDATE/DELETE/DDL).
- Use ONLY ONE table. No JOINs. No additional tables.
- Table name must be EXACTLY: {table_name}
- Never output multiple SQL statements (no extra semicolons).

WINDOW + AGG RULES:
- NEVER do window-over-aggregate in same SELECT (e.g., LAG(SUM(x)) is forbidden).
- Aggregate FIRST in a CTE, then apply window functions in NEXT CTE.
- MySQL cannot filter window outputs in WHERE/HAVING. Compute window in CTE, filter in OUTER SELECT.
- Any query using window functions MUST start with WITH and use layered CTEs.

RANKING RULES (STRICT):
- NEVER use HAVING to filter ranking results.
- Ranking filter MUST happen ONLY in OUTER SELECT using WHERE.
- If ranking is used:
    1) Aggregate in first CTE
    2) Apply LAG (if needed) in second CTE
    3) Compute growth + DENSE_RANK in third CTE
    4) Final SELECT must filter: WHERE rnk <= N
- Do NOT use LIMIT for per-group ranking.
- If question asks for single "highest/max/best" and ties may exist, prefer DENSE_RANK() over ROW_NUMBER().

GROWTH RULES:
- Growth rate: (current - previous) / previous
- Always use NULLIF(previous,0) to avoid division by zero.
- Use LAG() for previous period AFTER aggregation.
- Exclude previous is NULL rows when ranking growth.

DATE RULES:
- If Date might be text, parse using STR_TO_DATE(Date, '%m/%d/%Y').
- Monthly: DATE_FORMAT(parsed_date, '%Y-%m')
- Yearly: YEAR(parsed_date)

AMOUNT RULES:
- If Amount is currency text, cast using:
  CAST(REPLACE(REPLACE(Amount,'$',''),',','') AS DECIMAL(10,2))
- NEVER SUM raw string Amount.

STREAK RULES:
- If question asks consecutive/streak: must use gap-and-island (ROW_NUMBER difference) to prove consecutiveness.

User question:
{question}
""".strip()


def _build_rewrite_prompt(schema_text: str, question: str, table_name: str, sql: str, violations: List[str]) -> str:
    vtxt = "\n".join(f"- {x}" for x in violations)
    return f"""
The SQL below is INVALID or violates rules.

Violations:
{vtxt}

Rewrite the SQL correctly.

HARD RULES:
- Output ONLY SQL. No markdown. No comments.
- MySQL 8+ only.
- ONLY SELECT queries. No DDL/DML.
- Single-table only (no joins / no extra tables).
- Use ONLY schema columns.
- Table name EXACTLY: {table_name}
- No QUALIFY, no FILTER(), no DISTINCT ON.
- No window functions in WHERE/HAVING.
- No filtering on derived aliases in WHERE/HAVING.
- NEVER use HAVING for ranking queries.
- No window-over-aggregate in same SELECT.
- If using windows, MUST use layered CTEs.
- For per-group ranking:
    1) Compute ranking in a CTE
    2) Final SELECT must filter using: WHERE rnk <= N
    3) Do NOT use LIMIT for per-group ranking
    4) For "highest/max/best" questions, prefer DENSE_RANK() to avoid arbitrary tie-breaking.
- For growth divisions: use NULLIF(denominator,0).

Schema:
{schema_text}

User question:
{question}

Invalid SQL:
{sql}
""".strip()

# ============================================================
# Main LangGraph node
# ============================================================

def sql_node(state: Dict[str, Any]) -> Dict[str, Any]:
    schema_text = _normalize_schema(state.get("dataframe_schema"))
    question = state.get("user_input", "")

    if not isinstance(question, str) or not question.strip():
        raise ValueError("user_input missing or invalid")

    table_name = _extract_table_name(schema_text)
    columns = _extract_columns(schema_text)

    # 1) Generate
    prompt = _build_generation_prompt(schema_text, question, table_name)
    resp = llm.invoke(prompt)

    sql = _strip_markdown(getattr(resp, "content", "") or "")
    sql = _force_table_name(sql, table_name)

    # quick amount patch
    if _needs_amount_cast(schema_text):
        sql = _inject_amount_cast(sql)

    # 2) Validate + Repair loop
    max_attempts = 5
    for _ in range(max_attempts):
        violations = _validate_sql(sql, schema_text, table_name, question, columns)
        if not violations:
            break

        rewrite_prompt = _build_rewrite_prompt(schema_text, question, table_name, sql, violations)
        resp = llm.invoke(rewrite_prompt)

        sql = _strip_markdown(getattr(resp, "content", "") or "")
        sql = _force_table_name(sql, table_name)

        if _needs_amount_cast(schema_text):
            sql = _inject_amount_cast(sql)

    # Final (return best effort + violations list)
    return {
        **state,
        "output": sql,
        "table_name": table_name,
        "sql_violations": _validate_sql(sql, schema_text, table_name, question, columns)
    }