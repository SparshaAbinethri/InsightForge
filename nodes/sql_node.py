import env_setup  # must be first
from langchain_openai import ChatOpenAI
import re

# ============================================================
# LLM
# ============================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# ============================================================
# Helpers
# ============================================================

def _extract_table_name(schema: str) -> str:
    """
    Extract table name from schema.
    Expected format:
    Table name: data_table
    """
    match = re.search(r"Table name:\s*([A-Za-z_][A-Za-z0-9_]*)", schema)
    return match.group(1) if match else "data_table"


def _strip_markdown(sql: str) -> str:
    """
    Remove ```sql``` or ``` fences if model adds them.
    """
    s = (sql or "").strip()
    fence = re.match(r"^```(?:sql)?\s*(.*?)\s*```$", s, re.I | re.S)
    if fence:
        return fence.group(1).strip()

    s = re.sub(r"```(?:sql)?", "", s, flags=re.I)
    s = s.replace("```", "")
    return s.strip()


def _force_table_name(sql: str, table_name: str) -> str:
    """
    Force FIRST FROM clause to use correct table name.
    """
    return re.sub(
        r"\bFROM\b\s+[`\"]?[A-Za-z_][A-Za-z0-9_]*[`\"]?",
        f"FROM {table_name}",
        sql,
        count=1,
        flags=re.I
    )


def _needs_amount_cast(schema: str) -> bool:
    """
    If schema contains Amount column, we enforce numeric casting.
    Chocolate Sales dataset typically stores Amount like "$1,234.56".
    """
    return bool(re.search(r"\bAmount\b", schema, re.I))


def _is_growth_question(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in [
        "growth", "growth rate", "percentage growth", "percent increase",
        "mom", "month over month", "month-to-month",
        "yoy", "year over year", "year-over-year", "previous year", "last year"
    ])


def _is_streak_question(question: str) -> bool:
    """
    Detect questions requiring consecutive streak logic.
    Examples: "3 consecutive months", "in a row", "streak", "back to back".
    """
    q = question.lower()
    triggers = [
        "consecutive",
        "in a row",
        "streak",
        "continuous",
        "back to back",
        "back-to-back",
        "months of decline",
        "months of decrease",
        "months of increase",
        "consecutive months"
    ]
    return any(t in q for t in triggers)


def _needs_per_group_ranking(question: str) -> bool:
    """
    Stronger heuristic for per-group ranking questions.
    """
    q = question.lower()
    triggers = [
        "for each", "per ", "within each", "top", "highest", "lowest",
        "best", "rank", "by country", "by category", "by year", "by month"
    ]
    return any(t in q for t in triggers)


def _has_limit(sql: str) -> bool:
    return bool(re.search(r"\bLIMIT\b", sql, re.I))


def _has_ranking_window(sql: str) -> bool:
    return bool(re.search(r"\b(DENSE_RANK|ROW_NUMBER|RANK)\s*\(", sql, re.I))


def _has_window_alias_in_where(sql: str) -> bool:
    """
    Detect MySQL-illegal filtering on window function results in WHERE.
    (MySQL doesn't allow filtering on window results in WHERE/HAVING directly.)
    """
    return bool(re.search(
        r"WHERE\s+.*\b("
        r"rank|dense_rank|row_number|previous_|lag|lead"
        r")\b",
        sql,
        re.I | re.S
    ))


def _inject_amount_cast(sql: str) -> str:
    """
    Ensure SUM(Amount) is converted to numeric-safe SUM(CAST(REPLACE...)).
    Only patch common pattern SUM(Amount).
    """
    return re.sub(
        r"\bSUM\s*\(\s*Amount\s*\)",
        "SUM(CAST(REPLACE(REPLACE(Amount, '$', ''), ',', '') AS DECIMAL(10,2)))",
        sql,
        flags=re.I
    )


def _has_streak_grouping(sql: str) -> bool:
    """
    Detect gap-and-island streak logic:
      ROW_NUMBER() ... - ROW_NUMBER() ...
    This is the standard pattern to group consecutive records.
    """
    return bool(re.search(
        r"ROW_NUMBER\s*\(.*?\)\s*-\s*ROW_NUMBER\s*\(",
        sql,
        re.I | re.S
    ))

# ============================================================
# Rewrite Prompts (GUARDRAILS)
# ============================================================

def _build_limit_rewrite_prompt(schema, question, bad_sql, table_name):
    return f"""
The following SQL is INVALID because it uses LIMIT for per-group ranking.

RULES:
- LIMIT is FORBIDDEN
- Use DENSE_RANK() or ROW_NUMBER()
- Use PARTITION BY the grouping column
- Filter using rank <= N
- Use MySQL 8+ syntax ONLY
- Use table name EXACTLY as: {table_name}
- Do NOT include comments, markdown, or explanations
- Output ONLY SQL

Schema:
{schema}

User question:
{question}

INVALID SQL:
{bad_sql}

Rewrite the SQL correctly now.
""".strip()


def _build_ranking_rewrite_prompt(schema, question, bad_sql, table_name):
    return f"""
The following SQL is INCOMPLETE.

The user question requires PER-GROUP TOP-N ranking,
but the query does NOT use a ranking window function.

MANDATORY RULES:
- Use DENSE_RANK() or ROW_NUMBER()
- Use PARTITION BY the grouping column
- Compute ranking in a CTE or subquery
- Filter results using WHERE rank <= N
- Use MySQL 8+ syntax ONLY
- Do NOT use LIMIT
- Do NOT use QUALIFY
- Use table name EXACTLY as: {table_name}
- Output ONLY SQL

Schema:
{schema}

User question:
{question}

INVALID SQL:
{bad_sql}

Rewrite the SQL correctly now.
""".strip()


def _build_window_where_rewrite_prompt(schema, question, bad_sql, table_name):
    return f"""
The following SQL is INVALID in MySQL.

Reason:
- MySQL does NOT allow filtering on window function results in WHERE.

MANDATORY FIX:
- Compute window functions in one CTE
- Apply filtering in an OUTER SELECT or NEXT CTE
- Use MySQL 8+ syntax ONLY
- Use table name EXACTLY as: {table_name}
- Do NOT include comments, markdown, or explanations
- Output ONLY SQL

Schema:
{schema}

User question:
{question}

INVALID SQL:
{bad_sql}

Rewrite the SQL correctly now.
""".strip()


def _build_streak_rewrite_prompt(schema, question, bad_sql, table_name):
    return f"""
The following SQL is LOGICALLY INCORRECT.

The user question requires detection of CONSECUTIVE time-based streaks
(e.g. consecutive months of decline/increase).

ERRORS IN CURRENT SQL:
- GROUP BY collapses streaks
- COUNT(*) alone does NOT guarantee consecutiveness
- Missing gap-and-island (streak grouping) logic

MANDATORY FIX:
- Aggregate revenue by month first
- Use LAG() to detect increase/decrease
- Create a binary flag (is_decline / is_increase)
- Use ROW_NUMBER difference technique to group consecutive rows into streak_id:
  ROW_NUMBER() OVER (PARTITION BY entity ORDER BY month)
  -
  ROW_NUMBER() OVER (PARTITION BY entity, flag ORDER BY month)
- Group by (entity, streak_id)
- Use HAVING COUNT(*) >= N for streak length
- Use MySQL 8+ syntax ONLY
- Use table name EXACTLY as: {table_name}
- Do NOT include comments, markdown, or explanations
- Output ONLY SQL

Schema:
{schema}

User question:
{question}

INVALID SQL:
{bad_sql}

Rewrite the SQL correctly now.
""".strip()

# ============================================================
# Generation Prompt (ALL RULES, MYSQL ONLY)
# ============================================================

def _build_generation_prompt(schema: str, question: str, table_name: str) -> str:
    return f"""
You are a senior data engineer and SQL expert.

Given the following table schema:
{schema}

Your task:
Generate a SQL query that correctly answers the user's question.

================================================================
STRICT SQL RULES — MUST FOLLOW ALL (NO EXCEPTIONS)
================================================================

A. BASIC SAFETY
1. Use ONLY column names present in the schema.
2. Use the table name EXACTLY as specified in the schema ({table_name}).
3. Do NOT assume additional tables, joins, or columns.
4. Do NOT hallucinate primary keys, foreign keys, indexes, or relationships.
5. Do NOT infer business meaning beyond what is stated in the question.

B. SEMANTIC INTERPRETATION
6. If the question contains "growth", "growth rate", "percentage growth",
   "percent increase", or "rate of growth":
   - Compute percentage growth:
     (current_value - previous_value) / previous_value
7. If the question contains "change", "difference", "increase", "decrease":
   - Compute absolute difference:
     current_value - previous_value
8. If the question contains "trend", "over time", or "progression":
   - Use time ordering and window functions.
9. Treat as YoY:
   - yoy, year over year, year-over-year, year to year,
     annual growth, compared to last year, previous year
   - Use LAG()
10. Treat as MoM:
    - mom, month over month, month-to-month,
      compared to last month

C. AGGREGATION
11. Use GROUP BY for all non-aggregated columns.
12. Never mix aggregated and non-aggregated columns.
13. Aggregate metrics BEFORE analytics or ranking.

D. WINDOW FUNCTIONS
14. For ranking, growth, or comparisons:
    - Aggregate in a CTE first
    - Apply window functions next
15. NEVER apply window functions to aggregates directly.
16. Allowed window functions:
    LAG, LEAD, ROW_NUMBER, RANK, DENSE_RANK,
    SUM() OVER, AVG() OVER, MIN() OVER, MAX() OVER
17. Always define ORDER BY inside window functions.
18. Use PARTITION BY when results are required per group.

E. TOP-N / RANKING
19. For "top N per year/category/group/within each":
    - Use DENSE_RANK() or ROW_NUMBER()
    - PARTITION BY grouping column
    - Filter rank <= N
20. If "top" is used without specifying N:
    - Assume N = 3
21. NEVER use LIMIT for per-group ranking.
    LIMIT is allowed ONLY for global results.

F. DATE & TIME (MYSQL)
22. If Date column is not explicitly typed:
    - Parse using STR_TO_DATE(Date, '%m/%d/%Y')
23. For monthly analysis:
    - Use DATE_FORMAT(parsed_date, '%Y-%m')
24. Always ORDER BY time for sequential comparisons.
25. Use consistent granularity.

G. AMOUNT HANDLING
26. If Amount is not numeric:
    - Convert using:
      CAST(REPLACE(REPLACE(Amount, '$', ''), ',', '') AS DECIMAL(10,2))
27. NEVER aggregate raw string Amount values.

H. STREAK / CONSECUTIVE REQUIREMENTS
28. If the question asks for "consecutive", "streak", or "in a row":
    - You MUST use gap-and-island technique with ROW_NUMBER difference
      to group consecutive months into streak_id.
    - COUNT(*) alone is NOT sufficient to prove consecutiveness.
    - Output streak boundaries using MIN(month) as start and MAX(month) as end.

I. NULLS & EDGE CASES
29. Always use NULLIF to prevent division-by-zero.
30. Exclude rows where previous-period values are NULL when ranking growth.
31. Do NOT fabricate missing data.

J. SQL QUALITY
32. Use clear, meaningful aliases.
33. Prefer CTEs for readability.
34. SQL must be compatible with MySQL 8+.
35. Do NOT include comments, markdown, or explanations.
36. Output ONLY the final SQL query.

MYSQL DIALECT ENFORCEMENT
37. Generate SQL ONLY for MySQL 8+.
38. The following are NOT supported in MySQL and MUST NOT be used:
    - QUALIFY
    - FILTER clause
    - DISTINCT ON
    - Window functions in WHERE or HAVING
39. MySQL does NOT allow filtering directly on window functions.
40. To filter ranked results:
    - Compute window functions in a CTE or subquery
    - Filter using an outer SELECT with WHERE
41. Any query using QUALIFY is INVALID.
42. Any query using LIMIT for per-group ranking is INVALID.

================================================================

User question:
{question}
""".strip()

# ============================================================
# LangGraph Node
# ============================================================

def sql_node(state):
    schema = state["dataframe_schema"]
    question = state["user_input"]
    table_name = _extract_table_name(schema)

    # Initial generation
    prompt = _build_generation_prompt(schema, question, table_name)
    resp = llm.invoke(prompt)

    sql = _strip_markdown(resp.content or "")
    sql = _force_table_name(sql, table_name)

    # Enforce Amount numeric casting if schema has Amount
    if _needs_amount_cast(schema):
        sql = _inject_amount_cast(sql)

    # LIMIT guard
    if _needs_per_group_ranking(question) and _has_limit(sql):
        rewrite_prompt = _build_limit_rewrite_prompt(schema, question, sql, table_name)
        resp = llm.invoke(rewrite_prompt)
        sql = _strip_markdown(resp.content or "")
        sql = _force_table_name(sql, table_name)
        if _needs_amount_cast(schema):
            sql = _inject_amount_cast(sql)

    # Ranking guard
    if _needs_per_group_ranking(question) and not _has_ranking_window(sql):
        rewrite_prompt = _build_ranking_rewrite_prompt(schema, question, sql, table_name)
        resp = llm.invoke(rewrite_prompt)
        sql = _strip_markdown(resp.content or "")
        sql = _force_table_name(sql, table_name)
        if _needs_amount_cast(schema):
            sql = _inject_amount_cast(sql)

    # Window-WHERE guard (MySQL enforcement)
    if _has_window_alias_in_where(sql):
        rewrite_prompt = _build_window_where_rewrite_prompt(schema, question, sql, table_name)
        resp = llm.invoke(rewrite_prompt)
        sql = _strip_markdown(resp.content or "")
        sql = _force_table_name(sql, table_name)
        if _needs_amount_cast(schema):
            sql = _inject_amount_cast(sql)

    # Streak / consecutive guard (forces gap-and-island logic)
    if _is_streak_question(question) and not _has_streak_grouping(sql):
        rewrite_prompt = _build_streak_rewrite_prompt(schema, question, sql, table_name)
        resp = llm.invoke(rewrite_prompt)
        sql = _strip_markdown(resp.content or "")
        sql = _force_table_name(sql, table_name)
        if _needs_amount_cast(schema):
            sql = _inject_amount_cast(sql)

    return {
        **state,
        "output": sql
    }
