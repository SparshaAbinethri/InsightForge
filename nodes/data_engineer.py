import pandas as pd
from typing import Dict, List, Optional, Tuple

from graph.state import AppState
from utils.dataframe_io import extract_schema


class DataEngineer:
    """
    Production-grade, client-safe Data Engineering pipeline.

    Core guarantees:
    - NO hard-coded dataset column names (semantic detection)
    - Numeric nulls -> median (reason logged)
    - Categorical nulls -> "Unknown"
    - Datetime nulls -> kept as NaT (never imputed)
    - Identifier nulls -> never imputed
    - Canonical flags ALWAYS named:
        - order_date_missing_flag
        - customer_id_missing_flag
      (even if dataset has different actual column names)
    - Explicit downstream contract logging
    - Optional quarantine dataset
    - UI-friendly summaries: dataset_summary, change_summary, column_report, null_summary
    - generated_code returns reproducible Python (with reasoning + contract)
    """

    # semantic keywords
    DATE_KEYWORDS = ["date", "time", "timestamp", "dt"]
    CUSTOMER_ID_KEYWORDS = ["customer_id", "cust_id", "client_id", "user_id", "customer", "client", "user"]
    IDENTIFIER_KEYWORDS = ["id", "code", "uuid", "guid"]

    # common money/qty keywords to help numeric detection (optional boost)
    NUMERIC_HINT_KEYWORDS = ["amount", "price", "revenue", "sales", "qty", "quantity", "discount", "profit", "cost", "rate"]

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        # downstream policies
        order_date_policy: str = "exclude_time_analytics",   # exclude_time_analytics | non_temporal_only | allow_all
        customer_id_policy: str = "flag_only",               # flag_only | quarantine | allow_all
        # thresholds
        parse_success_threshold: float = 0.70,
        heavy_missing_threshold: float = 0.60,
    ):
        self.raw_df = df.copy(deep=True)
        self.cleaned_df = df.copy(deep=True)

        self.audit_log: List[str] = []
        self.profile_report: Dict = {}
        self.quality_warnings: List[str] = []

        self.order_date_policy = order_date_policy
        self.customer_id_policy = customer_id_policy
        self.parse_success_threshold = parse_success_threshold
        self.heavy_missing_threshold = heavy_missing_threshold

        # semantic picks (actual column names in client dataset)
        self.order_date_col: Optional[str] = None
        self.customer_id_col: Optional[str] = None
        self.date_cols: List[str] = []

        # quarantine output
        self.quarantine_df: Optional[pd.DataFrame] = None

        # bookkeeping
        self._rows_before = self.raw_df.shape[0]
        self._cols_before = self.raw_df.shape[1]

    # --------------------------------------------------
    # 🔍 Semantic helpers
    # --------------------------------------------------
    def _standardize_name(self, s: str) -> str:
        return s.strip().lower().replace(" ", "_").replace("-", "_")

    def _find_columns_by_keywords(self, keywords: List[str]) -> List[str]:
        cols = []
        for col in self.cleaned_df.columns:
            cl = col.lower()
            if any(k in cl for k in keywords):
                cols.append(col)
        return cols

    def _pick_best_column(self, candidates: List[str]) -> Optional[str]:
        # pick shortest / most direct hit (simple heuristic)
        if not candidates:
            return None
        # Prefer exact-ish matches: contains both tokens like "order" + "date" etc.
        ranked = sorted(candidates, key=lambda c: (len(c), c))
        return ranked[0]

    def _is_identifier_col(self, col: str) -> bool:
        cl = col.lower()
        # treat explicit customer_id semantic col as identifier as well
        if self.customer_id_col and col == self.customer_id_col:
            return True
        return any(k in cl for k in self.IDENTIFIER_KEYWORDS)

    def _null_count(self, col: str) -> int:
        return int(self.cleaned_df[col].isna().sum())

    def _null_pct(self, col: str) -> float:
        return float(self.cleaned_df[col].isna().mean())

    # --------------------------------------------------
    # 1️⃣ Profiling (no mutation)
    # --------------------------------------------------
    def profile_data(self) -> Dict:
        df = self.raw_df
        report = {
            "rows": len(df),
            "columns": len(df.columns),
            "duplicate_rows": int(df.duplicated().sum()),
            "column_summary": {},
        }

        for col in df.columns:
            series = df[col]
            report["column_summary"][col] = {
                "dtype": str(series.dtype),
                "missing_count": int(series.isna().sum()),
                "missing_pct": round(series.isna().mean() * 100, 2),
                "unique_values": int(series.nunique(dropna=True)),
            }

        self.profile_report = report
        self.audit_log.append(f"📊 Profiling completed: {report['rows']} rows, {report['columns']} columns")
        if report["duplicate_rows"] > 0:
            self.audit_log.append(f"⚠ Detected {report['duplicate_rows']} duplicate rows")

        return report

    # --------------------------------------------------
    # 2️⃣ Standardize column names
    # --------------------------------------------------
    def standardize_column_names(self):
        self.cleaned_df.columns = [self._standardize_name(c) for c in self.cleaned_df.columns]
        self.audit_log.append("🔧 Standardized column names")

    # --------------------------------------------------
    # 3️⃣ Remove duplicates
    # --------------------------------------------------
    def remove_duplicates(self):
        before = len(self.cleaned_df)
        self.cleaned_df = self.cleaned_df.drop_duplicates()
        removed = before - len(self.cleaned_df)
        if removed > 0:
            self.audit_log.append(f"🧹 Removed {removed} duplicate rows")

    # --------------------------------------------------
    # 4️⃣ Detect semantic columns
    # --------------------------------------------------
    def detect_semantics(self):
        # date columns
        self.date_cols = self._find_columns_by_keywords(self.DATE_KEYWORDS)

        # pick a primary order_date col for time analytics (best guess)
        # Prefer something that contains "order" if exists
        orderish = [c for c in self.date_cols if "order" in c.lower()]
        self.order_date_col = self._pick_best_column(orderish) or self._pick_best_column(self.date_cols)

        # customer id column
        cust_candidates = self._find_columns_by_keywords(self.CUSTOMER_ID_KEYWORDS)

        # prefer ones that also contain "id"
        cust_idish = [c for c in cust_candidates if "id" in c.lower()]
        self.customer_id_col = self._pick_best_column(cust_idish) or self._pick_best_column(cust_candidates)

        # log semantic detection (UI clarity)
        self.audit_log.append(
            f"🧠 Semantic detection: date_cols={self.date_cols if self.date_cols else '[]'}, "
            f"primary_date={self.order_date_col if self.order_date_col else 'None'}, "
            f"customer_id={self.customer_id_col if self.customer_id_col else 'None'}"
        )

    # --------------------------------------------------
    # 5️⃣ Numeric conversion
    # --------------------------------------------------
    def clean_numeric_columns(self):
        for col in self.cleaned_df.columns:
            if self.cleaned_df[col].dtype != object:
                continue
            if self._is_identifier_col(col):
                continue
            # skip date cols: we parse separately
            if col in self.date_cols:
                continue

            cleaned = (
                self.cleaned_df[col].astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.replace("₹", "", regex=False)
                .str.replace("€", "", regex=False)
                .str.strip()
            )

            numeric = pd.to_numeric(cleaned, errors="coerce")
            success_ratio = numeric.notna().mean()

            # small heuristic boost: if name hints numeric, accept slightly lower threshold
            threshold = self.parse_success_threshold
            if any(k in col.lower() for k in self.NUMERIC_HINT_KEYWORDS):
                threshold = min(threshold, 0.60)

            if success_ratio >= threshold:
                self.cleaned_df[col] = numeric
                self.audit_log.append(f"🔢 Converted '{col}' to numeric (success={round(success_ratio*100,2)}%)")

    # --------------------------------------------------
    # 6️⃣ Date parsing (all detected date cols)
    # --------------------------------------------------
    def parse_date_columns(self):
        if not self.date_cols:
            return

        for col in self.date_cols:
            parsed = pd.to_datetime(self.cleaned_df[col], errors="coerce")
            self.cleaned_df[col] = parsed

        # log primary date specifically (used for contract)
        if self.order_date_col:
            self.audit_log.append(f"📅 Parsed '{self.order_date_col}' as datetime (invalid → NaT)")
        else:
            self.audit_log.append("📅 Parsed date/time columns as datetime (invalid → NaT)")

    # --------------------------------------------------
    # 7️⃣ Handle missing values (must be BEFORE flags)
    # --------------------------------------------------
    def handle_missing_values(self):
        for col in self.cleaned_df.columns:
            missing_pct = self._null_pct(col)
            if missing_pct == 0:
                continue

            if missing_pct > self.heavy_missing_threshold:
                self.quality_warnings.append(
                    f"⚠ Column '{col}' heavily missing ({round(missing_pct*100,2)}%)"
                )

            # numeric -> median
            if pd.api.types.is_numeric_dtype(self.cleaned_df[col]):
                cnt = self._null_count(col)
                self.cleaned_df[col] = self.cleaned_df[col].fillna(self.cleaned_df[col].median())
                self.audit_log.append(
                    f"🩹 Filled missing values in '{col}' with median (filled={cnt}; reason=robust to outliers/skew)"
                )
                continue

            # datetime -> keep NaT
            if pd.api.types.is_datetime64_any_dtype(self.cleaned_df[col]):
                self.quality_warnings.append(f"⚠ '{col}' contains missing dates (left as NaT)")
                continue

            # identifiers -> never impute
            if self._is_identifier_col(col):
                self.quality_warnings.append(f"⚠ '{col}' has missing identifiers (not imputed)")
                continue

            # categorical -> Unknown
            cnt = self._null_count(col)
            self.cleaned_df[col] = self.cleaned_df[col].fillna("Unknown")
            self.audit_log.append(f"🩹 Filled missing values in '{col}' with 'Unknown' (filled={cnt})")

    # --------------------------------------------------
    # 8️⃣ Add canonical flags + explicit downstream contract
    # --------------------------------------------------
    def add_quality_flags_and_contract(self):
        # canonical flags always named consistently (UI + tests stable)
        # 1) order_date_missing_flag
        if self.order_date_col and self.order_date_col in self.cleaned_df.columns:
            self.cleaned_df["order_date_missing_flag"] = self.cleaned_df[self.order_date_col].isna()
            missing_cnt = int(self.cleaned_df["order_date_missing_flag"].sum())

            if missing_cnt > 0:
                if self.order_date_policy == "exclude_time_analytics":
                    self.audit_log.append(
                        f"🧭 Downstream rule: {missing_cnt} rows with missing '{self.order_date_col}' "
                        f"are excluded from time-based analytics (kept for non-temporal metrics)"
                    )
                elif self.order_date_policy == "non_temporal_only":
                    self.audit_log.append(
                        f"🧭 Downstream rule: {missing_cnt} rows with missing '{self.order_date_col}' "
                        f"are retained ONLY for non-temporal aggregations"
                    )
                else:
                    self.audit_log.append(
                        f"🧭 Downstream rule: {missing_cnt} rows with missing '{self.order_date_col}' "
                        f"(policy='{self.order_date_policy}')"
                    )
        else:
            # keep schema stable (optional): create flag column as all False if no date col exists
            # Uncomment if you want flags always present:
            # self.cleaned_df["order_date_missing_flag"] = False
            pass

        # 2) customer_id_missing_flag
        if self.customer_id_col and self.customer_id_col in self.cleaned_df.columns:
            self.cleaned_df["customer_id_missing_flag"] = self.cleaned_df[self.customer_id_col].isna()
            missing_cnt = int(self.cleaned_df["customer_id_missing_flag"].sum())

            if missing_cnt > 0:
                if self.customer_id_policy in ("flag_only", "quarantine"):
                    self.audit_log.append(
                        f"🧭 Downstream rule: {missing_cnt} rows with missing '{self.customer_id_col}' "
                        f"are excluded from customer-level analytics"
                    )
                else:
                    self.audit_log.append(
                        f"🧭 Downstream rule: {missing_cnt} rows with missing '{self.customer_id_col}' "
                        f"(policy='{self.customer_id_policy}')"
                    )
        else:
            # Uncomment if you want flags always present:
            # self.cleaned_df["customer_id_missing_flag"] = False
            pass

    # --------------------------------------------------
    # 9️⃣ Optional quarantine (based on canonical flag)
    # --------------------------------------------------
    def apply_quarantine(self):
        if self.customer_id_policy != "quarantine":
            return
        if "customer_id_missing_flag" not in self.cleaned_df.columns:
            return

        mask = self.cleaned_df["customer_id_missing_flag"] == True
        quarantined = self.cleaned_df[mask].copy()
        kept = self.cleaned_df[~mask].copy()

        if len(quarantined) > 0:
            self.quarantine_df = quarantined
            self.cleaned_df = kept
            self.audit_log.append(f"🚧 Quarantined {len(quarantined)} rows due to missing customer identifier")

    # --------------------------------------------------
    # UI helpers
    # --------------------------------------------------
    def build_null_summary(self) -> Dict:
        total_nulls = int(self.cleaned_df.isna().sum().sum())
        cols_with_nulls = {
            col: {
                "null_count": int(self.cleaned_df[col].isna().sum()),
                "null_pct": round(float(self.cleaned_df[col].isna().mean() * 100), 2),
            }
            for col in self.cleaned_df.columns
            if int(self.cleaned_df[col].isna().sum()) > 0
        }
        return {"total_nulls": total_nulls, "columns_with_nulls": cols_with_nulls}

    def build_column_report(self) -> List[Dict]:
        return [
            {
                "column": col,
                "dtype": str(self.cleaned_df[col].dtype),
                "null_count": int(self.cleaned_df[col].isna().sum()),
                "null_pct": round(float(self.cleaned_df[col].isna().mean() * 100), 2),
                "unique_values": int(self.cleaned_df[col].nunique(dropna=True)),
            }
            for col in self.cleaned_df.columns
        ]

    # --------------------------------------------------
    # 🧪 Reproducible code (real Python + reasons + semantic placeholders)
    # --------------------------------------------------
    def generate_cleaning_code(self) -> str:
        date_col = self.order_date_col or "<PRIMARY_DATE_COLUMN>"
        cust_col = self.customer_id_col or "<CUSTOMER_ID_COLUMN>"
        date_cols = self.date_cols if self.date_cols else ["<DATE_COLUMN_1>", "<DATE_COLUMN_2>"]

        # Note: This is "template-style reproducible code"
        # It mirrors the same rules, without hard-coding client schema.
        return f"""
import pandas as pd

df = pd.read_csv("raw_data.csv")

# 1) Standardize column names
df.columns = (
    df.columns.str.strip().str.lower()
      .str.replace(" ", "_")
      .str.replace("-", "_")
)

# 2) Drop duplicates
df = df.drop_duplicates()

def is_identifier(col: str) -> bool:
    col = col.lower()
    return any(k in col for k in ["id", "code", "uuid", "guid"])

# 3) Convert object -> numeric where parse success is high
for col in df.columns:
    if df[col].dtype == object and not is_identifier(col):
        cleaned = (
            df[col].astype(str)
              .str.replace(",", "")
              .str.replace("$", "")
              .str.replace("₹", "")
              .str.replace("€", "")
              .str.strip()
        )
        numeric = pd.to_numeric(cleaned, errors="coerce")
        if numeric.notna().mean() >= {self.parse_success_threshold}:
            df[col] = numeric

# 4) Parse date/time columns (semantic; update list based on your dataset)
date_cols = {date_cols}
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")  # invalid -> NaT

# 5) Handle missing values
for col in df.columns:
    if df[col].isna().sum() == 0:
        continue

    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())  # reason: robust to outliers/skew
    elif pd.api.types.is_datetime64_any_dtype(df[col]):
        pass  # keep NaT (downstream must handle)
    elif is_identifier(col):
        pass  # never impute identifiers
    else:
        df[col] = df[col].fillna("Unknown")

# 6) Canonical flags (used by downstream analytics)
if "{date_col}" in df.columns:
    df["order_date_missing_flag"] = df["{date_col}"].isna()

if "{cust_col}" in df.columns:
    df["customer_id_missing_flag"] = df["{cust_col}"].isna()

# Downstream contract:
# - order_date_missing_flag == True -> exclude from time-based analytics ({self.order_date_policy})
# - customer_id_missing_flag == True -> exclude from customer-level analytics ({self.customer_id_policy})
"""

    # --------------------------------------------------
    # 🚀 Run pipeline
    # --------------------------------------------------
    def run(self) -> Dict:
        self.profile_data()
        self.standardize_column_names()
        self.remove_duplicates()

        # detect semantics early so numeric conversion can avoid date/id columns
        self.detect_semantics()

        self.clean_numeric_columns()
        self.parse_date_columns()

        # IMPORTANT: fill first, THEN flags (so flags reflect final state)
        self.handle_missing_values()
        self.add_quality_flags_and_contract()
        self.apply_quarantine()

        # final validation log
        self.audit_log.append(
            f"✅ Final dataset shape: {self.cleaned_df.shape[0]} rows, {self.cleaned_df.shape[1]} columns"
        )
        self.audit_log.extend(self.quality_warnings)

        return {
            "raw_df": self.raw_df,
            "cleaned_df": self.cleaned_df,
            "quarantine_df": self.quarantine_df,

            # semantic transparency for UI/debug
            "semantic_summary": {
                "primary_date_column": self.order_date_col,
                "date_columns_detected": self.date_cols,
                "customer_id_column": self.customer_id_col,
            },

            # UI sections
            "dataset_summary": {
                "rows": self.cleaned_df.shape[0],
                "columns": self.cleaned_df.shape[1],
                "column_names": list(self.cleaned_df.columns),
            },
            "change_summary": {
                "rows_before": self._rows_before,
                "rows_after": self.cleaned_df.shape[0],
                "rows_removed": self._rows_before - self.cleaned_df.shape[0],
                "columns_before": self._cols_before,
                "columns_after": self.cleaned_df.shape[1],
            },
            "column_report": self.build_column_report(),
            "null_summary": self.build_null_summary(),

            # existing
            "dataframe_schema": extract_schema(self.cleaned_df),
            "profile_report": self.profile_report,
            "audit_log": self.audit_log,
            "generated_code": self.generate_cleaning_code(),
        }


# ==================================================
# ✅ LangGraph node wrapper
# ==================================================
def data_engineer_node(state: AppState):
    raw_df = state.get("raw_df")
    if raw_df is None:
        return {"error": "❌ No raw data available"}

    engineer = DataEngineer(
        raw_df,
        order_date_policy="exclude_time_analytics",
        customer_id_policy="flag_only",   # change to "quarantine" if you want
    )
    return engineer.run()
