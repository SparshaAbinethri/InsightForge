# ==========================================================
# 🔧 PYTHON PATH FIX (DO NOT REMOVE)
# This ensures pytest can find the `nodes/` package
# ==========================================================
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

# ==========================================================
# Imports
# ==========================================================
import pandas as pd
import numpy as np
import pytest

from nodes.data_engineer import DataEngineer


# ==========================================================
# Fixtures
# ==========================================================
@pytest.fixture
def sample_dirty_df():
    """
    Small dirty dataframe designed to trigger:
    - column standardization
    - numeric conversion
    - date parsing with NaT
    - categorical unknown fill
    - ID not imputed
    - flags created
    """
    return pd.DataFrame({
        "Order ID": [1, 2, 3, 3],
        "Order Date": ["2024-01-01", None, "not_a_date", "2024-01-01"],
        "Sales Person": ["Alice", None, "Bob", "Alice"],
        "Region": ["East", "West", None, "East"],
        "Category": ["Office", None, "Tech", "Office"],
        "Amount": ["$1,200.50", None, "300", "$1,200.50"],
        "Quantity": ["2", None, "5", "2"],
        "Discount": ["0.1", None, "0.05", "0.1"],
        "Customer ID": ["C001", None, "C003", "C001"],
        "Channel": ["Online", "Retail", None, "Online"],
    })


# ==========================================================
# Tests
# ==========================================================
def test_standardizes_column_names(sample_dirty_df):
    eng = DataEngineer(sample_dirty_df)
    result = eng.run()

    cols = result["cleaned_df"].columns.tolist()
    assert "order_id" in cols
    assert "order_date" in cols
    assert "sales_person" in cols
    assert "customer_id" in cols


def test_numeric_conversion(sample_dirty_df):
    eng = DataEngineer(sample_dirty_df)
    df = eng.run()["cleaned_df"]

    assert pd.api.types.is_numeric_dtype(df["amount"])
    assert pd.api.types.is_numeric_dtype(df["quantity"])
    assert pd.api.types.is_numeric_dtype(df["discount"])


def test_date_parsing_and_nat(sample_dirty_df):
    eng = DataEngineer(sample_dirty_df)
    df = eng.run()["cleaned_df"]

    assert pd.api.types.is_datetime64_any_dtype(df["order_date"])
    assert df["order_date"].isna().sum() >= 1


def test_categorical_unknown_fill(sample_dirty_df):
    eng = DataEngineer(sample_dirty_df)
    df = eng.run()["cleaned_df"]

    assert "Unknown" in df["sales_person"].astype(str).values
    assert "Unknown" in df["region"].astype(str).values
    assert "Unknown" in df["category"].astype(str).values
    assert "Unknown" in df["channel"].astype(str).values


def test_identifier_not_imputed(sample_dirty_df):
    eng = DataEngineer(sample_dirty_df)
    df = eng.run()["cleaned_df"]

    assert df["customer_id"].isna().sum() >= 1
    assert "Unknown" not in df["customer_id"].astype(str).values


def test_flags_created(sample_dirty_df):
    eng = DataEngineer(sample_dirty_df)
    df = eng.run()["cleaned_df"]

    assert "order_date_missing_flag" in df.columns
    assert "customer_id_missing_flag" in df.columns
    assert df["order_date_missing_flag"].dtype == bool
    assert df["customer_id_missing_flag"].dtype == bool


def test_downstream_contract_logged(sample_dirty_df):
    eng = DataEngineer(
        sample_dirty_df,
        order_date_policy="exclude_time_analytics",
        customer_id_policy="flag_only",
    )
    log_text = "\n".join(eng.run()["audit_log"])

    assert "Downstream rule" in log_text
    assert "order_date" in log_text
    assert "customer_id" in log_text


def test_null_summary_present(sample_dirty_df):
    eng = DataEngineer(sample_dirty_df)
    null_summary = eng.run()["null_summary"]

    assert "total_nulls" in null_summary
    assert "columns_with_nulls" in null_summary


def test_quarantine_mode_moves_rows(sample_dirty_df):
    eng = DataEngineer(
        sample_dirty_df,
        customer_id_policy="quarantine",
    )
    result = eng.run()

    cleaned_df = result["cleaned_df"]
    quarantine_df = result["quarantine_df"]

    assert quarantine_df is not None
    assert len(quarantine_df) >= 1
    assert cleaned_df["customer_id"].isna().sum() == 0


def test_generated_code_contains_contract(sample_dirty_df):
    eng = DataEngineer(sample_dirty_df)
    code = eng.run()["generated_code"]

    assert "order_date_missing_flag" in code
    assert "customer_id_missing_flag" in code
    assert "downstream" in code.lower()


def test_deterministic_output(sample_dirty_df):
    df1 = DataEngineer(sample_dirty_df).run()["cleaned_df"]
    df2 = DataEngineer(sample_dirty_df).run()["cleaned_df"]

    pd.testing.assert_frame_equal(
        df1.reset_index(drop=True),
        df2.reset_index(drop=True)
    )
