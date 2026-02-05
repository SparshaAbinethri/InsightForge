import pandas as pd
import pytest
from nodes.data_analyst import data_analyst_node


def run_agent(df, question):
    state = {"cleaned_df": df, "user_input": question, "dataframe_schema": None}
    return data_analyst_node(state)


def assert_success(result):
    assert "analysis_metadata" in result
    assert result["analysis_metadata"]["status"] == "success"
    assert "analysis_summary" in result


def assert_refused(result):
    assert "analysis_metadata" in result
    assert result["analysis_metadata"]["status"] == "refused"
    assert "analysis_summary" in result


@pytest.fixture
def sales_df():
    return pd.DataFrame({
        "order_id": [1, 2, 3, 4, 5],
        "amount": [100.0, 200.0, 150.0, 50.0, 300.0],
        "sales_person": ["A", "B", "A", "C", "B"],
        "region": ["US", "EU", "US", "EU", "US"],
        "category": ["Office", "Tech", "Office", "Tech", "Office"],
    })


@pytest.fixture
def concentration_df():
    return pd.DataFrame({"role": ["Producer"] * 40 + ["Editor"] * 30 + ["Writer"] * 30})


@pytest.fixture
def missing_df():
    return pd.DataFrame({
        "amount": [100, None, None, 50],
        "region": ["US", "EU", "EU", "US"]
    })


@pytest.fixture
def id_only_df():
    return pd.DataFrame({"anime_id": [101, 102, 103], "user_id": [1, 2, 3]})


# -----------------------------
# Contract
# -----------------------------
def test_output_contract(sales_df):
    result = run_agent(sales_df, "Total revenue")
    assert "analysis_summary" in result
    assert "analysis_metadata" in result
    assert isinstance(result["analysis_metadata"], dict)
    assert "status" in result["analysis_metadata"]


# -----------------------------
# KPI basics
# -----------------------------
def test_total_revenue_success(sales_df):
    result = run_agent(sales_df, "What is the total revenue?")
    assert_success(result)
    s = result["analysis_summary"].lower()
    assert "revenue" in s
    assert "using 'amount'" in s


def test_average_value_success(sales_df):
    result = run_agent(sales_df, "What is the average order value?")
    assert_success(result)
    s = result["analysis_summary"].lower()
    assert "average_value" in s or "average" in s
    assert "using 'amount'" in s


def test_count_records(sales_df):
    result = run_agent(sales_df, "How many records are there?")
    assert_success(result)
    assert "computed value is 5" in result["analysis_summary"].lower()


# -----------------------------
# Group-by prompts (your manual ones)
# -----------------------------
def test_sales_amount_by_region(sales_df):
    result = run_agent(sales_df, "Show sales amount by region")
    assert_success(result)
    s = result["analysis_summary"].lower()
    assert "by ['region']" in s
    assert "amount" in s


def test_revenue_by_category(sales_df):
    result = run_agent(sales_df, "Revenue by category")
    assert_success(result)
    s = result["analysis_summary"].lower()
    assert "by ['category']" in s
    assert "amount" in s


def test_average_revenue_by_sales_person(sales_df):
    result = run_agent(sales_df, "Average revenue by sales_person")
    assert_success(result)
    s = result["analysis_summary"].lower()
    assert "by ['sales_person']" in s
    assert "mean" in s  # should not be sum


def test_top_contributor(sales_df):
    result = run_agent(sales_df, "Which sales person contributes the most revenue?")
    assert_success(result)
    s = result["analysis_summary"].lower()
    assert "by ['sales_person']" in s
    # should return only one item or at least include a top person
    assert "a=" in s or "b=" in s or "c=" in s


def test_top_2_by_sales_person(sales_df):
    result = run_agent(sales_df, "Top 2 sales people by revenue")
    assert_success(result)
    s = result["analysis_summary"].lower()
    assert "by ['sales_person']" in s
    # crude check that it doesn't dump everything
    assert s.count("=") <= 3


# -----------------------------
# Distinct / unique
# -----------------------------
def test_distinct_regions(sales_df):
    result = run_agent(sales_df, "Distinct regions")
    assert_success(result)
    s = result["analysis_summary"].lower()
    assert "unique" in s
    assert "region" in s


# -----------------------------
# Data quality prompts
# -----------------------------
def test_show_null_values(missing_df):
    result = run_agent(missing_df, "Show null values")
    assert_success(result)
    s = result["analysis_summary"].lower()
    assert "missing" in s
    assert "duplicate" in s


def test_data_quality_issues(missing_df):
    result = run_agent(missing_df, "Are there data quality issues I should worry about?")
    assert_success(result)
    s = result["analysis_summary"].lower()
    assert "missing" in s
    assert "duplicate" in s


def test_duplicate_detection():
    df = pd.DataFrame({"amount": [100, 100, 200], "region": ["US", "US", "EU"]})
    result = run_agent(df, "Are there duplicates?")
    assert_success(result)
    assert "duplicate rows" in result["analysis_summary"].lower()


# -----------------------------
# Insights / concentration / skew
# -----------------------------
def test_give_me_insights(concentration_df):
    result = run_agent(concentration_df, "Give me insights")
    assert_success(result)
    s = result["analysis_summary"].lower()
    assert "concentrated" in s
    assert "%" in s


def test_concentration_issues(concentration_df):
    result = run_agent(concentration_df, "Any concentration issues?")
    assert_success(result)
    assert "concentrated" in result["analysis_summary"].lower()


def test_skewed(concentration_df):
    result = run_agent(concentration_df, "Is the data skewed?")
    assert_success(result)
    assert "concentrated" in result["analysis_summary"].lower()


# -----------------------------
# Outliers / extreme values
# -----------------------------
def test_outliers(sales_df):
    result = run_agent(sales_df, "Any outliers?")
    assert_success(result)
    s = result["analysis_summary"].lower()
    assert "outlier" in s or "not enough numeric data" in s


def test_extreme_values(sales_df):
    result = run_agent(sales_df, "Are there extreme values?")
    assert_success(result)
    s = result["analysis_summary"].lower()
    assert "outlier" in s or "not enough numeric data" in s


# -----------------------------
# Trend refusal without date
# -----------------------------
def test_trend_refuses_without_date(sales_df):
    result = run_agent(sales_df, "Show monthly revenue trend")
    # our node returns success if it finds a datetime col; here there is none, so refuse
    assert_refused(result)
    assert "date/time" in result["analysis_summary"].lower()


# -----------------------------
# Refusal: ID-only metrics
# -----------------------------
def test_revenue_with_only_id_columns_refuses(id_only_df):
    result = run_agent(id_only_df, "What is the total revenue?")
    assert_refused(result)
    s = result["analysis_summary"].lower()
    assert "id columns" in s or "id" in s


# -----------------------------
# Summary
# -----------------------------
def test_dataset_summary(sales_df):
    result = run_agent(sales_df, "Can you summarize this dataset for me?")
    assert_success(result)
    s = result["analysis_summary"].lower()
    assert "dataset summary" in s
    assert "rows=" in s
    assert "columns=" in s
