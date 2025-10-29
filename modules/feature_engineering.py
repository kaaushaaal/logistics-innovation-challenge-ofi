# file: modules/feature_engineering.py
import pandas as pd
import numpy as np

def dedup_columns(df):
    """
    Remove duplicate column names (keep first occurrence).
    """
    return df.loc[:, ~df.columns.duplicated()]

def standardize_cols(df):
    """
    Normalize column names to lowercase and underscores for easier referencing.
    """
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def merge_all(data):
    """
    Merge orders, delivery, routes, costs, feedback into a single dataframe.
    Keeps orders as the base (left join).
    """
    orders = data.get("orders", pd.DataFrame()).copy()
    delivery = data.get("delivery", pd.DataFrame()).copy()
    routes = data.get("routes", pd.DataFrame()).copy()
    costs = data.get("costs", pd.DataFrame()).copy()
    feedback = data.get("feedback", pd.DataFrame()).copy()

    # Standardize column names
    orders = standardize_cols(orders)
    delivery = standardize_cols(delivery)
    routes = standardize_cols(routes)
    costs = standardize_cols(costs)
    feedback = standardize_cols(feedback)

    # Use a canonical order id column name
    for df in (orders, delivery, routes, costs, feedback):
        for cname in ("order_id", "orderid", "order id"):
            if cname in df.columns:
                df.rename(columns={cname: "order_id"}, inplace=True)
                break

    # Merge stepwise
    df = orders.merge(delivery, how="left", on="order_id", suffixes=("", "_delivery"))
    df = df.merge(routes, how="left", on="order_id", suffixes=("", "_route"))
    df = df.merge(costs, how="left", on="order_id", suffixes=("", "_cost"))
    df = df.merge(feedback, how="left", on="order_id", suffixes=("", "_fb"))

    # Remove duplicate-named columns that can break pyarrow (streamlit)
    df = dedup_columns(df)

    # Parse dates if present
    for col in df.columns:
        if "date" in col:
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                pass

    # Derived features
    # delay: if promised/actual days exist, compute difference in days
    if "promised_delivery_days" in df.columns and "actual_delivery_days" in df.columns:
        # These appear to be numeric days counts; convert to numeric then difference
        df["promised_delivery_days"] = pd.to_numeric(df["promised_delivery_days"], errors="coerce")
        df["actual_delivery_days"] = pd.to_numeric(df["actual_delivery_days"], errors="coerce")
        df["delay_days"] = df["actual_delivery_days"] - df["promised_delivery_days"]
        df["delay_flag"] = (df["delay_days"] > 0).astype(int)
    else:
        # Fallback: use delivery_status if available
        if "delivery_status" in df.columns:
            df["delay_flag"] = df["delivery_status"].apply(lambda x: 0 if str(x).lower() in ("delivered","on time","ontime") else 1)
        else:
            df["delay_flag"] = 0

    # Normalize priority
    if "priority" in df.columns:
        df["priority_simple"] = df["priority"].astype(str).str.lower().fillna("standard")
    else:
        df["priority_simple"] = "standard"

    # Numeric fill
    for c in df.select_dtypes(include="number").columns:
        df[c] = df[c].fillna(0)

    # Clean product category / origin / destination names
    for c in ("product_category","origin","destination"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    return df