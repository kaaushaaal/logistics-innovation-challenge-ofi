# file: modules/data_loader.py
import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

def _read_csv_safe(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, encoding="latin1")

def load_all():
    """
    Returns a dict of raw dataframes keyed by short names.
    Expects CSVs in ../data/ with names:
      orders.csv, delivery_performance.csv, routes_distance.csv,
      vehicle_fleet.csv, warehouse_inventory.csv, customer_feedback.csv, cost_breakdown.csv
    """
    files = {
        "orders": "orders.csv",
        "delivery": "delivery_performance.csv",
        "routes": "routes_distance.csv",
        "vehicles": "vehicle_fleet.csv",
        "warehouse": "warehouse_inventory.csv",
        "feedback": "customer_feedback.csv",
        "costs": "cost_breakdown.csv",
    }
    data = {}
    for k, fn in files.items():
        p = os.path.join(DATA_DIR, fn)
        if os.path.exists(p):
            data[k] = _read_csv_safe(p)
        else:
            data[k] = pd.DataFrame()
    return data

def sample_info(data):
    """Return a small summary (rows, cols, columns names) for quick display."""
    info = {}
    for k, df in data.items():
        info[k] = {"rows": int(df.shape[0]), "cols": int(df.shape[1]), "columns": list(df.columns)}
    return info
