# file: modules/model_utils.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def _safe_label_encode(series):
    le = LabelEncoder()
    vals = series.fillna("NA").astype(str)
    return le.fit_transform(vals), le

def train_delay_model(df, sample_frac=1.0, random_state=42):
    """
    Trains a RandomForest classifier to predict delay_flag.
    Returns dict with model, features, metrics.
    """
    # pick features
    df = df.copy()
    features = []
    # numeric candidates
    for c in ["distance_km", "distance", "delivery_cost_inr", "fuel_consumption", "toll_charges", "traffic_delay_min"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
            features.append(c)

    # categorical encoded
    if "priority_simple" in df.columns:
        df["priority_enc"], le_priority = _safe_label_encode(df["priority_simple"])
        features.append("priority_enc")
    if "product_category" in df.columns:
        df["product_cat_enc"], _ = _safe_label_encode(df["product_category"])
        features.append("product_cat_enc")
    if "vehicle_type" in df.columns:
        df["vehicle_type_enc"], _ = _safe_label_encode(df["vehicle_type"])
        features.append("vehicle_type_enc")

    # fallback add some numeric columns up to 8 features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in numeric_cols:
        if c not in features and len(features) < 8:
            features.append(c)

    if "delay_flag" not in df.columns:
        raise ValueError("delay_flag not present in dataframe")

    modelling_df = df.dropna(subset=["delay_flag"]).copy()
    if sample_frac < 1.0:
        modelling_df = modelling_df.sample(frac=sample_frac, random_state=random_state)

    X = modelling_df[features].fillna(0)
    y = modelling_df["delay_flag"]

    if len(X) < 20:
        raise ValueError("Not enough rows to train reliably")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y if y.nunique()>1 else None)

    clf = RandomForestClassifier(n_estimators=150, random_state=random_state, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0))
    }
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Save model for reuse
    model_path = os.path.join(MODEL_DIR, "delay_rf.pkl")
    joblib.dump({"model": clf, "features": features}, model_path)

    return {"model": clf, "features": features, "metrics": metrics, "report": report, "model_path": model_path}

def load_saved_model(path=None):
    if path is None:
        path = os.path.join(MODEL_DIR, "delay_rf.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None

def predict_delay(model_obj, df):
    """
    Accepts model_obj (from train_delay_model or joblib.load)
    and dataframe df. Returns predicted labels array.
    """
    import numpy as np
    from sklearn.preprocessing import LabelEncoder

    if model_obj is None:
        return np.zeros(len(df), dtype=int)

    model = model_obj.get("model") if isinstance(model_obj, dict) else model_obj
    features = model_obj.get("features", [])

    df = df.copy()

    # --- Fix: recreate encoded cols if missing ---
    if "priority_simple" in df.columns and "priority_enc" not in df.columns:
        le = LabelEncoder()
        df["priority_enc"] = le.fit_transform(df["priority_simple"].astype(str))
    if "product_category" in df.columns and "product_cat_enc" not in df.columns:
        le = LabelEncoder()
        df["product_cat_enc"] = le.fit_transform(df["product_category"].astype(str))
    if "vehicle_type" in df.columns and "vehicle_type_enc" not in df.columns:
        le = LabelEncoder()
        df["vehicle_type_enc"] = le.fit_transform(df["vehicle_type"].astype(str))

    # Filter only features present in df
    available = [f for f in features if f in df.columns]
    if not available:
        raise KeyError("None of the trained features exist in the prediction dataframe")

    X = df[available].fillna(0)
    preds = model.predict(X)
    return preds

