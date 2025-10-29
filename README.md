# 🚚 Ultimate Logistics Suite

**An AI-powered logistics intelligence dashboard built with Streamlit** — designed to unify data analytics, predictive modeling, and route optimization for smarter supply chain management.

> 📦 Built by **Kaushal Shukla** — Final Year B.Tech CSE | Cloud, DevOps & Security Enthusiast

---

## 🧭 Table of Contents
- [Overview](#overview)
- [Project Architecture](#project-architecture)
- [Data Flow](#data-flow)
- [Feature Engineering](#feature-engineering)
- [Machine Learning Model](#machine-learning-model)
- [Prediction Logic](#prediction-logic)
- [Route Scoring](#route-scoring)
- [Vehicle Assignment](#vehicle-assignment)
- [Streamlit Interface](#streamlit-interface)
- [Error Handling & Edge Cases](#error-handling--edge-cases)
- [Validation & Monitoring](#validation--monitoring)
- [Future Improvements](#future-improvements)
- [Worked Example](#worked-example)
- [Run Locally](#run-locally)
- [Author](#author)

---

## 🧩 Overview

**Ultimate Logistics Suite** integrates multiple logistics operations under one roof:
- Predicts delivery delays using a **RandomForestClassifier**
- Optimizes route selection with cost–time–emission heuristics
- Manages vehicle assignments efficiently
- Visualizes KPIs and fleet data interactively

All powered by:
- 🧠 Machine Learning (`scikit-learn`)
- 📊 Visualization (`Plotly`, `Streamlit`)
- 🧰 Modular Python architecture

---

## 🧱 Project Architecture

ofi_case_study/
├── app.py                     # Streamlit main dashboard
├── data/                      # Raw CSV data files
│   ├── orders.csv
│   ├── delivery_performance.csv
│   ├── routes_distance.csv
│   ├── vehicle_fleet.csv
│   ├── cost_breakdown.csv
│   ├── warehouse_inventory.csv
│   └── customer_feedback.csv
├── modules/                   # Core logic modules
│   ├── data_loader.py         # Reads all CSVs and returns pandas DataFrames
│   ├── feature_engineering.py # Cleans, merges, and encodes data
│   ├── model_utils.py         # Trains and evaluates ML models
│   └── route_utils.py         # Route scoring and vehicle assignment logic
├── requirements.txt           # Python dependencies
├── README.md                  # Full documentation and usage guide
└── .gitignore                 # Ignored files (venv, cache, etc.)


### Why this design?
✅ Clean separation of logic  
✅ Easy debugging and testing  
✅ Reusable modules for other projects  
✅ Production-grade maintainability

---

## 🔄 Data Flow

1. **Load raw data** — via `data_loader.load_all()`.  
2. **Merge & clean** — unified DataFrame with delay and route data.  
3. **Engineer features** — delay flags, encoded priorities, normalized costs.  
4. **Model training** — builds a RandomForest delay predictor.  
5. **Prediction** — outputs delay risks per order.  
6. **Optimization** — scores routes & assigns optimal vehicles.  

---

## 🧮 Feature Engineering

| Derived Column | Description | Example |
|----------------|-------------|----------|
| `delay_days` | `actual_delivery_days - promised_delivery_days` | `5 - 3 = 2` |
| `delay_flag` | Binary delay indicator | `1 = delayed`, `0 = on time` |
| `priority_simple` | Cleaned lowercase priority | `"Express"` → `"express"` |
| `priority_enc` | Label-encoded version for ML | `"express" → 2` |
| `product_cat_enc` | Encoded product category | `"electronics" → 4` |

Duplicate columns after merging are automatically removed to prevent `pyarrow` errors.

---

## 🤖 Machine Learning Model

### Model Type
`RandomForestClassifier` — robust, interpretable, and fast for tabular logistics data.

### Training Logic
- Uses both numeric (`distance_km`, `delivery_cost_inr`, etc.) and encoded categorical features.
- Splits into 80% train / 20% test.
- Evaluates accuracy, F1-score, and classification report.

**Example:**
| Feature | Value |
|----------|--------|
| `distance_km` | 80 |
| `delivery_cost_inr` | 250 |
| `priority_enc` | 2 (`express`) |
| `product_cat_enc` | 4 (`electronics`) |

→ **Predicted delay_flag:** `1 (likely delayed)`

### Why RandomForest?
- Handles mixed datatypes  
- Resists overfitting  
- Works with small–medium datasets  
- Gives feature importance easily

---

## 🧠 Prediction Logic

**Issue you fixed:**  
Model expected encoded features like `priority_enc` that were missing during prediction.

**Fix:**  
`predict_delay()` now automatically re-encodes missing columns (self-healing).

**Result:**  
The model can always run predictions even if the dataset reloads fresh.

---

## 🗺️ Route Scoring

Each route is scored using a **weighted heuristic**:

score = α * time_norm + β * cost_norm + γ * emission_norm

| Parameter | Description | Default Weight |
|------------|-------------|----------------|
| α | Delivery time factor | 0.6 |
| β | Delivery cost factor | 0.3 |
| γ | CO₂ emissions factor | 0.1 |

### Example
| Metric | Value | Normalized |
|---------|--------|-------------|
| Time (min) | 90 | 0.75 |
| Cost (INR) | 300 | 0.3 |
| Emissions (kg) | 12 | 0.24 |

score = 0.60.75 + 0.30.3 + 0.1*0.24 = 0.564

→ Lower score = better route.

---

## 🚛 Vehicle Assignment

### Process
1. Filter available vehicles (`status = available, idle, in service`).  
2. Estimate:
   - **Time:** `(distance / avg_speed) * 60`
   - **Fuel cost:** `(distance / fuel_eff) * fuel_price`
   - **CO₂:** `(distance * co2_per_km)`
3. Compute route score and select lowest-score vehicle.

### Example
| Vehicle | Speed | FuelEff | Score | Chosen |
|----------|--------|----------|--------|---------|
| Truck A | 50 km/h | 12 km/l | 0.35 | ✅ |
| Van B | 30 km/h | 8 km/l | 0.56 | ❌ |

---

## 💻 Streamlit Interface

### Tabs
| Tab | Purpose |
|------|----------|
| 🧭 **Guide** | Explains how the dashboard works |
| 📊 **Overview** | Shows merged data and KPIs |
| 🤖 **Predictive** | Trains and tests delay prediction model |
| 🗺️ **Route Planner** | Visualizes routes and optimizes heuristics |
| 🚚 **Fleet Manager** | Displays vehicle data and status charts |
| 📤 **Export** | Downloads predictions and assignments |

### Sidebar
- `Reload Data` button to refresh all sources dynamically.

### Session State
- Stores trained model (`st.session_state["delay_model"]`) for use across tabs.

---

## 🧰 Error Handling & Edge Cases

| Case | Issue | Fix |
|-------|-------|------|
| Duplicate columns | PyArrow crash on dataframe render | `df.loc[:, ~df.columns.duplicated()]` |
| Missing encodings | Model key error | Auto re-encode inside `predict_delay()` |
| Too few samples | Training crash | Minimum rows enforced |
| Missing numeric fields | NaNs in model | Fill with `0` (safe default) |

---

## 📊 Validation & Monitoring

### During Training
- Classification report, F1-score, confusion matrix
- `st.json(report)` displays structured metrics

### Offline Checks
- Train on older orders → test on recent ones
- Compare predicted vs actual delays

### Production
- Monitor predicted vs real delay rates
- Detect drift in average distance or delay ratio

---

## 🚀 Future Improvements

| Goal | Description |
|------|--------------|
| 🧩 Persist encoders | Save LabelEncoders to `.pkl` for consistent mappings |
| 📈 Explainability | Add feature importance & confusion matrix chart |
| 🛣️ Route optimization | Replace heuristics with OR-Tools ILP |
| 🗺️ Mapping | Use Folium or Mapbox to visualize routes |
| ☁️ Deployment | Containerize (Docker) or host on Streamlit Cloud |
| ⚙️ Retraining | Automate nightly retraining jobs |

---

## 🧮 Worked Example

**orders.csv**
order_id,priority,product_category,origin,destination,order_value_inr
101,Express,Electronics,Mumbai,Delhi,15000

**delivery_performance.csv**
order_id,promised_delivery_days,actual_delivery_days,delivery_cost_inr
101,2,3,450

**Resulting Derived Columns**
| order_id | delay_days | delay_flag | priority_enc | product_cat_enc |
|-----------|-------------|-------------|---------------|----------------|
| 101 | 1 | 1 | 2 | 4 |

**Predicted Output**
predicted_delay = 1 (High Risk)

**Route Planner**
| Distance | Cost | Score | Recommended Vehicle |
|-----------|------|--------|----------------------|
| 120 km | ₹450 | 0.35 | Truck V1 |

---

## ⚙️ Run Locally

### 1️⃣ Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate
2️⃣ Install dependencies
bash
Copy code
pip install -r requirements.txt
3️⃣ Launch the app
bash
Copy code
streamlit run app.py
Open your browser → http://localhost:8501
```

👤 Author
Kaushal Shukla
Final Year B.Tech (CSE) — Manipal University Jaipur




