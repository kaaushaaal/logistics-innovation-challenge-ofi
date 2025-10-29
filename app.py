# file: app.py
import streamlit as st
import pandas as pd
from modules import data_loader, feature_engineering, model_utils, route_utils
import plotly.express as px

st.set_page_config(page_title="Ultimate Logistics Suite", layout="wide")

# Load raw data
st.sidebar.title("Controls")
if st.sidebar.button("Reload data"):
    st.experimental_rerun()

data = data_loader.load_all()
info = data_loader.sample_info(data)
st.sidebar.subheader("Data files")
for k, v in info.items():
    st.sidebar.write(f"{k}: {v['rows']} rows, {v['cols']} cols")

# Merge & featurize
df = feature_engineering.merge_all(data)

# Overview tab
tabs = st.tabs(["Guide", "Overview", "Predictive", "Route Planner", "Fleet", "Export"])
# --- Info / Guide tab ---
with tabs[0]:
    st.title("📘 Ultimate Logistics Suite — User Guide")
    st.markdown("""
    Welcome to **Ultimate Logistics Suite**, an integrated logistics intelligence dashboard built with **Streamlit**.
    It brings together **data analytics**, **machine learning**, and **route optimization** to help logistics teams improve delivery efficiency.

    ---
    ### 🔍 Overview Tab
    - Shows the merged dataset from all sources (orders, delivery performance, fleet, routes, etc.).
    - Displays **key performance indicators (KPIs)** like total orders, delayed orders, average delay days, and average delivery cost.
    - Helps you validate data integrity and get a quick performance snapshot.

    ---
    ### 🤖 Predictive Tab
    - Uses a **RandomForest model** to predict `delay_flag` (whether a delivery will be delayed).
    - You can control the training fraction, view evaluation metrics, and visualize delay risk distribution.
    - High predicted delays indicate risky orders that need attention or priority handling.

    ---
    ### 🗺️ Route Planner Tab
    - Visualizes order delivery routes and costs.
    - You can filter by origin and adjust **weight sliders** (time, cost, emissions) to generate a heuristic score.
    - Auto-assigns the best vehicle for selected orders based on cost, distance, and emission efficiency.

    ---
    ### 🚚 Fleet Tab
    - Displays current fleet snapshot and utilization.
    - Helps identify inactive or overused vehicles using visual summaries like status histograms.

    ---
    ### 📤 Export Tab
    - Download model predictions or sample vehicle assignments as CSVs.
    - Ideal for post-analysis or integrating into a logistics reporting pipeline.

    ---
    ### 💡 Technical Summary
    - **Backend:** Python, Streamlit, scikit-learn, Plotly
    - **ML Model:** RandomForestClassifier
    - **Architecture:** Modular with separate scripts for data loading, modeling, and routing
    - **Author:** Kaushal Shukla (Final Year B.Tech CSE | Cloud, DevOps & Security Enthusiast)
    """)

    st.success("💬 Tip: Use the sidebar 'Reload data' button if you’ve updated your CSVs or inputs.")

with tabs[1]:
    st.header("Data Overview & KPIs")
    st.markdown("### Data sample")
    try:
        st.dataframe(df.head(10))
    except Exception as e:
        st.error(f"Failed to show dataframe: {e}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total orders", len(df))
    col2.metric("Delayed orders", int(df['delay_flag'].sum()) if 'delay_flag' in df.columns else 0)
    if 'delay_days' in df.columns:
        col3.metric("Avg delay days", f"{df['delay_days'].mean():.2f}")
    if 'delivery_cost_inr' in df.columns:
        col4.metric("Avg delivery cost (INR)", f"{df['delivery_cost_inr'].mean():.0f}")

with tabs[2]:
    st.header("Predictive Delivery Optimizer")
    st.write("Train a RandomForest to predict delay_flag.")
    train_frac = st.slider("Training fraction", min_value=0.1, max_value=1.0, value=1.0, step=0.1)
    if st.button("Train model"):
        try:
            model_result = model_utils.train_delay_model(df, sample_frac=train_frac)
            st.success("Model trained")
            st.write("Metrics:", model_result["metrics"])
            st.json(model_result["report"])
            st.session_state["delay_model"] = model_result
        except Exception as e:
            st.error(f"Model training failed: {e}")

    if "delay_model" in st.session_state:
        preds = model_utils.predict_delay(st.session_state["delay_model"], df)
        df["predicted_delay"] = preds
        st.subheader("Predicted delay distribution")
        fig = px.histogram(df, x="predicted_delay")
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Top risky orders")
        st.dataframe(df[df["predicted_delay"] == 1].sort_values("delay_days", ascending=False).head(30))

with tabs[3]:
    st.header("Smart Route Planner")
    origin = st.selectbox("Filter by origin", options=["All"] + sorted(df["origin"].dropna().unique().tolist()))
    df_routes = df.copy()
    if origin != "All":
        df_routes = df_routes[df_routes["origin"] == origin]

    st.subheader("Route KPIs")
    if "distance_km" in df_routes.columns or "distance" in df_routes.columns:
        x = "distance_km" if "distance_km" in df_routes.columns else "distance"
        if "delivery_cost_inr" in df_routes.columns:
            fig = px.scatter(df_routes, x=x, y="delivery_cost_inr", color="priority_simple", hover_data=["order_id"])
            st.plotly_chart(fig, use_container_width=True)
    st.subheader("Rank orders by heuristic score")
    a = st.slider("Time weight", 0.0, 1.0, 0.6)
    b = st.slider("Cost weight", 0.0, 1.0, 0.3)
    c = st.slider("Emissions weight", 0.0, 1.0, 0.1)
    df_routes["heuristic_score"] = df_routes.apply(
        lambda r: route_utils.route_score_row(
            distance_km=r.get("distance_km", r.get("distance", 10)),
            estimated_time_min=r.get("traffic_delay_min", None),
            delivery_cost=r.get("delivery_cost_inr", None),
            co2_per_km=r.get("co2_emissions_kg_per_km", 0.2),
            alpha=a, beta=b, gamma=c
        ), axis=1
    )
    st.dataframe(df_routes.sort_values("heuristic_score").head(50))

    st.markdown("---")
    st.subheader("Auto-assign vehicles for selected orders")
    order_ids = df_routes["order_id"].dropna().astype(str).unique().tolist()
    selected = st.multiselect("Order IDs", options=order_ids, default=order_ids[:5])
    vehicles_df = data.get("vehicles", pd.DataFrame())
    if st.button("Assign best vehicles"):
        results = []
        for oid in selected:
            orow = df_routes[df_routes["order_id"].astype(str) == str(oid)].iloc[0].to_dict()
            assigned = route_utils.assign_vehicle(orow, vehicles_df)
            results.append({"order_id": oid, "assigned": assigned})
        st.dataframe(pd.DataFrame(results))

with tabs[4]:
    st.header("Fleet Manager")
    vdf = data.get("vehicles", pd.DataFrame())
    st.write("Fleet snapshot")
    st.dataframe(vdf.head(50))
    if "status" in vdf.columns:
        fig = px.histogram(vdf, x="status")
        st.plotly_chart(fig, use_container_width=True)

with tabs[5]:
    st.header("Export")
    if "delay_model" in st.session_state:
        df_out = df.copy()
        df_out["predicted_delay"] = model_utils.predict_delay(st.session_state["delay_model"], df)
        csv = df_out.to_csv(index=False).encode()
        st.download_button("Download predictions CSV", csv, file_name="predictions.csv", mime="text/csv")
    st.write("Sample assignments download")
    sample = pd.DataFrame([{"order_id":"sample1","vehicle_id":"veh_1","est_time_min":60}])
    st.download_button("Download sample assignments", sample.to_csv(index=False).encode(), file_name="assignments.csv")

