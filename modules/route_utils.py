# file: modules/route_utils.py
import math

def route_score_row(distance_km=None, estimated_time_min=None, delivery_cost=None, co2_per_km=0.1, alpha=0.6, beta=0.3, gamma=0.1):
    """
    Compute a weighted score (lower better) combining time, cost and emissions.
    These normalizations are heuristic and can be tuned.
    """
    # fallback defaults
    distance_km = float(distance_km) if distance_km is not None else 10.0
    estimated_time_min = float(estimated_time_min) if estimated_time_min is not None else (distance_km / 40.0) * 60.0
    delivery_cost = float(delivery_cost) if delivery_cost is not None else (distance_km * 5.0)
    co2_total = co2_per_km * distance_km

    t_norm = estimated_time_min / 120.0       # 120 min benchmark
    c_norm = delivery_cost / 1000.0           # 1000 INR benchmark
    e_norm = co2_total / 50.0                 # 50 kg CO2 benchmark

    score = alpha * t_norm + beta * c_norm + gamma * e_norm
    return score

def assign_vehicle(order_row, vehicles_df):
    """
    Basic heuristic: filter available vehicles with enough capacity and pick one with best route score.
    Returns a dict with chosen vehicle_id and details, or None.
    """
    candidates = []
    order_weight = order_row.get("weight", 0)  # if present
    distance = order_row.get("distance_km", order_row.get("distance", 10))
    delivery_cost = order_row.get("delivery_cost_inr", order_row.get("delivery_cost", None))

    for _, v in vehicles_df.iterrows():
        status = str(v.get("status", "")).lower()
        if status not in ("available", "idle", "in service", "active"):
            continue
        capacity = v.get("capacity", None)
        if capacity is not None and order_weight and capacity < order_weight:
            continue

        avg_speed = v.get("avg_speed_kmph", 40) if "avg_speed_kmph" in v else 40
        est_time_min = (distance / max(avg_speed, 1.0)) * 60.0
        fuel_eff = v.get("fuel_efficiency_km_per_l", v.get("fuel_efficiency_km_per_l", None))
        if fuel_eff and fuel_eff > 0:
            est_cost = (distance / fuel_eff) * v.get("fuel_price_per_l", 80)
        else:
            est_cost = delivery_cost if delivery_cost is not None else distance * 5.0
        co2 = v.get("co2_emissions_kg_per_km", v.get("co2_emissions_kg_per_km", v.get("co2_emissions_kg_per_km", 0.2)))

        score = route_score_row(distance_km=distance, estimated_time_min=est_time_min, delivery_cost=est_cost, co2_per_km=co2)
        candidates.append((score, v, est_time_min, est_cost, co2))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    best = candidates[0]
    chosen = {
        "vehicle_id": best[1].get("vehicle_id", best[1].get("vehicle_id", None)) if isinstance(best[1], dict) or hasattr(best[1], "get") else best[1].get("vehicle_id", None),
        "score": best[0],
        "est_time_min": best[2],
        "est_cost": best[3],
        "co2_kg_per_km": best[4]
    }
    return chosen
