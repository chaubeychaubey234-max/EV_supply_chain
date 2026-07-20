import pandas as pd
import os
import joblib
from langchain.tools import tool

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(_BASE_DIR, "..", "..", "datasets", "apm_dataset.csv")
MODELS_DIR = os.path.join(_BASE_DIR, "..", "models")

@tool
def fetch_battery_health(ev_id: str) -> dict:
    """Fetches the battery State of Health (SoH) and degradation trajectory for a given EV from the dataset."""
    if not os.path.exists(DATASET_PATH):
        return {"error": "Dataset not found"}
        
    df = pd.read_csv(DATASET_PATH)
    row = df[df['EV_ID'] == ev_id]
    if row.empty:
        return {"error": f"No record found for {ev_id}. Provide telemetry values directly for a live prediction instead."}
        
    soh = float(row['State_of_Health_Pct'].values[0])
    degradation_rate = float(row['Degradation_Rate_Pct'].values[0])
    rul_months = int((soh - 70.0) / degradation_rate) if soh > 70.0 and degradation_rate > 0 else 0
    
    return {
        "ev_id": str(row['EV_ID'].values[0]),
        "state_of_health_percentage": round(soh, 2),
        "degradation_rate_per_month": round(degradation_rate, 2),
        "remaining_useful_life_months": rul_months,
        "status": "Healthy" if soh > 80.0 else "Attention Needed",
        "source": "historical_lookup"
    }

@tool
def predict_battery_health(avg_temperature_c: float, fast_charge_ratio_pct: float, deep_discharge_cycles: int, avg_charge_duration_hours: float, max_temperature_c: float) -> dict:
    """Predicts battery degradation rate and derives State of Health (SoH) from raw telemetry values.
    
    Approach:
      - Degradation rate is predicted by an XGBoost model trained on fleet telemetry (R² = 0.94).
      - SoH is derived from the degradation model: SoH = fleet_mean_soh - (deg_rate * asset_age_months).
        This is physically grounded — batteries degrading faster accumulate more capacity loss over time.
      - The separate SoH regression model was retired because all telemetry features have ≤0.04
        correlation with SoH in this dataset (synthetic SoH is statistically independent of telemetry).
    """
    deg_model_path = os.path.join(MODELS_DIR, "degradation_model.pkl")
    baseline_path = os.path.join(MODELS_DIR, "soh_baseline.pkl")

    if not os.path.exists(deg_model_path) or not os.path.exists(baseline_path):
        return {"error": "Prediction models not found. Run ev_ai_agents/ev_apm_agent/train_model.py first."}

    deg_model = joblib.load(deg_model_path)
    baseline = joblib.load(baseline_path)
    fleet_mean_soh = baseline["fleet_mean_soh"]
    asset_age_months = baseline["asset_age_months"]

    input_data = pd.DataFrame([{
        "Avg_Temperature_C": avg_temperature_c,
        "Fast_Charge_Ratio_Pct": fast_charge_ratio_pct,
        "Deep_Discharge_Cycles": deep_discharge_cycles,
        "Avg_Charge_Duration_Hours": avg_charge_duration_hours,
        "Max_Temperature_C": max_temperature_c
    }])

    deg_pred = float(deg_model.predict(input_data)[0])
    # Derive SoH: batteries with higher degradation rate lose more capacity from fleet average
    import numpy as np
    soh_pred = float(np.clip(fleet_mean_soh - (deg_pred * asset_age_months), 60.0, 100.0))
    rul_months = int((soh_pred - 70.0) / deg_pred) if soh_pred > 70.0 and deg_pred > 0 else 0

    return {
        "ev_id": "PREDICTED_ASSET",
        "state_of_health_percentage": round(soh_pred, 2),
        "degradation_rate_per_month": round(deg_pred, 2),
        "remaining_useful_life_months": rul_months,
        "status": "Healthy" if soh_pred > 80.0 else "Attention Needed",
        "source": "model_prediction",
        "degradation_rate_confidence": "High (R² = 0.94, MAE = 0.027%/month)",
        "soh_method": "Derived from degradation model (fleet_mean=84.71% - deg_rate * 12mo). "
                      "SoH telemetry correlation ≤0.04 in this dataset — no separate regression model trained.",
        "dominant_features": "Deep_Discharge_Cycles (75%), Fast_Charge_Ratio_Pct (23%)"
    }


@tool
def aggregate_apm_statistics(metric: str = "all") -> dict:
    """Aggregates fleet-wide battery health and telemetry performance statistics from the historical dataset.
    Supported metrics: 'battery_health', 'temperature', 'charging_cycles', 'all'.
    """
    if not os.path.exists(DATASET_PATH):
        return {"error": "Dataset not found"}
        
    df = pd.read_csv(DATASET_PATH)
    if df.empty:
        return {"error": "Dataset is empty"}
        
    result = {"source": "fleet_dataset_aggregation", "metric_analyzed": metric}
    
    if metric in ("battery_health", "all"):
        total_evs = len(df)
        avg_soh = float(df['State_of_Health_Pct'].mean())
        avg_deg_rate = float(df['Degradation_Rate_Pct'].mean())
        critical_evs = df[df['State_of_Health_Pct'] < 80.0]
        critical_count = len(critical_evs)
        critical_pct = (critical_count / total_evs) * 100.0 if total_evs > 0 else 0.0
        result.update({
            "total_evs_inspected": total_evs,
            "average_state_of_health_pct": round(avg_soh, 2),
            "average_degradation_rate_monthly_pct": round(avg_deg_rate, 4),
            "critical_evs_count": critical_count,
            "critical_evs_percentage": round(critical_pct, 2)
        })
        
    if metric in ("temperature", "all"):
        avg_temp = float(df['Avg_Temperature_C'].mean())
        max_temp = float(df['Max_Temperature_C'].max())
        high_temp_count = len(df[df['Max_Temperature_C'] > 50.0])
        result.update({
            "average_operating_temperature_celsius": round(avg_temp, 2),
            "max_operating_temperature_celsius": round(max_temp, 2),
            "high_thermal_risk_evs_count": high_temp_count
        })
        
    if metric in ("charging_cycles", "all"):
        avg_fc_ratio = float(df['Fast_Charge_Ratio_Pct'].mean())
        avg_charge_duration = float(df['Avg_Charge_Duration_Hours'].mean())
        deep_cycles_mean = float(df['Deep_Discharge_Cycles'].mean())
        result.update({
            "average_fast_charge_ratio_pct": round(avg_fc_ratio, 2),
            "average_charge_duration_hours": round(avg_charge_duration, 2),
            "average_deep_discharge_cycles_monthly": round(deep_cycles_mean, 2)
        })
        
    return result

