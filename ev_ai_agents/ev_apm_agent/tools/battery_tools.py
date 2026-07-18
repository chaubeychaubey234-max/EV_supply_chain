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
    """Predicts battery State of Health (SoH) and degradation rate based on raw telemetry values."""
    soh_model_path = os.path.join(MODELS_DIR, "soh_model.pkl")
    deg_model_path = os.path.join(MODELS_DIR, "degradation_model.pkl")
    
    if not os.path.exists(soh_model_path) or not os.path.exists(deg_model_path):
        return {"error": "Prediction models not found. Please train models first."}
        
    soh_model = joblib.load(soh_model_path)
    deg_model = joblib.load(deg_model_path)
    
    # Create input dataframe matching training features
    input_data = pd.DataFrame([{
        "Avg_Temperature_C": avg_temperature_c,
        "Fast_Charge_Ratio_Pct": fast_charge_ratio_pct,
        "Deep_Discharge_Cycles": deep_discharge_cycles,
        "Avg_Charge_Duration_Hours": avg_charge_duration_hours,
        "Max_Temperature_C": max_temperature_c
    }])
    
    soh_pred = float(soh_model.predict(input_data)[0])
    deg_pred = float(deg_model.predict(input_data)[0])
    
    rul_months = int((soh_pred - 70.0) / deg_pred) if soh_pred > 70.0 and deg_pred > 0 else 0
    
    # Note: State of Health (SoH) regression performs poorly (R² ≈ -0.27) because the target SoH
    # has very low correlation with raw operating telemetry in this synthetic dataset.
    # We explicitly flag the SoH confidence as experimental/low, while degradation-rate prediction is high-confidence.
    return {
        "ev_id": "PREDICTED_ASSET",
        "state_of_health_percentage": round(soh_pred, 2),
        "degradation_rate_per_month": round(deg_pred, 2),
        "remaining_useful_life_months": rul_months,
        "status": "Healthy" if soh_pred > 80.0 else "Attention Needed",
        "source": "model_prediction",
        "soh_confidence": "Low (Experimental - R² ≈ -0.27 due to low dataset correlation)",
        "degradation_rate_confidence": "High (R² ≈ 0.94)",
        "notes": "State of Health (SoH) prediction has high uncertainty in this iteration. Degradation rate and Remaining Useful Life (RUL) are prioritized for operational planning."
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

