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
    
    return {
        "ev_id": "PREDICTED_ASSET",
        "state_of_health_percentage": round(soh_pred, 2),
        "degradation_rate_per_month": round(deg_pred, 2),
        "remaining_useful_life_months": rul_months,
        "status": "Healthy" if soh_pred > 80.0 else "Attention Needed",
        "source": "model_prediction"
    }
