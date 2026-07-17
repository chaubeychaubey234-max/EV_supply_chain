import pandas as pd
import os
from langchain.tools import tool

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(_BASE_DIR, "..", "..", "datasets", "apm_dataset.csv")

@tool
def fetch_charging_patterns(ev_id: str) -> dict:
    """Fetches charging cycle history and patterns for a given EV from the dataset."""
    if not os.path.exists(DATASET_PATH):
        return {"error": "Dataset not found"}
        
    df = pd.read_csv(DATASET_PATH)
    row = df[df['EV_ID'] == ev_id]
    if row.empty:
        return {"error": f"No record found for {ev_id}."}
        
    fast_charge_percentage = float(row['Fast_Charge_Ratio_Pct'].values[0])
    deep_discharge_cycles = int(row['Deep_Discharge_Cycles'].values[0])
    avg_charge_duration_hours = float(row['Avg_Charge_Duration_Hours'].values[0])
    
    return {
        "ev_id": str(row['EV_ID'].values[0]),
        "fast_charging_ratio_percentage": round(fast_charge_percentage, 1),
        "deep_discharge_cycles_last_month": deep_discharge_cycles,
        "average_charge_duration_hours": round(avg_charge_duration_hours, 1),
        "pattern_risk_level": "High" if fast_charge_percentage > 60.0 or deep_discharge_cycles > 30 else "Low"
    }
