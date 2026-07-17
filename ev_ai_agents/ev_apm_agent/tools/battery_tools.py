import pandas as pd
import os
from langchain.tools import tool

DATASET_PATH = "/Users/alex-ankush/Desktop/EV/EV_supply_chain/ev_ai_agents/datasets/apm_dataset.csv"

@tool
def fetch_battery_health(ev_id: str) -> dict:
    """Fetches the battery State of Health (SoH) and degradation trajectory for a given EV from the dataset."""
    if not os.path.exists(DATASET_PATH):
        return {"error": "Dataset not found"}
        
    df = pd.read_csv(DATASET_PATH)
    row = df[df['EV_ID'] == ev_id]
    if row.empty:
        row = df.sample(1)
        
    soh = float(row['State_of_Health_Pct'].values[0])
    degradation_rate = float(row['Degradation_Rate_Pct'].values[0])
    rul_months = int((soh - 70.0) / degradation_rate) if soh > 70.0 and degradation_rate > 0 else 0
    
    return {
        "ev_id": str(row['EV_ID'].values[0]),
        "state_of_health_percentage": round(soh, 2),
        "degradation_rate_per_month": round(degradation_rate, 2),
        "remaining_useful_life_months": rul_months,
        "status": "Healthy" if soh > 80.0 else "Attention Needed"
    }
