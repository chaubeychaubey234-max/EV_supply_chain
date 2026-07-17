import pandas as pd
import os
from langchain.tools import tool

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(_BASE_DIR, "..", "..", "datasets", "apm_dataset.csv")

@tool
def fetch_thermal_events(ev_id: str) -> dict:
    """Fetches thermal history and anomaly data for a given EV's battery pack from the dataset."""
    if not os.path.exists(DATASET_PATH):
        return {"error": "Dataset not found"}
        
    df = pd.read_csv(DATASET_PATH)
    row = df[df['EV_ID'] == ev_id]
    if row.empty:
        return {"error": f"No record found for {ev_id}."}
        
    avg_temp = float(row['Avg_Temperature_C'].values[0])
    max_temp = float(row['Max_Temperature_C'].values[0])
    thermal_events = 1 if max_temp > 50.0 else 0
    if max_temp > 55.0:
        thermal_events += 2
        
    return {
        "ev_id": str(row['EV_ID'].values[0]),
        "average_operating_temperature_celsius": round(avg_temp, 1),
        "max_recorded_temperature_celsius": round(max_temp, 1),
        "thermal_runaway_warnings": thermal_events,
        "cooling_system_status": "OK" if max_temp < 55.0 else "Degraded"
    }
