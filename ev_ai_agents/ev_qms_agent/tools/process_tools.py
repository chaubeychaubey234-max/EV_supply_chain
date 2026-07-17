import pandas as pd
import os
from langchain.tools import tool

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(_BASE_DIR, "..", "..", "datasets", "qms_dataset.csv")

@tool
def fetch_process_data(batch_id: str) -> dict:
    """Fetches machine process parameters during cell manufacturing from the dataset."""
    if not os.path.exists(DATASET_PATH):
        return {"error": "Dataset not found"}
        
    df = pd.read_csv(DATASET_PATH)
    row = df[df['Batch_ID'] == batch_id]
    if row.empty:
        return {"error": f"No record found for batch {batch_id}."}
        
    return {
        "batch_id": str(row['Batch_ID'].values[0]),
        "production_line": str(row['Production_Line'].values[0]),
        "shift": str(row['Shift'].values[0]),
        "ambient_temperature_celsius": round(float(row['Ambient_Temp_C'].values[0]), 1)
    }
