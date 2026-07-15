import pandas as pd
import os
from langchain.tools import tool

DATASET_PATH = "/Users/alex-ankush/Desktop/EV/datasets/qms_dataset.csv"

@tool
def fetch_process_data(batch_id: str) -> dict:
    """Fetches machine process parameters during cell manufacturing from the dataset."""
    if not os.path.exists(DATASET_PATH):
        return {"error": "Dataset not found"}
        
    df = pd.read_csv(DATASET_PATH)
    row = df[df['Batch_ID'] == batch_id]
    if row.empty:
        row = df.sample(1)
        
    return {
        "batch_id": str(row['Batch_ID'].values[0]),
        "production_line": str(row['Production_Line'].values[0]),
        "shift": str(row['Shift'].values[0]),
        "ambient_temperature_celsius": round(float(row['Ambient_Temp_C'].values[0]), 1)
    }
