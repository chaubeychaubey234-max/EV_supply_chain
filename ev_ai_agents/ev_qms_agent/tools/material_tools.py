import pandas as pd
import os
from langchain.tools import tool

DATASET_PATH = "/Users/alex-ankush/Desktop/EV/EV_supply_chain/ev_ai_agents/datasets/qms_dataset.csv"

@tool
def fetch_material_data(batch_id: str) -> dict:
    """Fetches incoming raw material quality metrics for a given batch from the dataset."""
    if not os.path.exists(DATASET_PATH):
        return {"error": "Dataset not found"}
        
    df = pd.read_csv(DATASET_PATH)
    row = df[df['Batch_ID'] == batch_id]
    if row.empty:
        row = df.sample(1)
        
    return {
        "batch_id": str(row['Batch_ID'].values[0]),
        "supplier": str(row['Supplier'].values[0]),
        "anode_overhang_mm": round(float(row['Anode_Overhang_mm'].values[0]), 2),
        "electrolyte_volume_ml": round(float(row['Electrolyte_Volume_ml'].values[0]), 2),
    }
