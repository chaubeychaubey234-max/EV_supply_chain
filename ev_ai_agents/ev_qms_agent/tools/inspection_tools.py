import pandas as pd
import os
from langchain.tools import tool

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(_BASE_DIR, "..", "..", "datasets", "qms_dataset.csv")

@tool
def fetch_inspection_data(batch_id: str) -> dict:
    """Fetches in-line Quality Assurance (QA) tests and defect rates from the dataset."""
    if not os.path.exists(DATASET_PATH):
        return {"error": "Dataset not found"}
        
    df = pd.read_csv(DATASET_PATH)
    row = df[df['Batch_ID'] == batch_id]
    if row.empty:
        return {"error": f"No record found for batch {batch_id}."}
        
    return {
        "batch_id": str(row['Batch_ID'].values[0]),
        "internal_resistance_mOhm": round(float(row['Internal_Resistance_mOhm'].values[0]), 2),
        "capacity_mAh": round(float(row['Capacity_mAh'].values[0]), 1),
        "retention_50_cycle_pct": round(float(row['Retention_50Cycle_Pct'].values[0]), 2),
        "qc_grade": str(row['QC_Grade'].values[0]),
        "defect_type": str(row['Defect_Type'].values[0]),
        "inspector_comment": str(row['Inspector_Comment'].values[0])
    }
