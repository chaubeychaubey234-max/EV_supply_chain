import pandas as pd
import os
import joblib
from langchain.tools import tool

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(_BASE_DIR, "..", "..", "datasets", "qms_dataset.csv")
MODELS_DIR = os.path.join(_BASE_DIR, "..", "models")

@tool
def fetch_material_data(batch_id: str) -> dict:
    """Fetches incoming raw material quality metrics for a given batch from the dataset."""
    if not os.path.exists(DATASET_PATH):
        return {"error": "Dataset not found"}
        
    df = pd.read_csv(DATASET_PATH)
    row = df[df['Batch_ID'] == batch_id]
    if row.empty:
        return {"error": f"No record found for batch {batch_id}. Provide values directly for prediction."}
        
    return {
        "batch_id": str(row['Batch_ID'].values[0]),
        "supplier": str(row['Supplier'].values[0]),
        "anode_overhang_mm": round(float(row['Anode_Overhang_mm'].values[0]), 2),
        "electrolyte_volume_ml": round(float(row['Electrolyte_Volume_ml'].values[0]), 2),
        "source": "historical_lookup"
    }

@tool
def predict_quality_drift(ambient_temp_c: float, anode_overhang_mm: float, electrolyte_volume_ml: float, internal_resistance_mohm: float, capacity_mah: float, retention_50cycle_pct: float) -> dict:
    """Predicts manufacturing quality grade (Pass/Scrap) based on raw process and material parameters."""
    model_path = os.path.join(MODELS_DIR, "qc_classifier.pkl")
    if not os.path.exists(model_path):
        return {"error": "Prediction model not found. Please train model first."}
        
    clf = joblib.load(model_path)
    
    # Create input dataframe
    input_data = pd.DataFrame([{
        "Ambient_Temp_C": ambient_temp_c,
        "Anode_Overhang_mm": anode_overhang_mm,
        "Electrolyte_Volume_ml": electrolyte_volume_ml,
        "Internal_Resistance_mOhm": internal_resistance_mohm,
        "Capacity_mAh": capacity_mah,
        "Retention_50Cycle_Pct": retention_50cycle_pct
    }])
    
    pred_idx = int(clf.predict(input_data)[0])
    probs = clf.predict_proba(input_data)[0]
    confidence = float(probs[pred_idx])
    
    grade_map = {0: "Grade A", 1: "Grade B", 2: "Scrap"}
    predicted_grade = grade_map.get(pred_idx, "Unknown")
    
    return {
        "predicted_qc_grade": predicted_grade,
        "confidence": round(confidence, 2),
        "risk_level": "High" if predicted_grade == "Scrap" else "Low" if predicted_grade == "Grade A" else "Medium",
        "source": "model_prediction"
    }
