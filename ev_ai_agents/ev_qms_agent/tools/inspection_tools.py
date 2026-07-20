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
        
    total_cells = len(row)
    scrap_cells = len(row[row['QC_Grade'] == 'Scrap'])
    scrap_rate = (scrap_cells / total_cells) * 100.0 if total_cells > 0 else 0.0
    
    # Collect defect types and inspector comments
    defect_types = row[row['Defect_Type'] != 'None']['Defect_Type'].unique().tolist()
    
    return {
        "batch_id": str(batch_id),
        "total_inspected": total_cells,
        "scrap_rate_pct": round(scrap_rate, 2),
        "avg_resistance_mOhm": round(row['Internal_Resistance_mOhm'].mean(), 2),
        "avg_capacity_mAh": round(row['Capacity_mAh'].mean(), 1),
        "avg_retention_50_cycle_pct": round(row['Retention_50Cycle_Pct'].mean(), 2),
        "avg_electrolyte_volume_ml": round(row['Electrolyte_Volume_ml'].mean(), 2) if 'Electrolyte_Volume_ml' in row.columns else 0.0,
        "defect_types": ", ".join(defect_types) if defect_types else "None",
    }

@tool
def aggregate_qms_statistics(metric: str = "all") -> dict:
    """Aggregates manufacturing quality metrics from the QMS dataset.
    Supported metrics: 'scrap_rate', 'capacity', 'resistance', 'all'.
    """
    if not os.path.exists(DATASET_PATH):
        return {"error": "Dataset not found"}
    df = pd.read_csv(DATASET_PATH)
    if df.empty:
        return {"error": "Dataset is empty"}
        
    result = {"source": "qms_dataset_aggregation", "metric_analyzed": metric}
    total_cells = len(df)
    total_batches = int(df['Batch_ID'].nunique())
    
    if metric in ("scrap_rate", "all"):
        scrap_df = df[df['QC_Grade'] == 'Scrap']
        scrap_count = len(scrap_df)
        scrap_rate_pct = (scrap_count / total_cells) * 100.0 if total_cells > 0 else 0.0
        grade_a_count = len(df[df['QC_Grade'] == 'Grade A'])
        grade_b_count = len(df[df['QC_Grade'] == 'Grade B'])
        
        # Distributions by line
        line_stats = {}
        for line in df['Production_Line'].unique():
            line_df = df[df['Production_Line'] == line]
            line_total = len(line_df)
            line_scrap = len(line_df[line_df['QC_Grade'] == 'Scrap'])
            line_stats[str(line)] = round((line_scrap / line_total) * 100.0, 2) if line_total > 0 else 0.0
            
        # Distributions by shift
        shift_stats = {}
        for shift in df['Shift'].unique():
            shift_df = df[df['Shift'] == shift]
            shift_total = len(shift_df)
            shift_scrap = len(shift_df[shift_df['QC_Grade'] == 'Scrap'])
            shift_stats[str(shift)] = round((shift_scrap / shift_total) * 100.0, 2) if shift_total > 0 else 0.0
            
        defect_series = df['Defect_Type'].dropna()
        defect_series = defect_series[defect_series != 'None']
        defect_series = defect_series[defect_series != 'nan']
        defect_counts = defect_series.value_counts().to_dict()
        
        result.update({
            "total_cells_inspected": total_cells,
            "total_batches_inspected": total_batches,
            "overall_scrap_defect_rate_pct": round(scrap_rate_pct, 2),
            "grades_distribution": {
                "Grade A": grade_a_count,
                "Grade B": grade_b_count,
                "Scrap": scrap_count
            },
            "scrap_rate_by_line_pct": line_stats,
            "scrap_rate_by_shift_pct": shift_stats,
            "defect_categories": defect_counts
        })
        
    if metric in ("capacity", "all"):
        avg_capacity = float(df['Capacity_mAh'].mean())
        avg_electrolyte_vol = float(df['Electrolyte_Volume_ml'].mean())
        result.update({
            "average_cell_capacity_mAh": round(avg_capacity, 2),
            "average_electrolyte_volume_ml": round(avg_electrolyte_vol, 2)
        })
        
    if metric in ("resistance", "all"):
        avg_internal_resistance = float(df['Internal_Resistance_mOhm'].mean())
        avg_ambient_temp = float(df['Ambient_Temp_C'].mean())
        result.update({
            "average_internal_resistance_mOhm": round(avg_internal_resistance, 2),
            "average_ambient_temperature_celsius": round(avg_ambient_temp, 2)
        })
        
    return result

