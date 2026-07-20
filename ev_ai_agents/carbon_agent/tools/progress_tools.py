import pandas as pd
from langchain.tools import tool
from ev_ai_agents.carbon_agent.utils.data_loader import load_co2_emissions

class ToolError(Exception):
    """Custom exception for progress tools."""
    pass

@tool
def track_net_zero_progress() -> dict:
    """Track organizational progress toward net zero goals using historical data.
    
    Returns:
        dict: A dictionary containing progress_percentage, emissions_gap, and status.
    """
    try:
        df = load_co2_emissions()
        
        # Sort by year
        df = df.sort_values(by='year').reset_index(drop=True)
        
        if df.empty:
            raise ToolError("CO2 emissions tracking dataset is empty.")
            
        # Baseline is the first year in the dataset (2020)
        baseline_row = df.iloc[0]
        baseline_emissions = baseline_row['total_emissions']
        
        # Let's assess the latest actual reporting year (e.g., 2024)
        # 2025 and 2030 are projection/target years in our dataset
        actual_df = df[df['year'] <= 2024]
        if actual_df.empty:
            actual_df = df
            
        latest_row = actual_df.iloc[-1]
        latest_year = int(latest_row['year'])
        current_emissions = latest_row['total_emissions']
        target_emissions = latest_row['target_emissions']
        
        emissions_gap = current_emissions - target_emissions
        
        # Progress Calculation: How much of the targeted reduction from baseline have we achieved?
        # Target reduction = baseline - target
        # Achieved reduction = baseline - current
        target_reduction = baseline_emissions - target_emissions
        achieved_reduction = baseline_emissions - current_emissions
        
        if target_reduction > 0:
            progress_percentage = (achieved_reduction / target_reduction) * 100.0
        else:
            progress_percentage = 100.0 if current_emissions <= target_emissions else 0.0
            
        progress_percentage = round(min(max(progress_percentage, 0.0), 100.0), 2)
        
        # Determine status
        if current_emissions <= target_emissions:
            status = "On Track"
        elif emissions_gap / target_emissions <= 0.05:
            status = "At Risk"
        else:
            status = "Behind"
            
        return {
            "year": latest_year,
            "current_emissions": round(current_emissions, 2),
            "target_emissions": round(target_emissions, 2),
            "progress_percentage": progress_percentage,
            "emissions_gap": round(emissions_gap, 2),
            "status": status,
            "co2_history": df.to_dict(orient='records')
        }
    except Exception as e:
        raise ToolError(f"Error tracking net zero progress: {str(e)}")
