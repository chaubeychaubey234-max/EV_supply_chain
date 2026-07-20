"""
maintenance_statistical_tool.py
================================
Tool for calculating fleet-wide maintenance statistics.
"""

from __future__ import annotations
import os
import pandas as pd
from langchain_core.tools import tool

# Import path resolution and cache-loaders from utils
from ev_ai_agents.ev_maintenance_operations_agent.tools.utils import (
    load_maintenance_history,
    load_workshop_capacity,
    load_charging_stations
)

@tool
def calculate_maintenance_statistics() -> dict:
    """Calculate fleet-wide maintenance statistics from the datasets.
    
    Computes average battery health, number of overdue vehicles, average
    workshop utilization, average charging station uptime, and lists of critical
    or overdue vehicle samples.
    
    Returns:
        A dictionary containing fleet-wide maintenance statistics.
    """
    try:
        df_hist = load_maintenance_history()
        df_work = load_workshop_capacity()
        df_char = load_charging_stations()
        
        # 1. Average battery health
        avg_battery_health = float(df_hist['battery_health_percent'].mean())
        
        # 2. Number of overdue vehicles (overdue flag or days since service > 180)
        overdue_mask = (
            (df_hist['maintenance_overdue_flag'] == True) | 
            (df_hist['maintenance_overdue_flag'] == 'True') | 
            (df_hist['maintenance_overdue_flag'] == 1) |
            (df_hist['days_since_service'] > 180)
        )
        num_overdue = int(overdue_mask.sum())
        
        # 3. Average workshop workload/utilization
        avg_workshop_utilization = float(df_work['current_workload_percent'].mean())
        
        # 4. Average charging station uptime
        avg_charging_uptime = float(df_char['station_uptime_percent'].mean())
        
        # 5. Total vehicles
        total_vehicles = int(df_hist['vehicle_id'].nunique())
        
        # 6. High risk and critical risk counts
        high_risk_count = int((df_hist['vehicle_risk_score'] > 50).sum())
        critical_risk_count = int((df_hist['vehicle_risk_score'] > 75).sum())
        
        # Lists for narrative context
        overdue_sample = df_hist[overdue_mask]['vehicle_id'].astype(str).tolist()[:5]
        critical_sample = df_hist[df_hist['vehicle_risk_score'] > 50]['vehicle_id'].astype(str).tolist()[:5]
        
        return {
            "total_vehicles_inspected": total_vehicles,
            "average_battery_health_pct": round(avg_battery_health, 2),
            "number_of_overdue_vehicles": num_overdue,
            "average_workshop_utilization_pct": round(avg_workshop_utilization, 2),
            "average_charging_uptime_pct": round(avg_charging_uptime, 2),
            "high_risk_vehicles_count": high_risk_count,
            "critical_risk_vehicles_count": critical_risk_count,
            "sample_overdue_vehicles": overdue_sample,
            "sample_critical_vehicles": critical_sample
        }
    except Exception as e:
        return {"error": f"Failed to compute statistics: {str(e)}"}
