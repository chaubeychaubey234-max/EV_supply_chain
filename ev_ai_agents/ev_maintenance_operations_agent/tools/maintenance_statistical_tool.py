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
    
    Computes average battery health, average risk score, risk level breakdown
    (critical/high/medium/low), number of overdue vehicles, average workshop utilization,
    average charging station uptime, and top critical asset details.
    
    Returns:
        A dictionary containing fleet-wide maintenance statistics.
    """
    try:
        df_hist = load_maintenance_history()
        df_work = load_workshop_capacity()
        df_char = load_charging_stations()
        
        # 1. Total vehicles
        total_vehicles = int(df_hist['vehicle_id'].nunique())

        # 2. Average battery health
        avg_battery_health = float(df_hist['battery_health_percent'].mean())
        
        # 3. Compute composite risk score for each vehicle
        def _calc_score(row):
            bat = float(row['battery_health_percent'])
            b_pts = 30 if bat < 60 else (20 if bat < 70 else (10 if bat < 80 else (5 if bat < 90 else 0)))
            age = float(row['vehicle_age_years'])
            a_pts = 20 if age > 8 else (12 if age > 5 else (6 if age > 3 else (2 if age > 1 else 0)))
            km = float(row['total_km_driven'])
            k_pts = 20 if km > 200000 else (12 if km > 120000 else (6 if km > 60000 else (2 if km > 20000 else 0)))
            cyc = int(row['charging_cycles'])
            c_pts = 15 if cyc > 1500 else (10 if cyc > 1000 else (5 if cyc > 500 else 0))
            flt = str(row['fault_code']).strip().upper()
            f_pts = 10 if flt.startswith('P0') else (7 if flt.startswith('P1') else (3 if flt.startswith('P2') else 0))
            days = int(row.get('days_since_service', 100))
            s_pts = 5 if days > 365 else (3 if days > 180 else (1 if days > 90 else 0))
            return min(100, b_pts + a_pts + k_pts + c_pts + f_pts + s_pts)

        df_hist['computed_risk_score'] = df_hist.apply(_calc_score, axis=1)
        avg_risk_score = float(df_hist['computed_risk_score'].mean())

        # Risk breakdown
        critical_risk_count = int((df_hist['computed_risk_score'] >= 76).sum())
        high_risk_count = int(((df_hist['computed_risk_score'] >= 51) & (df_hist['computed_risk_score'] < 76)).sum())
        medium_risk_count = int(((df_hist['computed_risk_score'] >= 26) & (df_hist['computed_risk_score'] < 51)).sum())
        low_risk_count = int((df_hist['computed_risk_score'] <= 25).sum())

        # 4. Number of overdue vehicles
        overdue_mask = (
            (df_hist['maintenance_overdue_flag'] == True) | 
            (df_hist['maintenance_overdue_flag'] == 'True') | 
            (df_hist['maintenance_overdue_flag'] == 1) |
            (df_hist['days_since_service'] > 180)
        )
        num_overdue = int(overdue_mask.sum())
        
        # 5. Average workshop workload/utilization
        avg_workshop_utilization = float(df_work['current_workload_percent'].mean())
        
        # 6. Average charging station uptime
        avg_charging_uptime = float(df_char['station_uptime_percent'].mean())

        # Detailed records for top critical/high risk vehicles
        top_critical_df = df_hist.sort_values('computed_risk_score', ascending=False).head(5)
        top_critical_list = []
        for _, r in top_critical_df.iterrows():
            top_critical_list.append({
                "vehicle_id": str(r["vehicle_id"]),
                "vehicle_model": str(r["vehicle_model"]),
                "risk_score": int(r["computed_risk_score"]),
                "risk_level": "CRITICAL" if r["computed_risk_score"] >= 76 else ("HIGH" if r["computed_risk_score"] >= 51 else "MEDIUM"),
                "battery_health_percent": float(r["battery_health_percent"]),
                "fault_code": str(r["fault_code"]),
                "total_km_driven": float(r["total_km_driven"]),
                "charging_cycles": int(r["charging_cycles"]),
                "vehicle_age_years": float(r["vehicle_age_years"]),
                "temperature_avg": float(r.get("temperature_avg", 35.0)),
                "days_since_service": int(r.get("days_since_service", 90))
            })
        
        overdue_sample = df_hist[overdue_mask]['vehicle_id'].astype(str).tolist()[:5]
        
        return {
            "total_vehicles_inspected": total_vehicles,
            "average_battery_health_pct": round(avg_battery_health, 2),
            "average_risk_score": round(avg_risk_score, 2),
            "critical_risk_vehicles_count": critical_risk_count,
            "high_risk_vehicles_count": high_risk_count,
            "medium_risk_vehicles_count": medium_risk_count,
            "low_risk_vehicles_count": low_risk_count,
            "number_of_overdue_vehicles": num_overdue,
            "average_workshop_utilization_pct": round(avg_workshop_utilization, 2),
            "average_charging_uptime_pct": round(avg_charging_uptime, 2),
            "sample_overdue_vehicles": overdue_sample,
            "top_critical_assets": top_critical_list
        }
    except Exception as e:
        return {"error": f"Failed to compute statistics: {str(e)}"}

