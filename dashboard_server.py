import os
import sys
import json
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
import pandas as pd
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Add ev_ai_agents subfolder to sys.path
EV_AI_AGENTS_DIR = os.path.join(PROJECT_ROOT, "ev_ai_agents")
if EV_AI_AGENTS_DIR not in sys.path:
    sys.path.insert(0, EV_AI_AGENTS_DIR)

# Import real agents
try:
    from ev_ai_agents.ev_fleet_electrification_agent.agent import run_agent as run_fleet_agent
except ImportError:
    run_fleet_agent = None

try:
    from ev_ai_agents.ev_maintenance_operations_agent.agent import run_agent as run_maint_agent
except ImportError:
    run_maint_agent = None


app = FastAPI(
    title="EV Fleet Intelligence & Supply Chain AI Platform API",
    description="Backend service orchestrating the 6 specialized AI Agents.",
    version="1.0.0"
)

# CORS configuration for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dashboard_server")

# ----------------------------------------------------------------------
# Local Safe Data Loaders (using relative paths)
# ----------------------------------------------------------------------
def load_csv_safe(path: str) -> Optional[pd.DataFrame]:
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            logger.error(f"Error loading CSV at {path}: {e}")
            return None
    logger.warning(f"CSV not found at {path}")
    return None

# Datasets path mapping
APM_CSV = os.path.join(PROJECT_ROOT, "ev_ai_agents", "datasets", "apm_dataset.csv")
QMS_CSV = os.path.join(PROJECT_ROOT, "ev_ai_agents", "datasets", "qms_dataset.csv")
SUPPLY_CHAIN_CSV = os.path.join(PROJECT_ROOT, "ev_ai_agents", "datasets", "ev_supply_chain.csv")
MINERAL_RISK_CSV = os.path.join(PROJECT_ROOT, "ev_ai_agents", "datasets", "critical_minerals_risk.csv")
BATTERY_QUALITY_CSV = os.path.join(PROJECT_ROOT, "ev_ai_agents", "datasets", "battery_quality.csv")
CO2_CSV = os.path.join(PROJECT_ROOT, "ev_ai_agents", "carbon_agent", "datasets", "co2_emissions.csv")
LOGISTICS_CSV = os.path.join(PROJECT_ROOT, "ev_ai_agents", "carbon_agent", "datasets", "green_logistics.csv")
EMISSION_FACTORS_CSV = os.path.join(PROJECT_ROOT, "ev_ai_agents", "carbon_agent", "datasets", "supply_chain_emission_factors.csv")
FLEET_OPS_CSV = os.path.join(PROJECT_ROOT, "ev_ai_agents", "ev_fleet_electrification_agent", "datasets", "fleet_operations_clean.csv")


def map_vehicle_id(vid: str) -> str:
    """Returns the vehicle ID directly to query the cleaned dataset rows."""
    return vid.strip()

# ----------------------------------------------------------------------
# API Endpoints
# ----------------------------------------------------------------------

@app.get("/api/meta/vehicles")
def get_meta_vehicles():
    """Returns a list of available vehicles and basic details for selection dropdowns."""
    df = load_csv_safe(FLEET_OPS_CSV)
    if df is not None:
        return df[['vehicle_id', 'current_vehicle_make', 'current_vehicle_model', 'vehicle_type', 'depot_location']].head(30).to_dict(orient='records')
    return [
        {"vehicle_id": "VH_15592", "current_vehicle_make": "Ford", "current_vehicle_model": "Transit 350", "vehicle_type": "delivery_van", "depot_location": "Pune"},
        {"vehicle_id": "VH_10116", "current_vehicle_make": "Tata Motors", "current_vehicle_model": "Starbus Ultra", "vehicle_type": "passenger_bus", "depot_location": "Delhi"},
        {"vehicle_id": "VH_98912", "current_vehicle_make": "Freightliner", "current_vehicle_model": "Cascadia 126", "vehicle_type": "heavy_duty_truck", "depot_location": "Mumbai"},
        {"vehicle_id": "VH_01074", "current_vehicle_make": "Mercedes-Benz", "current_vehicle_model": "Actros 1845", "vehicle_type": "medium_duty_truck", "depot_location": "Nashik"},
        {"vehicle_id": "VH_37500", "current_vehicle_make": "Toyota", "current_vehicle_model": "Hilux GD-6", "vehicle_type": "pickup_truck", "depot_location": "Aurangabad"}
    ]

@app.get("/api/meta/evs")
def get_meta_evs():
    """Returns a list of EV battery tracking IDs for selection dropdowns."""
    df = load_csv_safe(APM_CSV)
    if df is not None:
        return sorted(df['EV_ID'].unique().tolist())[:30]
    return [f"EV-90{i:02d}" for i in range(10)]

@app.get("/api/meta/batches")
def get_meta_batches():
    """Returns a list of manufacturing batch IDs for selection dropdowns."""
    df = load_csv_safe(QMS_CSV)
    if df is not None:
        return sorted(df['Batch_ID'].unique().tolist())[:30]
    return [f"BTH-{i:04d}" for i in range(1, 11)]

# 1. Fleet Electrification Readiness & Procurement Intelligence Agent
@app.get("/api/agents/fleet_electrification")
def get_fleet_electrification(
    query: str = Query("Evaluate my delivery fleet for electrification and estimate annual savings.", description="User query"),
    vehicle_id: Optional[str] = Query(None, description="Target vehicle ID to focus analysis (optional)")
):
    target_id = map_vehicle_id(vehicle_id) if vehicle_id and vehicle_id.strip() else None
    logger.info(f"Fleet Electrification: target vehicle_id -> {target_id}")
    
    if run_fleet_agent:
        try:
            res = run_fleet_agent(query, target_id)
            if target_id:
                # Verify if the tool outputs actually succeeded or contained errors
                readiness_tool_data = res.get("tool_outputs", {}).get("readiness_score_tool", {})
                if "error" in readiness_tool_data:
                    logger.warning(f"Tool execution returned error: {readiness_tool_data['error']}")
                    res["recommendations"].insert(0, f"Tool Notice: The backend analysis for {vehicle_id} (mapped to {target_id}) encountered a tool warning: {readiness_tool_data['error']}. Check if dataset configurations are aligned.")
                
                # Map the response vehicle_id back to user expectation
                if "tool_outputs" in res:
                    for tk, t_out in res["tool_outputs"].items():
                        if isinstance(t_out, dict) and t_out.get("vehicle_id") == target_id:
                            t_out["vehicle_id"] = vehicle_id
            return res
        except Exception as e:
            logger.error(f"Error calling real fleet electrification agent: {e}")
            raise HTTPException(status_code=500, detail=f"Fleet agent execution failed: {str(e)}")
    
    raise HTTPException(status_code=503, detail="Fleet Electrification Agent module is not available.")

# 2. Maintenance Operations Optimiser Agent
@app.get("/api/agents/maintenance_operations")
def get_maintenance_operations(
    query: str = Query("Identify high-risk vehicles and prepare a plan.", description="User query")
):
    if run_maint_agent:
        try:
            res = run_maint_agent(query)
            return res
        except Exception as e:
            logger.error(f"Error calling real maintenance agent: {e}")
            raise HTTPException(status_code=500, detail=f"Maintenance agent execution failed: {str(e)}")
            
    raise HTTPException(status_code=503, detail="Maintenance Agent module is not available.")

from ev_ai_agents.ev_apm_agent.agent import apm_app
from ev_ai_agents.ev_qms_agent.agent import qms_app

# 3. EV Asset Performance Management (APM) Agent
@app.get("/api/agents/ev_apm")
def get_ev_apm(
    user_query: str = Query(None, description="Natural language query (primary path)"),
    ev_id: str = Query(None, description="Target EV ID (override for programmatic calls)"),
    avg_temp: float = Query(None),
    max_temp: float = Query(None),
    fc_ratio: float = Query(None),
    deep_cycles: int = Query(None),
    charge_dur: float = Query(None)
):
    try:
        # Natural language query is the primary path — goes through the agent's own planner
        if user_query and user_query.strip():
            res = apm_app.invoke({"user_query": user_query})
        else:
            # Advanced override: ID or raw telemetry
            input_data = {}
            if ev_id:
                input_data["ev_id"] = ev_id
            if avg_temp is not None:
                input_data["avg_temperature_c"] = avg_temp
                input_data["max_temperature_c"] = max_temp
                input_data["fast_charge_ratio_pct"] = fc_ratio
                input_data["deep_discharge_cycles"] = deep_cycles
                input_data["avg_charge_duration_hours"] = charge_dur
            res = apm_app.invoke(input_data)

        battery = res.get("battery_analysis", {})
        safety = res.get("safety_analysis", {})
        telemetry = res.get("telemetry_data", {})

        # Pull LLM reasoning output directly
        reasoning = res.get("reasoning_output") or {}
        llm_summary = reasoning.get("summary", "")
        llm_explanation = reasoning.get("explanation", "")
        llm_recommendations = reasoning.get("recommendations", [])
        llm_triggers = reasoning.get("maintenance_triggers", [])

        # Compose the AI response text shown in the summary box
        ai_text = llm_summary
        if llm_explanation:
            ai_text += "\n\n" + llm_explanation

        # Fallback if LLM didn't run
        if not ai_text.strip():
            soh = battery.get("state_of_health_percentage", battery.get("state_of_health_pct", 0))
            deg = battery.get("degradation_rate_per_month", 0)
            ai_text = (
                f"Fleet analysis: avg SoH {soh}%, avg degradation {deg}%/month. "
                f"Avg operating temp {safety.get('average_operating_temperature_celsius', 0)}°C, "
                f"{safety.get('thermal_runaway_warnings', 0)} thermal risk EVs identified."
            )

        return {
            "ev_id": ev_id or battery.get("ev_id", "QUERY_RESULT"),
            "battery_analysis": {
                "state_of_health_pct": battery.get("state_of_health_percentage", battery.get("state_of_health_pct", 0)),
                "degradation_rate_monthly_pct": battery.get("degradation_rate_per_month", battery.get("degradation_rate_monthly_pct", 0)),
                "remaining_useful_life_months": battery.get("remaining_useful_life_months", 0),
                "status": battery.get("status", "Unknown")
            },
            "safety_analysis": {
                "avg_temp_c": safety.get("average_operating_temperature_celsius", safety.get("avg_temp_c", 0)),
                "max_temp_c": safety.get("max_recorded_temperature_celsius", safety.get("max_temp_c", 0)),
                "cooling_status": safety.get("cooling_system_status", safety.get("cooling_status", "OK")),
                "thermal_runaway_warnings": safety.get("thermal_runaway_warnings", 0)
            },
            "telemetry_data": {
                "fast_charge_ratio_pct": telemetry.get("fast_charging_ratio_percentage", telemetry.get("fast_charge_ratio_pct", 0)),
                "deep_discharge_cycles": telemetry.get("deep_discharge_cycles_last_month", telemetry.get("deep_discharge_cycles", 0)),
                "avg_charge_duration_hours": telemetry.get("average_charge_duration_hours", telemetry.get("avg_charge_duration_hours", 0))
            },
            "maintenance_triggers": llm_triggers or res.get("maintenance_triggers", []),
            "recommendations": llm_recommendations or res.get("recommendations", []),
            "summary": ai_text
        }
    except Exception as e:
        logger.error(f"Error calling APM agent: {e}")
        raise HTTPException(status_code=500, detail=f"APM agent execution failed: {str(e)}")

# 4. EV Manufacturing Quality Intelligence (QMS) Agent
@app.get("/api/agents/ev_qms")
def get_ev_qms(
    user_query: str = Query(None, description="Natural language query (primary path)"),
    batch_id: str = Query(None, description="Manufacturing Batch ID"),
    elec_vol: float = Query(None),
    int_res: float = Query(None),
    cell_cap: float = Query(None),
    amb_temp: float = Query(None)
):
    try:
        # Natural language query is the primary path
        if user_query and user_query.strip():
            res = qms_app.invoke({"user_query": user_query})
        else:
            input_data = {}
            if batch_id:
                input_data["batch_id"] = batch_id
            if elec_vol is not None:
                input_data["electrolyte_volume_ml"] = elec_vol
                input_data["internal_resistance_mohm"] = int_res
                input_data["capacity_mah"] = cell_cap
                input_data["ambient_temp_c"] = amb_temp
            res = qms_app.invoke(input_data)

        # Pull LLM reasoning output directly
        reasoning = res.get("reasoning_output") or {}
        llm_summary = reasoning.get("summary", "")
        llm_explanation = reasoning.get("explanation", "")
        llm_recommendations = reasoning.get("recommendations", [])

        ai_text = llm_summary
        if llm_explanation:
            ai_text += "\n\n" + llm_explanation

        # Batch/factory metrics
        batch_metrics = res.get("batch_metrics") or {}
        # Fallback: read directly from tool_outputs if batch_metrics is empty
        if not batch_metrics:
            tool_outs = res.get("tool_outputs", {})
            insp = tool_outs.get("fetch_inspection_data", {})
            if insp and "error" not in insp:
                batch_metrics = {
                    "total_inspected": insp.get("total_inspected", 0),
                    "defect_rate_pct": insp.get("scrap_rate_pct", 0.0),
                    "avg_resistance_mohm": insp.get("avg_resistance_mOhm", 0.0),
                    "avg_capacity_mah": insp.get("avg_capacity_mAh", 0.0),
                    "avg_electrolyte_ml": insp.get("avg_electrolyte_volume_ml", 0.0),
                }

        # Quality distributions — pull from inspection_data or tool_outputs
        insp_data = res.get("inspection_data") or {}
        tool_outs2 = res.get("tool_outputs") or {}
        agg_stats = tool_outs2.get("aggregate_qms_statistics") or insp_data
        grades_dist = agg_stats.get("grades_distribution", {})
        defect_cats = agg_stats.get("defect_categories", {})
        line_stats = agg_stats.get("scrap_rate_by_line_pct", {})

        # Fallback summary if LLM didn't run
        if not ai_text.strip():
            total = batch_metrics.get("total_inspected", 0)
            rate = batch_metrics.get("defect_rate_pct", 0)
            ai_text = f"Factory analysis: {total} cells inspected, {rate}% scrap defect rate. Avg capacity {batch_metrics.get('avg_capacity_mah', 0)} mAh, avg resistance {batch_metrics.get('avg_resistance_mohm', 0)} mΩ."

        return {
            "batch_id": batch_id or "QUERY_RESULT",
            "cell_metrics": {
                "total_cells_inspected": batch_metrics.get("total_inspected", 0),
                "scrap_defect_rate_pct": batch_metrics.get("defect_rate_pct", 0.0),
                "average_internal_resistance_mOhm": batch_metrics.get("avg_resistance_mohm", 0.0),
                "average_cell_capacity_mAh": batch_metrics.get("avg_capacity_mah", 0.0),
                "average_electrolyte_volume_ml": batch_metrics.get("avg_electrolyte_ml", 0.0)
            },
            "quality_distributions": {
                "grades": grades_dist,
                "defect_categories": defect_cats,
                "scrap_rate_by_line_pct": line_stats
            },
            "quality_drift_analysis": res.get("process_drift", ""),
            "root_cause_analysis": res.get("root_cause", ""),
            "alerts": llm_recommendations or res.get("corrective_actions", []),
            "recommendations": llm_recommendations or res.get("recommendations", []),
            "summary": ai_text
        }
    except Exception as e:
        logger.error(f"Error calling QMS agent: {e}")
        raise HTTPException(status_code=500, detail=f"QMS agent execution failed: {str(e)}")


# 5. EV Supply Chain & Manufacturing Logistics Agent
@app.get("/api/agents/supply_chain")
def get_supply_chain(query: str = Query("Audit supplier SUP-001 and review ESG mineral risk.", description="User query")):
    from ev_ai_agents.ev_supply_chain_agent.agent import supply_chain_app
    
    # Invoke REAL agent!
    unified_report = "No report generated."
    tool_outs = {}
    try:
        res = supply_chain_app.invoke({"user_query": query})
        unified_report = res.get("reasoning_output", {}).get("unified_report", "No report generated.")
        tool_outs = res.get("tool_outputs", {})
    except Exception as e:
        logger.error(f"Supply Chain Agent failed: {e}")
        raise HTTPException(status_code=500, detail=f"Supply Chain agent execution failed: {str(e)}")
    
    supplier_info = tool_outs.get("get_supplier_profile", {})
    risk_info = tool_outs.get("calculate_supplier_risk_score", {})
    quality_info = tool_outs.get("assess_battery_quality", {})
    
    nodes = tool_outs.get("trace_material_batch", {}).get("traceability_nodes", [])
    
    return {
        "query": query,
        "supplier_details": supplier_info,
        "mineral_risk": risk_info,
        "battery_quality": quality_info,
        "unified_report": unified_report,
        "traceability_nodes": nodes,
        "risk_rating": risk_info.get('risk_level', 'Conceptual') if risk_info else 'Conceptual'
    }

# 6. Net Zero Progress & Carbon Intelligence Agent
@app.get("/api/agents/carbon_tracker")
def get_carbon_tracker(query: str = Query("What is our net zero target progress?", description="User query")):
    from ev_ai_agents.carbon_agent.agent import carbon_app
    
    # Invoke REAL agent!
    unified_report = "No report generated."
    ai_status = "Unknown"
    ai_reduction = 0.0
    tool_outs = {}
    
    try:
        res = carbon_app.invoke({"user_query": query})
        reasoning = res.get("reasoning_output", {})
        unified_report = reasoning.get("unified_report", "No report generated.")
        ai_status = reasoning.get("status", "Unknown")
        ai_reduction = reasoning.get("carbon_reduction_summary_pct", 0.0)
        tool_outs = res.get("tool_outputs", {})
    except Exception as e:
        logger.error(f"Carbon Agent failed: {e}")
        raise HTTPException(status_code=500, detail=f"Carbon agent execution failed: {str(e)}")

    co2_history = tool_outs.get("track_net_zero_progress", {}).get("co2_history", [])
    routes = tool_outs.get("track_scope_emissions", {}).get("top_routes", [])
    emission_factors = tool_outs.get("track_scope_emissions", {}).get("emission_factors", [])

    return {
        "status": ai_status,
        "co2_history": co2_history,
        "top_routes": routes,
        "emission_factors": emission_factors,
        "carbon_reduction_summary_pct": ai_reduction,
        "unified_report": unified_report
    }

# 7. Central Supervisor Orchestrator Chatbot Agent
@app.get("/api/agents/supervisor")
def get_supervisor(
    query: str = Query(..., description="User query")
):
    """The central EV Operations Supervisor Agent."""
    import logging
    logger = logging.getLogger("dashboard")
    logger.info(f"Supervisor Chatbot processing: '{query}'")
    
    if not os.getenv("GROQ_API_KEY"):
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set. Cannot run supervisor.")

    try:
        from orchestrator.supervisor import orchestrator_app
        logger.info("Executing online LangGraph orchestrator...")
        initial_state = {
            "user_query": query,
            "agent_responses": {},
            "next_agents": [],
            "final_synthesis": ""
        }
        res = orchestrator_app.invoke(initial_state)
        return {
            "status": "success",
            "mode": "online",
            "query": query,
            "response": res.get("final_synthesis", "No synthesis returned."),
            "next_agents": res.get("next_agents", []),
            "agent_responses": res.get("agent_responses", {})
        }
    except Exception as e:
        from fastapi import HTTPException
        logger.error(f"Error in LangGraph orchestrator: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Supervisor execution failed: {str(e)}")

# ----------------------------------------------------------------------
# Serving the Static Frontend
# ----------------------------------------------------------------------
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")

@app.get("/")
def get_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Frontend index.html not found.")

if os.path.exists(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("dashboard_server:app", host="127.0.0.1", port=8001, reload=True)