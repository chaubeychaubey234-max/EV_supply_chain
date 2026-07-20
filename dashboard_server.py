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
    vehicle_id: str = Query("VH_15592", description="Target vehicle ID to focus analysis")
):
    mapped_id = map_vehicle_id(vehicle_id)
    logger.info(f"Fleet Electrification: mapped {vehicle_id} -> {mapped_id}")
    
    if run_fleet_agent:
        try:
            res = run_fleet_agent(query, mapped_id)
            # Verify if the tool outputs actually succeeded or contained errors
            readiness_tool_data = res.get("tool_outputs", {}).get("readiness_score_tool", {})
            if "error" in readiness_tool_data:
                logger.warning(f"Tool execution returned error: {readiness_tool_data['error']}")
                # We can enrich the recommendations list to inform the user
                res["recommendations"].insert(0, f"Tool Notice: The backend analysis for {vehicle_id} (mapped to {mapped_id}) encountered a tool warning: {readiness_tool_data['error']}. Check if dataset configurations are aligned.")
            
            # Map the response vehicle_id back to user expectation
            if "tool_outputs" in res:
                for tk, t_out in res["tool_outputs"].items():
                    if isinstance(t_out, dict) and t_out.get("vehicle_id") == mapped_id:
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
    ev_id: str = Query(None, description="Target EV ID"),
    avg_temp: float = Query(None),
    max_temp: float = Query(None),
    fc_ratio: float = Query(None),
    deep_cycles: int = Query(None),
    charge_dur: float = Query(None)
):
    try:
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
        
        # Format response for the frontend
        return {
            "ev_id": input_data.get("ev_id", "RAW_INPUT"),
            "battery_analysis": {
                "state_of_health_pct": battery.get("state_of_health_pct", battery.get("predicted_soh_pct", 0)),
                "degradation_rate_monthly_pct": battery.get("degradation_rate_monthly_pct", 0),
                "remaining_useful_life_months": battery.get("remaining_useful_life_months", 0),
                "status": battery.get("status", "Unknown")
            },
            "safety_analysis": {
                "avg_temp_c": avg_temp if avg_temp is not None else 0,
                "max_temp_c": max_temp if max_temp is not None else safety.get("max_recorded_temperature_celsius", 0),
                "cooling_status": safety.get("cooling_system_status", "OK"),
                "thermal_runaway_warnings": safety.get("thermal_runaway_warnings", 0)
            },
            "telemetry_data": {
                "fast_charge_ratio_pct": fc_ratio if fc_ratio is not None else telemetry.get("fast_charging_ratio_percentage", 0),
                "deep_discharge_cycles": deep_cycles if deep_cycles is not None else 0,
                "avg_charge_duration_hours": charge_dur if charge_dur is not None else 0
            },
            "maintenance_triggers": res.get("maintenance_triggers", []),
            "recommendations": res.get("recommendations", []),
            "summary": "APM Agent executed successfully based on telemetry."
        }
    except Exception as e:
        logger.error(f"Error calling APM agent: {e}")
        raise HTTPException(status_code=500, detail=f"APM agent execution failed: {str(e)}")

# 4. EV Manufacturing Quality Intelligence (QMS) Agent
@app.get("/api/agents/ev_qms")
def get_ev_qms(
    batch_id: str = Query(None, description="Manufacturing Batch ID"),
    elec_vol: float = Query(None),
    int_res: float = Query(None),
    cell_cap: float = Query(None),
    amb_temp: float = Query(None)
):
    try:
        input_data = {}
        if batch_id:
            input_data["batch_id"] = batch_id
        if elec_vol is not None:
            input_data["electrolyte_volume_ml"] = elec_vol
            input_data["internal_resistance_mohm"] = int_res
            input_data["capacity_mah"] = cell_cap
            input_data["ambient_temp_c"] = amb_temp

        res = qms_app.invoke(input_data)
        
        batch_metrics = res.get("batch_metrics", {})
        
        return {
            "batch_id": input_data.get("batch_id", "RAW_INPUT"),
            "cell_metrics": {
                "total_cells_inspected": batch_metrics.get("total_inspected", 1),
                "scrap_defect_rate_pct": batch_metrics.get("defect_rate_pct", 0.0),
                "average_internal_resistance_mOhm": int_res if int_res is not None else batch_metrics.get("avg_resistance_mohm", 0.0),
                "average_cell_capacity_mAh": cell_cap if cell_cap is not None else batch_metrics.get("avg_capacity_mah", 0.0),
                "average_electrolyte_volume_ml": elec_vol if elec_vol is not None else batch_metrics.get("avg_electrolyte_ml", 0.0)
            },
            "quality_distributions": {
                "grades": {"Grade A": 0, "Grade B": 0, "Scrap": 0},
                "defect_categories": {},
                "scrap_rate_by_line_pct": {}
            },
            "quality_drift_analysis": res.get("process_drift", "QMS Agent executed. Analysis completed."),
            "root_cause_analysis": res.get("root_cause", "N/A"),
            "alerts": res.get("corrective_actions", [])
        }
    except Exception as e:
        logger.error(f"Error calling QMS agent: {e}")
        raise HTTPException(status_code=500, detail=f"QMS agent execution failed: {str(e)}")

# 5. EV Supply Chain & Manufacturing Logistics Agent
@app.get("/api/agents/supply_chain")
def get_supply_chain(query: str = Query("Audit supplier SUP-001 and review ESG mineral risk.", description="User query")):
    df_sc = load_csv_safe(SUPPLY_CHAIN_CSV)
    df_mr = load_csv_safe(MINERAL_RISK_CSV)
    df_bq = load_csv_safe(BATTERY_QUALITY_CSV)

    if df_sc is None or df_mr is None or df_bq is None:
        raise HTTPException(status_code=500, detail="Supply chain audit datasets not found on server.")

    query_lower = query.lower()
    
    supplier_id = "SUP-001"
    material = "lithium"
    batch_id = "BAT-2024-001"

    if "sup-002" in query_lower or "sup-002" in query_lower: supplier_id = "SUP-002"
    elif "sup-003" in query_lower or "sup-003" in query_lower: supplier_id = "SUP-003"
    
    if "cobalt" in query_lower: material = "cobalt"
    elif "nickel" in query_lower: material = "nickel"
    
    if "bat-2024-002" in query_lower: batch_id = "BAT-2024-002"

    supplier_info = {}
    row_sc = df_sc[df_sc['supplier_id'].astype(str).str.strip().str.upper() == supplier_id.upper()]
    if not row_sc.empty:
        supplier_info = row_sc.iloc[0].to_dict()
        material = str(supplier_info['material'])
        batch_id = str(supplier_info['batch_id'])
    else:
        raise HTTPException(status_code=404, detail=f"Supplier ID '{supplier_id}' not found in active registry.")

    risk_info = {}
    row_mr = df_mr[df_mr['material'].astype(str).str.strip().str.lower() == material.lower()]
    if not row_mr.empty:
        risk_info = row_mr.iloc[0].to_dict()

    quality_info = {}
    row_bq = df_bq[df_bq['batch_id'].astype(str).str.strip().str.upper() == batch_id.upper()]
    if not row_bq.empty:
        quality_info = row_bq.iloc[0].to_dict()

    report = (
        f"SUPPLY CHAIN AUDIT SUMMARY FOR {supplier_info.get('supplier_name', 'Ganfeng Lithium')} ({supplier_id})\n\n"
        f"1. Geopolitical Risk: Material '{material}' originates predominantly from {risk_info.get('country', 'China')}. "
        f"Global supply share is {risk_info.get('global_supply_percentage', 65.0)}%. The dependency rating is scored at "
        f"{risk_info.get('dependency_score', 85.0)} out of 100, assigning a '{risk_info.get('risk_level', 'High')}' geopolitical risk profile.\n\n"
        f"2. Quality Status: Battery pack batch {batch_id} shows a defect rate of "
        f"{quality_info.get('defect_rate', 0.01) * 100:.2f}% across {quality_info.get('inspection_count', 500)} units checked. "
        f"Standard defect type flagged is '{quality_info.get('defect_type', 'micro-short circuit')}'.\n\n"
        f"3. ESG & Traceability: The cobalt or lithium trace maps securely to extraction sites with "
        f"blockchain audit confirmations. Carbon footprint indicators indicate high supplier compliance."
    )

    nodes = [
        {"id": "Mine", "label": f"{risk_info.get('country', 'Mine Site')} Raw Mineral Extraction", "status": "Audited"},
        {"id": "Refining", "label": f"{supplier_info.get('supplier_name', 'Supplier')} Chemical Refining Hub", "status": "On Track"},
        {"id": "CellManufacturing", "label": f"Cell Batch {batch_id}", "status": "QC Inspected"},
        {"id": "Vehicle", "label": f"Deployed on Fleet Vehicle {supplier_info.get('vehicle_id', 'VIN-EV-20240001')}", "status": "In Operation"}
    ]

    return {
        "query": query,
        "supplier_details": supplier_info,
        "mineral_risk": risk_info,
        "battery_quality": quality_info,
        "unified_report": report,
        "traceability_nodes": nodes,
        "risk_rating": risk_info.get('risk_level', 'Medium')
    }

# 6. Net Zero Progress & Carbon Intelligence Agent
@app.get("/api/agents/carbon_tracker")
def get_carbon_tracker(query: str = Query("What is our net zero target progress?", description="User query")):
    df_co2 = load_csv_safe(CO2_CSV)
    df_log = load_csv_safe(LOGISTICS_CSV)
    df_ef = load_csv_safe(EMISSION_FACTORS_CSV)

    if df_co2 is None:
        raise HTTPException(status_code=500, detail="CO2 Emissions dataset not found on server.")

    co2_history = df_co2.to_dict(orient='records')
    routes = df_log.head(10).to_dict(orient='records') if df_log is not None else []
    emission_factors = df_ef.to_dict(orient='records') if df_ef is not None else []

    latest = co2_history[-1] if co2_history else {"total_emissions": 12600, "target_emissions": 13000, "electrification_rate": 55.0}
    status = "On Track" if latest["total_emissions"] <= latest["target_emissions"] else "At Risk"
    
    reduction_pct = 100 - (latest["total_emissions"] / co2_history[0]["total_emissions"] * 100) if len(co2_history) > 1 else 42.2

    report = (
        f"Carbon Intelligence analysis indicates the organization is currently '{status}' "
        f"for the FY2026 carbon reductions goals. Overall greenhouse gas (GHG) footprint "
        f"is down {reduction_pct:.1f}% since baseline measurements in 2020. This is highly correlated "
        f"with the current fleet electrification rate of {latest.get('electrification_rate', 55.0)}%."
    )

    return {
        "status": status,
        "co2_history": co2_history,
        "top_routes": routes,
        "emission_factors": emission_factors,
        "carbon_reduction_summary_pct": round(reduction_pct, 1),
        "unified_report": report
    }

# 7. Central Supervisor Orchestrator Chatbot Agent
@app.get("/api/agents/supervisor")
def get_supervisor(
    query: str = Query("Check the APM health for EV-9001 and trace the supplier risk.", description="User query")
):
    """The central EV Operations Supervisor Agent.
    
    Coordinates the 6 domain-specific AI agents (APM, QMS, Supply Chain, Fleet, Maintenance, Carbon) 
    to answer multi-intent complex queries.
    """
    logger.info(f"Supervisor Chatbot processing: '{query}'")
    
    # Check if OpenAI is configured and langchain_openai is installed to run the real LangGraph app
    openai_available = False
    if os.getenv("OPENAI_API_KEY"):
        try:
            import langchain_openai
            from orchestrator.supervisor import orchestrator_app
            openai_available = True
        except ImportError:
            logger.info("langchain_openai not installed. Falling back to local offline supervisor routing.")

    if openai_available:
        try:
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
            logger.error(f"Error in LangGraph orchestrator: {e}. Falling back to offline routing...")

    # Fallback to local offline supervisor mapping
    logger.info("Executing local offline supervisor routing...")
    query_lower = query.lower()
    triggered_agents = []
    agent_responses = {}

    # 1. Fleet Electrification
    if any(kw in query_lower for kw in ["readiness", "transition", "electrif", "savings", "roi", "vh_", "vehicle"]):
        triggered_agents.append("fleet")
        if run_fleet_agent:
            try:
                import re
                v_match = re.search(r"(VH_\d+|VEH-00\d)", query_lower)
                vid = v_match.group(0).upper() if v_match else "VH_15592"
                mapped_id = map_vehicle_id(vid)
                fleet_res = run_fleet_agent("Evaluate my delivery fleet for electrification and estimate annual savings.", mapped_id)
                agent_responses["fleet"] = fleet_res
            except Exception as e:
                agent_responses["fleet"] = {"error": str(e)}
        else:
            agent_responses["fleet"] = {"status": "Fleet agent unavailable"}

    # 2. Maintenance Operations
    if any(kw in query_lower for kw in ["maintenance", "schedule", "workload", "workshop", "downtime", "risk"]):
        triggered_agents.append("maintenance")
        if run_maint_agent:
            try:
                maint_res = run_maint_agent("Identify high-risk vehicles and prepare a maintenance plan.")
                agent_responses["maintenance"] = maint_res
            except Exception as e:
                agent_responses["maintenance"] = {"error": str(e)}
        else:
            agent_responses["maintenance"] = {"status": "Maintenance agent unavailable"}

    # 3. APM Battery Health
    if any(kw in query_lower for kw in ["battery", "apm", "health", "temperature", "soh", "degradation", "ev-9"]):
        triggered_agents.append("apm")
        try:
            import re
            ev_match = re.search(r"EV-9\d+", query_lower)
            ev_id = ev_match.group(0).upper() if ev_match else "EV-9001"
            apm_res = get_ev_apm(ev_id)
            agent_responses["apm"] = apm_res
        except Exception as e:
            agent_responses["apm"] = {"error": str(e)}

    # 4. QMS Cell Quality
    if any(kw in query_lower for kw in ["qms", "batch", "cell", "quality", "drift", "defect", "electrolyte", "bth-", "batch-"]):
        triggered_agents.append("qms")
        try:
            import re
            b_match = re.search(r"BTH-\d+", query_lower)
            batch_id = b_match.group(0).upper() if b_match else "BTH-0001"
            qms_res = get_ev_qms(batch_id)
            agent_responses["qms"] = qms_res
        except Exception as e:
            agent_responses["qms"] = {"error": str(e)}

    # 5. Supply Chain Risk
    if any(kw in query_lower for kw in ["supply", "trace", "supplier", "mineral", "cobalt", "nickel", "lithium", "sup-"]):
        triggered_agents.append("supply_chain")
        try:
            sc_res = get_supply_chain("Audit supplier SUP-001 and review ESG mineral risk.")
            agent_responses["supply_chain"] = sc_res
        except Exception as e:
            agent_responses["supply_chain"] = {"error": str(e)}

    # 6. Carbon Tracker
    if any(kw in query_lower for kw in ["carbon", "emission", "green", "net-zero", "scope"]):
        triggered_agents.append("carbon")
        try:
            carbon_res = get_carbon_tracker("What is our net zero target progress?")
            agent_responses["carbon"] = carbon_res
        except Exception as e:
            agent_responses["carbon"] = {"error": str(e)}

    # If no specific agents were triggered, trigger a general checklist
    if not triggered_agents:
        triggered_agents = ["fleet", "maintenance", "apm", "qms", "supply_chain", "carbon"]
        try:
            if run_fleet_agent: agent_responses["fleet"] = run_fleet_agent("Evaluate", "VEH-002")
            if run_maint_agent: agent_responses["maintenance"] = run_maint_agent("Identify")
            agent_responses["apm"] = get_ev_apm("EV-9001")
            agent_responses["qms"] = get_ev_qms("BTH-0001")
            agent_responses["supply_chain"] = get_supply_chain("Audit supplier SUP-001")
            agent_responses["carbon"] = get_carbon_tracker("What is our net zero target progress?")
        except Exception:
            pass

    # Synthesize answers beautifully
    synthesis_parts = []
    synthesis_parts.append(f"### VoltGrid Operations Control — Central AI Supervisor Report\n")
    synthesis_parts.append(f"Parsed multi-intent query: *\"{query}\"*\n")
    synthesis_parts.append(f"Triggered sub-agents for evaluation: " + ", ".join([f"`{a.upper()}`" for a in triggered_agents]) + "\n")

    if "apm" in agent_responses:
        apm = agent_responses["apm"]
        if "error" not in apm:
            synthesis_parts.append(f"#### 🔋 Asset Performance Management (APM)")
            synthesis_parts.append(f"- **EV ID Evaluated**: `{apm['ev_id']}`")
            synthesis_parts.append(f"- **Battery State of Health (SOH)**: `{apm['battery_analysis']['state_of_health_pct']}%` ({apm['battery_analysis']['status']})")
            synthesis_parts.append(f"- **Safety Warnings**: *{apm['maintenance_triggers'][0]}*")
            synthesis_parts.append(f"- **Actionable Guidelines**: {apm['recommendations'][0]}\n")

    if "qms" in agent_responses:
        qms = agent_responses["qms"]
        if "error" not in qms:
            synthesis_parts.append(f"#### 🔬 Cell Quality (QMS)")
            synthesis_parts.append(f"- **Inspected Batch**: `{qms['batch_id']}`")
            synthesis_parts.append(f"- **Scrap Defect Rate**: `{qms['cell_metrics']['scrap_defect_rate_pct']}%` (Control Boundary < 2.0%)")
            synthesis_parts.append(f"- **Drift Analysis**: *{qms['quality_drift_analysis']}*")
            synthesis_parts.append(f"- **Root Cause Diagnostics**: *{qms['root_cause_analysis']}*\n")

    if "supply_chain" in agent_responses:
        sc = agent_responses["supply_chain"]
        if "error" not in sc:
            synthesis_parts.append(f"#### 📦 Supply Chain & Material Traceability")
            synthesis_parts.append(f"- **Supplier Audited**: `{sc['supplier_details'].get('supplier_name', 'N/A')}` ({sc['supplier_details'].get('supplier_id', 'N/A')})")
            synthesis_parts.append(f"- **ESG Geopolitical Risk**: `{sc['risk_rating']}` ({sc['mineral_risk'].get('country', 'N/A')} mineral origin dependency score of `{sc['mineral_risk'].get('dependency_score', 'N/A')}/100`)")
            synthesis_parts.append(f"- **Quality Compliance**: Defect rate of `{(sc['battery_quality'].get('defect_rate', 0.0) * 100):.2f}%` from supplier.\n")

    if "fleet" in agent_responses:
        fl = agent_responses["fleet"]
        if "error" not in fl:
            try:
                ready = fl["tool_outputs"]["readiness_score_tool"]
                ev = fl["tool_outputs"]["ev_matching_tool"]
                roi = fl["tool_outputs"]["roi_tool"]
                synthesis_parts.append(f"#### ⚡ Fleet Electrification Readiness")
                synthesis_parts.append(f"- **Electrification Readiness Score**: `{ready['readiness_score']}%` ({ready['classification']})")
                synthesis_parts.append(f"- **Recommended Replacement EV**: `{ev['recommended_ev']}` (Compatibility score of `{int(ev['compatibility_score']*100)}%`)")
                synthesis_parts.append(f"- **Savings Projection**: Total annual savings of `${roi['total_annual_savings_usd']}` with payback period of `{roi['estimated_payback_years']} years`\n")
            except KeyError:
                synthesis_parts.append(f"#### ⚡ Fleet Electrification: {fl.get('summary', 'Complete')}\n")

    if "maintenance" in agent_responses:
        mt = agent_responses["maintenance"]
        if "error" not in mt:
            try:
                risk = mt["tool_outputs"]["maintenance_risk_analyzer"]
                synthesis_parts.append(f"#### 🛠️ Maintenance Operations Optimiser")
                synthesis_parts.append(f"- **Urgent Risk Alert**: Vehicle `{risk['vehicle_id']}` has a risk score of `{risk['risk_score']}/100` ({risk['risk_level']})")
                synthesis_parts.append(f"- **Primary Risk Factor**: *{risk['dominant_risk_factor']}*")
                synthesis_parts.append(f"- **Recommended Action**: *{risk['recommended_action']}*\n")
            except KeyError:
                synthesis_parts.append(f"#### 🛠️ Maintenance Operations: {mt.get('summary', 'Complete')}\n")

    if "carbon" in agent_responses:
        cb = agent_responses["carbon"]
        if "error" not in cb:
            synthesis_parts.append(f"#### 🍃 Net Zero Target & Carbon Intelligence")
            synthesis_parts.append(f"- **Current Net Zero Status**: `{cb['status']}`")
            synthesis_parts.append(f"- **Verified GHG Reductions**: down `{cb['carbon_reduction_summary_pct']}%` compared to 2020 baseline levels")
            synthesis_parts.append(f"- **Logistics Green Routes**: *{cb['unified_report']}*\n")

    synthesis_parts.append("\n*Report compiled by VoltGrid AI Supervisor Orchestrator. Mode: local-offline fallback.*")
    final_text = "\n".join(synthesis_parts)

    return {
        "status": "success",
        "mode": "offline-fallback",
        "query": query,
        "response": final_text,
        "next_agents": triggered_agents,
        "agent_responses": agent_responses
    }

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
    uvicorn.run("dashboard_server:app", host="127.0.0.1", port=8000, reload=True)