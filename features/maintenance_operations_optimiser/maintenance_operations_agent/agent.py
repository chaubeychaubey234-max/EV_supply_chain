import os
import sys
import re
import logging
from typing import Any, Dict, List

# Setup pathing
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MOO_DIR = os.path.dirname(_THIS_DIR)
_FEAT_DIR = os.path.dirname(_MOO_DIR)
_ROOT_DIR = os.path.dirname(_FEAT_DIR)

if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

# Import state
from features.maintenance_operations_optimiser.maintenance_operations_agent.state import AgentState

# Import tools
from features.maintenance_operations_optimiser.tools.maintenance_risk_analyzer import (
    analyze_maintenance_risk,
    _fetch_record_from_csv
)
from features.maintenance_operations_optimiser.tools.maintenance_schedule_optimizer import optimize_maintenance_schedule
from features.maintenance_operations_optimiser.tools.charging_availability_planner import plan_charging_availability
from features.maintenance_operations_optimiser.tools.utils import (
    load_maintenance_history,
    load_fleet_operations
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("maintenance_operations_agent")

def generate_llm_response(user_query: str, tool_outputs: dict) -> dict:
    """Placeholder function for LLM response generation.
    
    All future Groq integration should happen ONLY inside this function.
    No other code should depend on the LLM implementation.
    """
    logger.info("generate_llm_response: LLM not yet integrated - returning placeholder.")
    
    recommendations = []
    next_steps = []
    
    # Generate realistic placeholder recommendations/next steps based on tool outputs
    if "maintenance_risk_analyzer" in tool_outputs:
        risk_data = tool_outputs["maintenance_risk_analyzer"]
        if "error" not in risk_data:
            vid = risk_data.get("vehicle_id", "")
            score = risk_data.get("risk_score", 0)
            level = risk_data.get("risk_level", "")
            action = risk_data.get("recommended_action", "")
            recommendations.append(
                f"Vehicle {vid} is at {level} risk with a score of {score}. Action: {action}"
            )
            next_steps.append(f"Schedule diagnostic checks for {vid} to address dominant risk factors.")
            
    if "maintenance_schedule_optimizer" in tool_outputs:
        sched_data = tool_outputs["maintenance_schedule_optimizer"]
        if isinstance(sched_data, list) and len(sched_data) > 0:
            recommendations.append(f"Optimized schedule prepared for {len(sched_data)} vehicles.")
            first_slot = sched_data[0]
            next_steps.append(
                f"Notify {first_slot.get('workshop_name')} of booking for {first_slot.get('vehicle_id')} on {first_slot.get('scheduled_day')} at {first_slot.get('scheduled_time_slot')}."
            )
            
    if "charging_availability_planner" in tool_outputs:
        charge_data = tool_outputs["charging_availability_planner"]
        if "error" not in charge_data:
            vid = charge_data.get("vehicle_id", "")
            station = charge_data.get("recommended_station", "")
            time = charge_data.get("charging_time", "")
            recommendations.append(
                f"Assigned charging window for {vid} at {station} starting at {time}."
            )
            next_steps.append(f"Confirm charger availability at {station} for vehicle {vid}.")

    if not recommendations:
        recommendations = ["Review tool outputs for individual vehicle risk levels."]
    if not next_steps:
        next_steps = ["Run risk analysis on fleet to identify candidates for scheduled maintenance."]

    return {
        "status": "LLM integration pending",
        "summary": "Tools executed successfully.",
        "tool_outputs": tool_outputs,
        "recommendations": recommendations,
        "next_steps": next_steps
    }

def run_agent(user_query: str) -> dict:
    """Main orchestration entry point.
    
    1. Understand user intent and select tools.
    2. Execute selected tools.
    3. Call the placeholder LLM function.
    4. Return the combined response.
    """
    logger.info(f"run_agent: Processing query: '{user_query}'")
    
    state = AgentState(user_query=user_query)
    state.conversation_history.append({"role": "user", "content": user_query})
    
    # 1. Intent Routing & Tool Selection
    query_lower = user_query.lower()
    selected_tools = []
    
    if "high-risk" in query_lower or ("risk" in query_lower and "plan" in query_lower):
        selected_tools = ["maintenance_risk_analyzer", "maintenance_schedule_optimizer", "charging_availability_planner"]
    else:
        if any(kw in query_lower for kw in ["risk", "health", "priority", "critical", "analyze", "which vehicle", "need maintenance", "need"]):
            selected_tools.append("maintenance_risk_analyzer")
        if any(kw in query_lower for kw in ["schedule", "plan", "week", "calendar", "optimize", "optimizer"]):
            selected_tools.append("maintenance_schedule_optimizer")
        if any(kw in query_lower for kw in ["charger", "charging", "station"]):
            selected_tools.append("charging_availability_planner")
            
    if not selected_tools:
        # Default fallback
        selected_tools = ["maintenance_risk_analyzer"]
        
    state.selected_tools = selected_tools
    logger.info(f"run_agent: Selected tools: {selected_tools}")
    
    # 2. Extract inputs from query
    # Look for EV_XXXX or VH_XXXXX vehicle IDs
    match_ev = re.search(r'(EV_\d{4})', user_query, re.IGNORECASE)
    match_vh = re.search(r'(VH_\d{5})', user_query, re.IGNORECASE)
    
    vehicle_id = None
    if match_ev:
        vehicle_id = match_ev.group(1).upper()
    elif match_vh:
        vehicle_id = match_vh.group(1).upper()
        
    # Execute tools
    tool_outputs = {}
    success_count = 0
    
    for tool_name in selected_tools:
        try:
            if tool_name == "maintenance_risk_analyzer":
                # Requires vehicle_record dict
                active_id = vehicle_id if vehicle_id and vehicle_id.startswith("EV_") else "EV_2001"
                record = _fetch_record_from_csv(active_id)
                res = analyze_maintenance_risk.invoke({"vehicle_record": record})
                tool_outputs[tool_name] = res
                success_count += 1
                
            elif tool_name == "maintenance_schedule_optimizer":
                # Requires vehicle_ids (list) and date_range_days (int)
                if vehicle_id and vehicle_id.startswith("EV_"):
                    vehicle_ids = [vehicle_id]
                else:
                    df = load_maintenance_history()
                    vehicle_ids = df["vehicle_id"].tolist()[:10]
                    
                date_range_days = 5
                if "week" in query_lower:
                    date_range_days = 7
                elif "10 days" in query_lower:
                    date_range_days = 10
                    
                res = optimize_maintenance_schedule.invoke({
                    "vehicle_ids": vehicle_ids,
                    "date_range_days": date_range_days
                })
                tool_outputs[tool_name] = res
                success_count += 1
                
            elif tool_name == "charging_availability_planner":
                # Requires vehicle_id (str), depot_location (str), shift_end_time (str)
                active_id = vehicle_id if vehicle_id and vehicle_id.startswith("VH_") else "VH_15592"
                
                # Fetch depot details
                depot_location = "Delhi"
                shift_end_time = "18:00"
                try:
                    df_fleet = load_fleet_operations()
                    row = df_fleet[df_fleet["vehicle_id"].astype(str).str.strip() == active_id]
                    if not row.empty:
                        depot_location = str(row.iloc[0]["depot_location"])
                        shift_end_time = str(row.iloc[0]["shift_end_time"])
                except Exception:
                    pass
                    
                res = plan_charging_availability.invoke({
                    "vehicle_id": active_id,
                    "depot_location": depot_location,
                    "shift_end_time": shift_end_time
                })
                tool_outputs[tool_name] = res
                success_count += 1
                
        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}': {e}")
            tool_outputs[tool_name] = {"error": str(e)}
            
    state.tool_outputs = tool_outputs
    
    # Determine execution status
    if success_count == len(selected_tools):
        state.execution_status = "success"
    elif success_count > 0:
        state.execution_status = "partial"
    else:
        state.execution_status = "error"
        
    # 3. Call placeholder LLM response
    llm_res = generate_llm_response(user_query, tool_outputs)
    
    # 4. Assemble final response
    final_response = {
        "status": state.execution_status,
        "selected_tools": selected_tools,
        "tool_outputs": tool_outputs,
        "summary": llm_res.get("summary", "Tools executed successfully."),
        "recommendations": llm_res.get("recommendations", []),
        "next_steps": llm_res.get("next_steps", [])
    }
    
    state.final_response = final_response
    state.conversation_history.append({"role": "assistant", "content": final_response["summary"]})
    
    return final_response

if __name__ == "__main__":
    import json
    
    # Test queries
    test_queries = [
        "Which vehicles need maintenance this week?",
        "Generate an optimized maintenance schedule.",
        "Which charging stations should be used after maintenance?",
        "Identify high-risk vehicles and prepare a maintenance plan."
    ]
    
    for q in test_queries:
        print("\n" + "="*80)
        print(f"QUERY: '{q}'")
        print("="*80)
        resp = run_agent(q)
        print(json.dumps(resp, indent=2, default=str))