"""
agent.py — Orchestration layer for the Maintenance Operations Optimiser Agent.
"""

from __future__ import annotations

import os
import sys
import re
import logging
from typing import Any, Dict, List, Optional, Generic, TypeVar
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# Setup pathing
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MOO_DIR = os.path.dirname(_THIS_DIR)
_FEAT_DIR = os.path.dirname(_MOO_DIR)
_ROOT_DIR = os.path.dirname(_FEAT_DIR)

if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

# Import state
from ev_ai_agents.ev_maintenance_operations_agent.state import AgentState

# Import tools
from ev_ai_agents.ev_maintenance_operations_agent.tools.maintenance_risk_analyzer import (
    analyze_maintenance_risk,
    _fetch_record_from_csv
)
from ev_ai_agents.ev_maintenance_operations_agent.tools.maintenance_schedule_optimizer import optimize_maintenance_schedule
from ev_ai_agents.ev_maintenance_operations_agent.tools.charging_availability_planner import plan_charging_availability
from ev_ai_agents.ev_maintenance_operations_agent.tools.utils import (
    load_maintenance_history,
    load_fleet_operations
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("maintenance_operations_agent")

# Centralized Tool Registry
_TOOL_REGISTRY: dict[str, Any] = {
    "maintenance_risk_analyzer":       analyze_maintenance_risk,
    "maintenance_schedule_optimizer":  optimize_maintenance_schedule,
    "charging_availability_planner":   plan_charging_availability,
}

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Schemas for Structured LLM Outputs & Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class MaintenanceQueryPlan(BaseModel):
    query_type: str = Field(description="Query type: conceptual, statistical, asset, or hybrid")
    requires_dataset: bool = Field(description="True if dataset access is required")
    tools: List[str] = Field(description="List of tool keys to run. Choose from: maintenance_risk_analyzer, maintenance_schedule_optimizer, charging_availability_planner")
    requires_llm: bool = Field(description="True if LLM reasoning is required to synthesize the final answer")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0 representing plan confidence")

class MaintenanceReasoningOutput(BaseModel):
    summary: str = Field(description="Summary of the maintenance analysis.")
    reasoning: str = Field(description="Detailed logic explaining the risk, schedules, and capacity.")
    recommendations: List[str] = Field(description="List of actionable predictive maintenance recommendations.")
    risks: List[str] = Field(description="Key operational or maintenance risks identified.")
    next_steps: List[str] = Field(description="Concrete immediate next steps for operators.")

T = TypeVar('T')

class LLMResponse(BaseModel, Generic[T]):
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None

# ─────────────────────────────────────────────────────────────────────────────
# Isolated LLM Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_llm() -> ChatGroq:
    """Initialize the Groq model. Validates that GROQ_API_KEY exists."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or api_key.startswith("dummy"):
        raise ValueError("GROQ_API_KEY not found or invalid in environment.")
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

def generate_llm_response(prompt_messages: list, response_model: Any = None) -> LLMResponse:
    """Invokes the LLM and catches expected configuration and API infrastructure errors.
    Propagates programming exceptions normally.
    """
    try:
        llm = get_llm()
        if response_model:
            structured_llm = llm.with_structured_output(response_model)
            res = structured_llm.invoke(prompt_messages)
            return LLMResponse(success=True, data=res, error=None)
        else:
            res = llm.invoke(prompt_messages)
            return LLMResponse(success=True, data=res, error=None)
    except (ValueError, Exception) as exc:
        exc_class_name = type(exc).__name__
        is_infra = False
        
        # Determine if it's an expected infrastructure error
        if isinstance(exc, ValueError):
            is_infra = True
        else:
            mod_name = type(exc).__module__.lower()
            if "groq" in mod_name or "openai" in mod_name:
                is_infra = True
            elif "Connection" in exc_class_name or "Timeout" in exc_class_name or "RateLimit" in exc_class_name:
                is_infra = True
                
        if is_infra:
            logger.warning(f"LLM infrastructure error caught: {exc_class_name} - {str(exc)}")
            return LLMResponse(success=False, data=None, error="LLM_NOT_CONFIGURED")
        else:
            # Propagate normal developer/programming errors
            raise exc

# ─────────────────────────────────────────────────────────────────────────────
# LangGraph Nodes
# ─────────────────────────────────────────────────────────────────────────────

def planner_node(state: AgentState) -> dict:
    """Planner node: Requests an execution plan from the LLM and stores the LLMResponse."""
    user_query = state.user_query
    
    system_prompt = (
        "You are the Query Planner for an EV Fleet Maintenance Operations Optimiser system.\n"
        "Analyze the user's natural language query and decide on an execution plan.\n"
        "Available tool capabilities in the registry:\n"
        "- 'maintenance_risk_analyzer': Predict risk level, issue, and corrective action for a specific EV.\n"
        "- 'maintenance_schedule_optimizer': Generate and optimize maintenance dates/slots for fleet EVs.\n"
        "- 'charging_availability_planner': Schedule and confirm charging station slots after servicing.\n"
        "\n"
        "Classify the query into one of: 'conceptual', 'statistical', 'asset', or 'hybrid'.\n"
        "Determine which tools are required to fulfill the user's query.\n"
        "Confidence must be a float between 0.0 and 1.0."
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Generate a query plan for user query: '{user_query}'")
    ]
    
    response = generate_llm_response(messages, MaintenanceQueryPlan)
    
    updates: dict[str, Any] = {
        "planner_response": response
    }
    
    if response.success and response.data:
        plan: MaintenanceQueryPlan = response.data
        updates["detected_intent"] = plan.query_type
        updates["analysis_plan"] = plan.tools
        updates["confidence"] = plan.confidence
        updates["analysis_mode"] = plan.query_type
        
    return updates

def tool_executor_node(state: AgentState) -> dict:
    """Tool Executor node: Executes the planned tools if planning was successful."""
    planner_res: Optional[LLMResponse] = state.planner_response
    if planner_res is None or not planner_res.success:
        logger.info("Tool Executor: Skipping tool execution because planning was unavailable (success=False).")
        return {"tool_outputs": {}, "selected_tools": []}
        
    plan: Optional[MaintenanceQueryPlan] = planner_res.data
    tools_to_run = plan.tools if plan else []
    tool_outputs: dict[str, Any] = {}
    
    # Extract vehicle_id from query
    user_query = state.user_query or ""
    query_lower = user_query.lower()
    match_ev = re.search(r'(EV_\d{4})', user_query, re.IGNORECASE)
    match_vh = re.search(r'(VH_\d{5})', user_query, re.IGNORECASE)
    
    vehicle_id = None
    if match_ev:
        vehicle_id = match_ev.group(1).upper()
    elif match_vh:
        vehicle_id = match_vh.group(1).upper()
        
    for tool_name in tools_to_run:
        tool_callable = _TOOL_REGISTRY.get(tool_name)
        if not tool_callable:
            logger.error(f"Tool '{tool_name}' not found in registry.")
            tool_outputs[tool_name] = {"error": f"Tool '{tool_name}' not available in registry."}
            continue
            
        try:
            logger.info(f"Executing tool: {tool_name}")
            if tool_name == "maintenance_risk_analyzer":
                if state.vehicle_record:
                    record = state.vehicle_record
                else:
                    active_id = vehicle_id if vehicle_id and vehicle_id.startswith("EV_") else "EV_2001"
                    record = _fetch_record_from_csv(active_id)
                res = tool_callable.invoke({"vehicle_record": record})
                tool_outputs[tool_name] = res
                
            elif tool_name == "maintenance_schedule_optimizer":
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
                    
                res = tool_callable.invoke({
                    "vehicle_ids": vehicle_ids,
                    "date_range_days": date_range_days
                })
                tool_outputs[tool_name] = res
                
            elif tool_name == "charging_availability_planner":
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
                    
                res = tool_callable.invoke({
                    "vehicle_id": active_id,
                    "depot_location": depot_location,
                    "shift_end_time": shift_end_time
                })
                tool_outputs[tool_name] = res
                
        except Exception as e:
            logger.exception(f"Error executing tool '{tool_name}'")
            tool_outputs[tool_name] = {"error": f"Tool execution failed: {str(e)}"}
            
    return {"tool_outputs": tool_outputs, "selected_tools": tools_to_run}

def llm_reasoning_node(state: AgentState) -> dict:
    """LLM Reasoning node: Asks the LLM to interpret tool outputs or conceptually answer queries."""
    planner_res: Optional[LLMResponse] = state.planner_response
    if planner_res is None or not planner_res.success:
        logger.info("LLM Reasoning: Skipping reasoning because planner was unavailable (success=False).")
        return {
            "reasoner_response": LLMResponse(success=False, error="LLM_NOT_CONFIGURED")
        }
        
    user_query = state.user_query
    detected_intent = state.detected_intent
    tool_outputs = state.tool_outputs
    
    system_prompt = (
        "You are an expert AI advisor for EV fleet maintenance and charging infrastructure operations.\n"
        "Your task is to interpret tool outputs and provide recommendations, schedules, risks, and next steps.\n"
        "Strict Rule: You must NEVER perform calculations or invent numerical metrics. All numerical data must originate from tool outputs.\n"
        "Strict Rule: If tool outputs are missing or empty, state that the analysis is conceptual only."
    )
    
    user_prompt = (
        f"User Query: {user_query}\n"
        f"Intent/Mode: {detected_intent}\n\n"
        f"Tool Outputs:\n"
    )
    for t_name, t_out in tool_outputs.items():
        user_prompt += f"--- {t_name} ---\n{t_out}\n\n"
        
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = generate_llm_response(messages, MaintenanceReasoningOutput)
    
    updates: dict[str, Any] = {
        "reasoner_response": response
    }
    if response.success and response.data:
        updates["reasoning_output"] = response.data.model_dump()
        
    return updates

def response_builder_node(state: AgentState) -> dict:
    """Response Builder node: Converts the state into the central Supervisor response model."""
    planner_res: Optional[LLMResponse] = state.planner_response
    reasoner_res: Optional[LLMResponse] = state.reasoner_response
    
    # Check if LLM is unconfigured
    llm_configured = True
    error_msg = ""
    if planner_res and not planner_res.success and planner_res.error == "LLM_NOT_CONFIGURED":
        llm_configured = False
        error_msg = "LLM is not configured. Add the GROQ_API_KEY to enable AI reasoning."
    elif reasoner_res and not reasoner_res.success and reasoner_res.error == "LLM_NOT_CONFIGURED":
        llm_configured = False
        error_msg = "LLM is not configured. Add the GROQ_API_KEY to enable AI reasoning."
        
    if not llm_configured:
        final_response = {
            "status": "error",
            "selected_tools": [],
            "tool_outputs": {},
            "summary": error_msg,
            "recommendations": ["Add the GROQ_API_KEY to enable AI reasoning."],
            "next_steps": ["Configure GROQ_API_KEY in the environment."]
        }
        return {
            "final_response": final_response,
            "execution_status": "error"
        }
        
    # Success/Partial path: Merging LLM narrative and raw tool outputs
    reasoning = state.reasoning_output or {}
    final_response = {
        "status": "success",
        "selected_tools": state.selected_tools,
        "tool_outputs": state.tool_outputs,
        "summary": reasoning.get("summary", ""),
        "recommendations": reasoning.get("recommendations", []),
        "next_steps": reasoning.get("next_steps", [])
    }
    
    # Inspect tool outputs for any error keys
    failed = False
    for k, v in state.tool_outputs.items():
        if isinstance(v, dict) and "error" in v:
            failed = True
            break
            
    if failed:
        final_response["status"] = "partial"
        
    return {
        "final_response": final_response,
        "execution_status": final_response["status"]
    }

# ─────────────────────────────────────────────────────────────────────────────
# Standardized LangGraph Pipeline Setup
# ─────────────────────────────────────────────────────────────────────────────

workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_node)
workflow.add_node("tool_executor", tool_executor_node)
workflow.add_node("llm_reasoning", llm_reasoning_node)
workflow.add_node("response_builder", response_builder_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "tool_executor")
workflow.add_edge("tool_executor", "llm_reasoning")
workflow.add_edge("llm_reasoning", "response_builder")
workflow.add_edge("response_builder", END)

maintenance_app = workflow.compile()

# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_agent(user_query: str, vehicle_record: dict = None) -> dict[str, Any]:
    """Execute the Maintenance Operations Optimiser Agent using LangGraph.

    Args:
        user_query: Natural-language question from the user.
        vehicle_record: Optional dictionary payload representing a vehicle data record.

    Returns:
        Structured response dictionary matching Supervisor expectations.
    """
    logger.info("=" * 60)
    logger.info("Maintenance Agent (LangGraph Workflow) — START")
    logger.info("  Query     : %s", user_query)
    logger.info("=" * 60)
    
    # Validate inputs
    if not isinstance(user_query, str) or not user_query.strip():
        logger.error("Invalid user_query: must be a non-empty string.")
        return {
            "status":          "error",
            "selected_tools":  [],
            "tool_outputs":    {},
            "summary":         "Invalid query. Please provide a non-empty question.",
            "recommendations": [],
            "next_steps":      [],
        }
        
    # Invoke StateGraph workflow
    state = AgentState(
        user_query=user_query.strip(),
        vehicle_record=vehicle_record,
        execution_status="pending"
    )
    
    try:
        res = maintenance_app.invoke(state)
        if isinstance(res, dict):
            final_res = res.get("final_response", {})
        else:
            final_res = getattr(res, "final_response", {})
            
        logger.info("Agent completed with status: %s", final_res.get("status"))
        return final_res
    except Exception as e:
        logger.exception("Workflow failed during execution")
        return {
            "status":          "error",
            "selected_tools":  [],
            "tool_outputs":    {},
            "summary":         f"Internal Agent execution error: {str(e)}",
            "recommendations": ["Retry execution or inspect backend logs."],
            "next_steps":      []
        }

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