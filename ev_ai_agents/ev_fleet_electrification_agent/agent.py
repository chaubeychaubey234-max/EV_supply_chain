"""
agent.py — Orchestration layer for the Fleet Electrification Readiness Agent.
"""

from __future__ import annotations

import logging
import sys
import os
import re
from typing import Any, List, Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# Resolve project root so imports work
_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))                   # …/ev_fleet_electrification_agent/
_FEAT_DIR  = os.path.dirname(_AGENT_DIR)                                   # …/fleet_electrification_readiness/
_FEATS_DIR = os.path.dirname(_FEAT_DIR)                                    # …/features/
_ROOT_DIR  = os.path.dirname(_FEATS_DIR)                                   # project root
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

# Existing tool imports
from ev_ai_agents.ev_fleet_electrification_agent.tools.fleet_data_tool      import fetch_vehicle_data, analyze_fleet_csv
from ev_ai_agents.ev_fleet_electrification_agent.tools.readiness_score_tool  import calculate_readiness_score
from ev_ai_agents.ev_fleet_electrification_agent.tools.ev_matching_tool      import recommend_ev_replacement
from ev_ai_agents.ev_fleet_electrification_agent.tools.roi_tool              import calculate_roi
from ev_ai_agents.ev_fleet_electrification_agent.tools.route_analysis_tool   import analyze_vehicle_route
from ev_ai_agents.ev_fleet_electrification_agent.tools.procurement_tool      import recommend_procurement

from .state import AgentState

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Centralized Tool Registry
_TOOL_REGISTRY: dict[str, Any] = {
    "fleet_data_tool":       fetch_vehicle_data,
    "analyze_fleet_csv":     analyze_fleet_csv,
    "readiness_score_tool":  calculate_readiness_score,
    "ev_matching_tool":      recommend_ev_replacement,
    "roi_tool":              calculate_roi,
    "route_analysis_tool":   analyze_vehicle_route,
    "procurement_tool":      recommend_procurement,
}

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Schemas for Structured LLM Outputs & Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class FleetQueryPlan(BaseModel):
    query_type: str = Field(description="Query type: conceptual, statistical, asset, or hybrid")
    requires_dataset: bool = Field(description="True if dataset access is required")
    tools: List[str] = Field(description="List of tool keys to run. Choose from: fleet_data_tool, readiness_score_tool, ev_matching_tool, roi_tool, procurement_tool, route_analysis_tool, analyze_fleet_csv")
    requires_llm: bool = Field(description="True if LLM reasoning is required to synthesize the final answer")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0 representing plan confidence")

class FleetReasoningOutput(BaseModel):
    summary: str = Field(description="Summary of the fleet electrification analysis.")
    reasoning: str = Field(description="Detailed logic explaining the recommendations, ROI, and feasibility.")
    recommendations: List[str] = Field(description="List of actionable electrification recommendations.")
    risks: List[str] = Field(description="Key deployment or transition risks identified.")
    next_steps: List[str] = Field(description="Concrete immediate next steps for fleet managers.")

class LLMResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
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
            # Safely check module or package info without hard dependency failures
            mod_name = type(exc).__module__.lower()
            if "groq" in mod_name or "openai" in mod_name:
                is_infra = True
            elif "Connection" in exc_class_name or "Timeout" in exc_class_name or "RateLimit" in exc_class_name:
                is_infra = True
                
        if is_infra:
            log.warning(f"LLM infrastructure error caught: {exc_class_name} - {str(exc)}")
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
        "You are the Query Planner for a Fleet Electrification & Procurement Intelligence system.\n"
        "Analyze the user's natural language query and decide on an execution plan.\n"
        "Available tool capabilities in the registry:\n"
        "- 'fleet_data_tool': Retrieve specifications and basic data for a specific vehicle_id.\n"
        "- 'readiness_score_tool': Calculate an electrification readiness score (0-100) for a vehicle.\n"
        "- 'ev_matching_tool': Recommends EV replacement models based on existing vehicle specs.\n"
        "- 'roi_tool': Calculates financial return on investment and estimated payback years.\n"
        "- 'procurement_tool': Recommends purchase/lease strategy and schedule.\n"
        "- 'route_analysis_tool': Performs route and range safety analysis for daily vehicle routes.\n"
        "- 'analyze_fleet_csv': Batch/bulk csv upload processing tool.\n"
        "\n"
        "Classify the query into one of: 'conceptual', 'statistical', 'asset', or 'hybrid'.\n"
        "Determine which tools are required to fulfill the user's query.\n"
        "Confidence must be a float between 0.0 and 1.0."
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Generate a query plan for user query: '{user_query}'")
    ]
    
    response = generate_llm_response(messages, FleetQueryPlan)
    
    updates: dict[str, Any] = {
        "planner_response": response
    }
    
    if response.success and response.data:
        plan: FleetQueryPlan = response.data
        updates["detected_intent"] = plan.query_type
        updates["analysis_plan"] = plan.tools
        updates["confidence"] = plan.confidence
        updates["analysis_mode"] = plan.query_type
        
    return updates

def tool_executor_node(state: AgentState) -> dict:
    """Tool Executor node: Executes the planned tools if planning was successful."""
    planner_res: Optional[LLMResponse] = state.planner_response
    if planner_res is None or not planner_res.success:
        log.info("Tool Executor: Skipping tool execution because planning was unavailable (success=False).")
        return {"tool_outputs": {}, "selected_tools": []}
        
    plan: Optional[FleetQueryPlan] = planner_res.data
    tools_to_run = plan.tools if plan else []
    tool_outputs: dict[str, Any] = {}
    vehicle_id = state.vehicle_id or "VEH-002"
    
    for tool_name in tools_to_run:
        tool_callable = _TOOL_REGISTRY.get(tool_name)
        if not tool_callable:
            log.error(f"Tool '{tool_name}' not found in registry.")
            tool_outputs[tool_name] = {"error": f"Tool '{tool_name}' not available in registry."}
            continue
            
        try:
            log.info(f"Executing tool: {tool_name} for vehicle: {vehicle_id}")
            result = tool_callable.invoke({"vehicle_id": vehicle_id})
            tool_outputs[tool_name] = result
        except Exception as e:
            log.exception(f"Error executing tool {tool_name}")
            tool_outputs[tool_name] = {"error": f"Tool execution failed: {str(e)}"}
            
    return {"tool_outputs": tool_outputs, "selected_tools": tools_to_run}

def llm_reasoning_node(state: AgentState) -> dict:
    """LLM Reasoning node: Asks the LLM to interpret tool outputs or conceptually answer queries."""
    planner_res: Optional[LLMResponse] = state.planner_response
    if planner_res is None or not planner_res.success:
        log.info("LLM Reasoning: Skipping reasoning because planner was unavailable (success=False).")
        return {
            "reasoner_response": LLMResponse(success=False, error="LLM_NOT_CONFIGURED")
        }
        
    user_query = state.user_query
    detected_intent = state.detected_intent
    tool_outputs = state.tool_outputs
    
    system_prompt = (
        "You are an expert consultant in fleet electrification and EV procurement.\n"
        "Your job is to interpret tool outputs and explain them clearly, justify recommendations, and suggest next steps.\n"
        "Strict Rule: You must NEVER perform calculations or invent numerical values. All metrics, scores, and costs must come directly from tool outputs.\n"
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
    
    response = generate_llm_response(messages, FleetReasoningOutput)
    
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

fleet_app = workflow.compile()

# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_agent(user_query: str, vehicle_id: str = "VEH-002") -> dict[str, Any]:
    """Execute the Fleet Electrification Readiness Agent using LangGraph.

    Args:
        user_query: Natural-language question from the user.
        vehicle_id: Fleet vehicle identifier to evaluate (default ``"VEH-002"``).

    Returns:
        Structured response dictionary matching Supervisor expectations.
    """
    log.info("=" * 60)
    log.info("Fleet Electrification Agent (LangGraph Workflow) — START")
    log.info("  Query     : %s", user_query)
    log.info("  Vehicle   : %s", vehicle_id)
    log.info("=" * 60)
    
    # 1. Validate inputs
    if not isinstance(user_query, str) or not user_query.strip():
        log.error("Invalid user_query: must be a non-empty string.")
        return {
            "status":          "error",
            "selected_tools":  [],
            "tool_outputs":    {},
            "summary":         "Invalid query. Please provide a non-empty question.",
            "recommendations": [],
            "next_steps":      [],
        }

    if not isinstance(vehicle_id, str) or not vehicle_id.strip():
        log.error("Invalid vehicle_id: must be a non-empty string.")
        return {
            "status":          "error",
            "selected_tools":  [],
            "tool_outputs":    {},
            "summary":         "Invalid vehicle_id. Please provide a valid fleet identifier.",
            "recommendations": [],
            "next_steps":      [],
        }
        
    # 2. Invoke StateGraph workflow
    state = AgentState(
        user_query=user_query.strip(),
        vehicle_id=vehicle_id.strip(),
        execution_status="pending"
    )
    
    try:
        res = fleet_app.invoke(state)
        if isinstance(res, dict):
            final_res = res.get("final_response", {})
        else:
            final_res = getattr(res, "final_response", {})
            
        log.info("Agent completed with status: %s", final_res.get("status"))
        return final_res
    except Exception as e:
        log.exception("Workflow failed during execution")
        return {
            "status":          "error",
            "selected_tools":  [],
            "tool_outputs":    {},
            "summary":         f"Internal Agent execution error: {str(e)}",
            "recommendations": ["Retry execution or inspect backend logs."],
            "next_steps":      []
        }

# ─────────────────────────────────────────────────────────────────────────────
# Local test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    TEST_QUERY      = "Evaluate my delivery fleet for electrification and estimate annual savings."
    TEST_VEHICLE_ID = "VEH-002"

    print("=" * 60)
    print("Fleet Electrification Agent — Test Run")
    print("=" * 60)
    print(f"Query     : {TEST_QUERY}")
    print(f"Vehicle ID: {TEST_VEHICLE_ID}")
    print()

    response = run_agent(TEST_QUERY, vehicle_id=TEST_VEHICLE_ID)

    print()
    print("=" * 60)
    print("FINAL RESPONSE")
    print("=" * 60)
    print(f"Status         : {response['status']}")
    print(f"Selected Tools : {response['selected_tools']}")
    print()
    print(f"Summary:")
    print(f"  {response['summary']}")
    print()

    print("Recommendations:")
    for i, rec in enumerate(response["recommendations"], 1):
        print(f"  {i}. {rec}")
    print()

    print("Next Steps:")
    for i, ns in enumerate(response["next_steps"], 1):
        print(f"  {i}. {ns}")
    print()

    print("Tool Outputs (JSON):")
    print(json.dumps(response["tool_outputs"], indent=2, default=str))
