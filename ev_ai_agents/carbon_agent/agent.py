import os
import logging
from typing import List, Optional, Any, Dict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from .state import CarbonState
from ev_ai_agents.carbon_agent.tools import (
    calculate_emissions_reduction,
    track_scope_emissions,
    analyze_route_emissions,
    identify_high_impact_routes,
    recommend_electrification,
    generate_and_save_route_map,
    track_net_zero_progress
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("carbon_agent")

_TOOL_REGISTRY = {
    "calculate_emissions_reduction": calculate_emissions_reduction,
    "track_scope_emissions": track_scope_emissions,
    "analyze_route_emissions": analyze_route_emissions,
    "identify_high_impact_routes": identify_high_impact_routes,
    "recommend_electrification": recommend_electrification,
    "generate_and_save_route_map": generate_and_save_route_map,
    "track_net_zero_progress": track_net_zero_progress
}

class CarbonQueryPlan(BaseModel):
    query_type: str = Field(description="Query type: conceptual, statistical, asset, or hybrid")
    requires_dataset: bool = Field(description="True if dataset access is required")
    tools: List[str] = Field(description="List of tool keys to run.")
    requires_llm: bool = Field(description="True if LLM reasoning is required to synthesize the final answer")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")

class CarbonReasoningOutput(BaseModel):
    status: str = Field(description="Current Net Zero Status (e.g. 'On Track', 'At Risk')")
    carbon_reduction_summary_pct: float = Field(description="Verified GHG Reductions % compared to baseline")
    unified_report: str = Field(description="Summary report of logistics green routes and findings")

def generate_llm_response(prompt_messages: list, response_model: Any = None) -> Any:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or api_key.startswith("dummy"):
        raise ValueError("GROQ_API_KEY is missing or invalid. LLM execution cannot proceed.")
            
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
    if response_model:
        structured_llm = llm.with_structured_output(response_model)
        return structured_llm.invoke(prompt_messages)
    return llm.invoke(prompt_messages)

def planner_node(state: CarbonState) -> dict:
    user_query = state.get("user_query", "")
    
    system_prompt = (
        "You are the Query Planner for a Net Zero Carbon Intelligence system.\n"
        "Your task is to analyze the user's query and decide on an execution plan.\n"
        "Available tool capabilities to schedule in the plan:\n"
        "- 'calculate_emissions_reduction': Calculate emissions reduction for an EV compared to ICE.\n"
        "- 'track_scope_emissions': Track scope 1, 2, 3 emissions.\n"
        "- 'analyze_route_emissions': Analyze emissions for a specific route.\n"
        "- 'identify_high_impact_routes': Identify routes with highest emissions.\n"
        "- 'recommend_electrification': Recommend electrification for specific routes.\n"
        "- 'generate_and_save_route_map': Generate a route map visual.\n"
        "- 'track_net_zero_progress': Track overall progress towards net zero goals.\n"
        "\n"
        "Classify the query into one of: 'conceptual', 'statistical', 'asset', or 'hybrid'.\n"
        "Select the appropriate tools to answer the user's query."
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Generate a query plan for the user query: '{user_query}'")
    ]
    
    plan = generate_llm_response(messages, CarbonQueryPlan)
    
    return {
        "detected_intent": plan.query_type,
        "analysis_mode": f"{plan.query_type}_analysis",
        "analysis_plan": plan.tools,
        "confidence": plan.confidence,
        "tool_outputs": {}
    }

def tool_executor_node(state: CarbonState) -> dict:
    tools_to_run = state.get("analysis_plan", [])
    tool_outputs = {}
    
    for tool_name in tools_to_run:
        tool_callable = _TOOL_REGISTRY.get(tool_name)
        if not tool_callable:
            tool_outputs[tool_name] = {"error": f"Tool {tool_name} not found"}
            continue
            
        try:
            logger.info(f"Executing tool: {tool_name}")
            if hasattr(tool_callable, "args") and ("diesel_usage" in tool_callable.args or "ev_usage" in tool_callable.args):
                result = {"message": f"Tool {tool_name} skipped. Required usage parameters not provided. Answer conceptually based on user query description."}
            else:
                result = tool_callable.invoke({})
            tool_outputs[tool_name] = result
        except Exception as e:
            logger.exception(f"Error executing tool {tool_name}")
            tool_outputs[tool_name] = {"error": f"Tool execution failed: {str(e)}"}
            
    return {"tool_outputs": tool_outputs}
def llm_reasoning_node(state: CarbonState) -> dict:
    user_query = state.get("user_query", "No query provided.")
    detected_intent = state.get("detected_intent", "asset")
    tool_outputs = state.get("tool_outputs", {})
    
    system_prompt = (
        "You are an expert Net Zero Carbon Intelligence Agent.\n"
        "Your task is to interpret tool outputs and user queries to provide comprehensive carbon impact analysis and sustainability recommendations.\n"
    )
    
    user_prompt = f"User Query: {user_query}\nIntent: {detected_intent}\n\nTool Outputs:\n"
    for tool_name, output in tool_outputs.items():
        user_prompt += f"--- {tool_name} ---\n{output}\n\n"
        
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    reasoning_result = generate_llm_response(messages, CarbonReasoningOutput)
    return {"reasoning_output": reasoning_result.model_dump()}

def response_builder_node(state: CarbonState) -> dict:
    reasoning = state.get("reasoning_output", {})
    return {
        "status": reasoning.get("status", "Unknown"),
        "carbon_reduction_summary_pct": reasoning.get("carbon_reduction_summary_pct", 0.0),
        "unified_report": reasoning.get("unified_report", "No report generated."),
        "tool_outputs": state.get("tool_outputs", {})
    }

workflow = StateGraph(CarbonState)
workflow.add_node("planner", planner_node)
workflow.add_node("tool_executor", tool_executor_node)
workflow.add_node("llm_reasoning", llm_reasoning_node)
workflow.add_node("response_builder", response_builder_node)
workflow.set_entry_point("planner")
workflow.add_edge("planner", "tool_executor")
workflow.add_edge("tool_executor", "llm_reasoning")
workflow.add_edge("llm_reasoning", "response_builder")
workflow.add_edge("response_builder", END)

carbon_app = workflow.compile()

def get_agent_executor():
    # Shim to support legacy entrypoint
    class AppWrapper:
        def invoke(self, inputs):
            res = carbon_app.invoke({"user_query": inputs.get("input", "")}); res["output"] = res.get("unified_report", ""); return res # ({"user_query": inputs.get("input", "")})
    return AppWrapper()
