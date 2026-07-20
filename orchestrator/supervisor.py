import os
import json
import re
import time
import logging
from typing import Literal, List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.constants import Send
from pydantic import BaseModel, Field

from orchestrator.state import OrchestratorState

# Import sub-agents
from ev_ai_agents.ev_apm_agent.agent import apm_app
from ev_ai_agents.ev_qms_agent.agent import qms_app
from ev_ai_agents.ev_supply_chain_agent.agent import supply_chain_app
from ev_ai_agents.ev_fleet_electrification_agent.agent import run_agent as run_fleet_agent
from ev_ai_agents.ev_maintenance_operations_agent.agent import run_agent as run_maintenance_agent
from ev_ai_agents.carbon_agent.agent import get_agent_executor as get_carbon_agent

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("supervisor")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# --- 1. Planner Model ---
class ExecutionPlan(BaseModel):
    agents: List[Literal["apm", "qms", "supply_chain", "fleet", "maintenance", "carbon"]] = Field(
        description="List of required agent keys to run based on user query intent."
    )
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(description="Reasoning for selecting these specific agents and skipping others")

class SynthesisOutput(BaseModel):
    final_answer: str = Field(description="The final comprehensive answer synthesizing all agent data.")


# --- Transient Error Retry Helper ---
def run_with_transient_retry(callable_func, *args, **kwargs):
    """Executes a callable with exactly 1 retry on transient infrastructure errors (HTTP 429, 503, Timeout, ConnectionError)."""
    transient_keywords = ("429", "503", "timeout", "connectionerror", "rate limit")
    try:
        return callable_func(*args, **kwargs)
    except Exception as e:
        err_msg = str(e).lower()
        if any(keyword in err_msg for keyword in transient_keywords):
            logger.warning(f"Transient infrastructure error caught: {e}. Retrying once in 3 seconds...")
            time.sleep(3.0)
            return callable_func(*args, **kwargs)
        raise e


# --- 2. Central Agent Registry ---
def call_apm_agent(state: OrchestratorState) -> dict:
    return apm_app.invoke({"user_query": state["user_query"]})

def call_qms_agent(state: OrchestratorState) -> dict:
    return qms_app.invoke({"user_query": state["user_query"]})

def call_supply_chain_agent(state: OrchestratorState) -> dict:
    return supply_chain_app.invoke({"user_query": state["user_query"]})

def call_fleet_agent(state: OrchestratorState) -> dict:
    return run_fleet_agent(state["user_query"])

def call_maintenance_agent(state: OrchestratorState) -> dict:
    return run_maintenance_agent(state["user_query"])

def call_carbon_agent(state: OrchestratorState) -> dict:
    carbon_agent = get_carbon_agent()
    return carbon_agent.invoke({"input": state["user_query"]})


_AGENT_REGISTRY = {
    "apm": call_apm_agent,
    "fleet": call_fleet_agent,
    "qms": call_qms_agent,
    "maintenance": call_maintenance_agent,
    "carbon": call_carbon_agent,
    "supply_chain": call_supply_chain_agent,
}

ALL_AGENTS = ["apm", "fleet", "qms", "supply_chain", "maintenance", "carbon"]


# --- Wrapper Nodes for Error Isolation & Timing ---
def safe_execute_agent(agent_key: str, state: OrchestratorState) -> dict:
    start_time = time.perf_counter()
    logger.info(f"Launching {agent_key.upper()}...")
    
    agent_func = _AGENT_REGISTRY.get(agent_key)
    if not agent_func:
        return {"status": "failed", "error": f"Agent key '{agent_key}' not found in registry"}
        
    try:
        res = run_with_transient_retry(agent_func, state)
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(f"{agent_key.upper()} Success ({elapsed_ms} ms)")
        return {
            "status": "success",
            "data": res,
            "execution_time_ms": elapsed_ms
        }
    except Exception as e:
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.error(f"{agent_key.upper()} Failed after {elapsed_ms} ms: {str(e)}")
        return {
            "status": "failed",
            "error": str(e),
            "execution_time_ms": elapsed_ms
        }


# Adapter LangGraph Node Handlers
def node_apm(state: OrchestratorState) -> dict:
    res = safe_execute_agent("apm", state)
    return {"agent_responses": {"apm": res.get("data", res)}}

def node_qms(state: OrchestratorState) -> dict:
    res = safe_execute_agent("qms", state)
    return {"agent_responses": {"qms": res.get("data", res)}}

def node_supply_chain(state: OrchestratorState) -> dict:
    res = safe_execute_agent("supply_chain", state)
    return {"agent_responses": {"supply_chain": res.get("data", res)}}

def node_fleet(state: OrchestratorState) -> dict:
    res = safe_execute_agent("fleet", state)
    return {"agent_responses": {"fleet": res.get("data", res)}}

def node_maintenance(state: OrchestratorState) -> dict:
    res = safe_execute_agent("maintenance", state)
    return {"agent_responses": {"maintenance": res.get("data", res)}}

def node_carbon(state: OrchestratorState) -> dict:
    res = safe_execute_agent("carbon", state)
    return {"agent_responses": {"carbon": res.get("data", res)}}


def node_apm_then_maintenance(state: OrchestratorState) -> dict:
    """Dependency execution node: runs APM first, then conditionally triggers Maintenance if SoH < 80% or status != Healthy."""
    apm_res = safe_execute_agent("apm", state)
    apm_data = apm_res.get("data", apm_res)
    
    battery = apm_data.get("battery_analysis", {}) if isinstance(apm_data, dict) else {}
    soh = battery.get("state_of_health_percentage", 100.0)
    status = battery.get("status", "Healthy")
    
    responses = {"apm": apm_data}
    
    if soh < 80.0 or status != "Healthy":
        logger.info(f"APM flagged battery issue (SoH={soh}%, status='{status}'). Launching Maintenance...")
        maint_res = safe_execute_agent("maintenance", state)
        responses["maintenance"] = maint_res.get("data", maint_res)
    else:
        logger.info(f"Maintenance Skipped: Battery healthy (SOH={soh}% >= 80%, status='{status}')")
        responses["maintenance"] = {
            "status": "skipped",
            "reason": f"Battery healthy (SOH={soh}% >= 80%, status='{status}')"
        }
        
    return {"agent_responses": responses}


# --- Stage 1: Planner Node ---
def planner_node(state: OrchestratorState) -> OrchestratorState:
    logger.info("==========================================")
    logger.info("Supervisor Started")
    logger.info("Planner Running...")
    start_time = time.perf_counter()
    
    system_prompt = (
        "You are the central Execution Planner for an EV Supply Chain Orchestrator.\n"
        "Your goal is to parse the user query and determine ONLY the relevant expert agent domains required to answer the user query.\n\n"
        "Available Agent Domain Keys:\n"
        "- 'apm': Battery degradation, state of health (SoH), RUL, thermal events, charging health.\n"
        "- 'fleet': Fleet electrification readiness, route ROI, vehicle replacement matching, procurement priority.\n"
        "- 'qms': Manufacturing defects, quality drift, batch inspection, scrap rate, cell specifications.\n"
        "- 'supply_chain': Supplier risk, geopolitical exposure, material traceability, ESG mineral compliance.\n"
        "- 'maintenance': Preventive maintenance scheduling, workshop capacity, charging availability.\n"
        "- 'carbon': Scope 3 GHG emissions, route carbon impact, net zero progress.\n\n"
        "RULES:\n"
        "1. Select ONLY the agents required for the query intent. Do NOT run all agents by default.\n"
        "2. If query asks about battery degradation -> select ONLY ['apm'].\n"
        "3. If query asks about fleet electrification -> select ONLY ['fleet'].\n"
        "4. If query asks about manufacturing defects -> select ONLY ['qms'].\n"
        "5. If query asks about lithium sourcing / supply chain -> select ONLY ['supply_chain'].\n"
        "6. If query asks about carbon emissions -> select ONLY ['carbon'].\n"
        "7. If query asks about maintenance -> select ONLY ['maintenance'].\n"
        "8. For multi-domain queries, select ONLY the required subset.\n"
    )
    
    try:
        query_text = state.get("user_query", "")
        query_snippet = query_text[:300] + "..." if len(query_text) > 300 else query_text
        
        structured_llm = llm.with_structured_output(ExecutionPlan)
        plan = run_with_transient_retry(
            structured_llm.invoke,
            [SystemMessage(content=system_prompt), HumanMessage(content=f"User Query: {query_snippet}")]
        )
        selected = plan.agents
        confidence = plan.confidence
        reasoning = plan.reasoning
    except Exception as e:
        logger.error(f"Planner error: {e}. Falling back to default domain routing.")
        selected = ["apm", "fleet"]
        confidence = 0.5
        reasoning = f"Planner fallback due to error: {str(e)}"
        
    planner_ms = round((time.perf_counter() - start_time) * 1000, 2)
    
    skipped = [a for a in ALL_AGENTS if a not in selected]
    
    logger.info("Planner Complete")
    logger.info(f"Reasoning: {reasoning} (Confidence: {confidence})")
    logger.info("Selected:")
    for a in selected:
        logger.info(f"  ✓ {a.upper()}")
    logger.info("Skipped:")
    for a in skipped:
        logger.info(f"  ✗ {a.upper()}")
        
    metadata = {
        "planner_reasoning": reasoning,
        "confidence": confidence,
        "selected_agents": selected,
        "skipped_agents": skipped,
        "execution_time_ms": {"planner": planner_ms}
    }
    
    return {
        "next_agents": selected,
        "execution_metadata": metadata
    }


# --- Stage 3: Agent Executor Router Conditional Edge ---
def route_to_parallel_agents(state: OrchestratorState):
    sends = []
    selected = state.get("next_agents", [])
    
    # Dependency handling: if BOTH apm and maintenance are selected, use apm_then_maintenance
    if "apm" in selected and "maintenance" in selected:
        sends.append(Send("apm_then_maintenance", state))
        for agent in selected:
            if agent not in ("apm", "maintenance") and agent in _AGENT_REGISTRY:
                sends.append(Send(agent, state))
    else:
        for agent in selected:
            if agent in _AGENT_REGISTRY:
                sends.append(Send(agent, state))

    if not sends:
        return "pre_aggregation_validator"

    return sends


# --- Stage 5: Pre-Aggregation Validator ---
def pre_aggregation_validator_node(state: OrchestratorState) -> OrchestratorState:
    logger.info("Validation Complete")
    agent_responses = state.get("agent_responses", {})
    validated_responses = {}
    
    for key, val in agent_responses.items():
        if key.startswith("_"):
            validated_responses[key] = val
            continue
            
        if not isinstance(val, dict):
            validated_responses[key] = {
                "status": "failed",
                "error": "Invalid response format: output is not a dictionary"
            }
        else:
            if val.get("status") == "error" and not val.get("tool_outputs") and not val.get("summary"):
                validated_responses[key] = {
                    "status": "failed",
                    "error": val.get("summary", "Invalid agent output")
                }
            else:
                validated_responses[key] = val
                
    return {"agent_responses": validated_responses}


# --- Stage 6: Aggregator LLM Node ---
def aggregator_node(state: OrchestratorState) -> OrchestratorState:
    logger.info("Aggregation Started")
    start_time = time.perf_counter()
    
    system_prompt = (
        "You are the Central Orchestrator Aggregator for an EV Fleet Intelligence System.\n"
        "Your task is to synthesize the sub-agent outputs into a coherent, executive-level technical finding report.\n\n"
        "STRICT NUMERICAL & FACTUAL RULES:\n"
        "1. You perform REASONING ONLY. You MUST NOT calculate, estimate, average, interpolate, or invent any numerical values.\n"
        "2. Every reported metric MUST originate directly from sub-agent tool outputs.\n"
        "3. If a specific metric was not provided by sub-agent output, write: 'Not available from current analysis.'\n"
        "4. Synthesize findings across active domains, connect cross-domain relationships, explain risks, and preserve sub-agent recommendations.\n"
        "5. Distinguish clear observations from recommendations.\n"
    )

    responses_json = json.dumps(state.get("agent_responses", {}), indent=2, default=str)
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"User Query:\n{state['user_query']}\n\nSub-Agent Raw Responses:\n{responses_json}")
    ]

    try:
        res = run_with_transient_retry(llm.invoke, messages)
        synthesis_text = res.content if hasattr(res, "content") else str(res)
    except Exception as e:
        logger.error(f"Aggregator LLM error: {e}")
        synthesis_text = f"Aggregation finding synthesis: Sub-agent responses captured. Details: {str(e)}"
        
    agg_ms = round((time.perf_counter() - start_time) * 1000, 2)
    logger.info(f"Aggregation Complete ({agg_ms} ms)")
    
    return {"final_synthesis": synthesis_text}


# --- Stage 7: Response Builder Node ---
def response_builder_node(state: OrchestratorState) -> OrchestratorState:
    start_time = time.perf_counter()
    
    agent_responses = state.get("agent_responses", {})
    metadata = state.get("execution_metadata", {})
    synthesis = state.get("final_synthesis", "")
    selected_agents = metadata.get("selected_agents", [])
    
    # Build Agent Execution Summary Table
    exec_summary_lines = ["| Domain Agent | Execution Status | Key Details / Notes |", "|---|---|---|"]
    for agent_key in ALL_AGENTS:
        if agent_key in agent_responses:
            resp = agent_responses[agent_key]
            if isinstance(resp, dict) and resp.get("status") == "skipped":
                exec_summary_lines.append(f"| {agent_key.upper()} | Skipped | {resp.get('reason', 'Dependency skipped')} |")
            elif isinstance(resp, dict) and resp.get("status") == "failed":
                exec_summary_lines.append(f"| {agent_key.upper()} | Failed | Error: {resp.get('error', 'Execution error')} |")
            else:
                exec_summary_lines.append(f"| {agent_key.upper()} | Executed (Success) | Completed successfully |")
        else:
            exec_summary_lines.append(f"| {agent_key.upper()} | Skipped (Not Requested) | N/A |")
            
    exec_summary_table = "\n".join(exec_summary_lines)
    
    # Assemble Dynamic Report Sections
    report_sections = []
    
    report_sections.append("# Executive Summary\n" + synthesis)
    report_sections.append("## Agent Execution Summary\n" + exec_summary_table)
    
    if "fleet" in selected_agents or "fleet" in agent_responses:
        fleet_data = agent_responses.get("fleet", {})
        report_sections.append("## Fleet Electrification\n" + _format_agent_section("Fleet Electrification", fleet_data))
        
    if "apm" in selected_agents or "apm" in agent_responses:
        apm_data = agent_responses.get("apm", {})
        report_sections.append("## Battery Health\n" + _format_agent_section("Battery Health (APM)", apm_data))
        
    if "qms" in selected_agents or "qms" in agent_responses:
        qms_data = agent_responses.get("qms", {})
        report_sections.append("## Manufacturing Quality\n" + _format_agent_section("Manufacturing Quality (QMS)", qms_data))
        
    if "supply_chain" in selected_agents or "supply_chain" in agent_responses:
        sc_data = agent_responses.get("supply_chain", {})
        report_sections.append("## Supply Chain\n" + _format_agent_section("Supply Chain Intelligence", sc_data))
        
    if "maintenance" in selected_agents or "maintenance" in agent_responses:
        maint_data = agent_responses.get("maintenance", {})
        report_sections.append("## Maintenance\n" + _format_agent_section("Maintenance Operations", maint_data))
        
    if "carbon" in selected_agents or "carbon" in agent_responses:
        carbon_data = agent_responses.get("carbon", {})
        report_sections.append("## Carbon Impact\n" + _format_agent_section("Scope 3 Carbon Impact", carbon_data))
        
    report_sections.append("## Strategic Roadmap\n" + _extract_strategic_roadmap(agent_responses))
    
    final_report = "\n\n".join(report_sections)
    
    builder_ms = round((time.perf_counter() - start_time) * 1000, 2)
    logger.info(f"Response Builder Complete ({builder_ms} ms)")
    logger.info("Supervisor Finished")
    
    # Return backward-compatible dict preserving final_synthesis
    return {"final_synthesis": final_report}


def _format_agent_section(domain_name: str, agent_output: Any) -> str:
    if not agent_output or not isinstance(agent_output, dict):
        return "Not available from current analysis."
        
    if agent_output.get("status") == "skipped":
        return f"Skipped: {agent_output.get('reason', 'Domain analysis skipped by planner/dependency.')}"
        
    if agent_output.get("status") == "failed":
        return f"Execution Failed: {agent_output.get('error', 'Unknown error during agent execution.')}"
        
    summary = agent_output.get("summary") or agent_output.get("unified_report") or agent_output.get("output")
    if summary:
        return str(summary)
        
    return json.dumps(agent_output, indent=2, default=str)


def _extract_strategic_roadmap(agent_responses: dict) -> str:
    recommendations = []
    for agent_key, resp in agent_responses.items():
        if isinstance(resp, dict) and resp.get("recommendations"):
            recs = resp.get("recommendations")
            if isinstance(recs, list):
                recommendations.extend([f"- [{agent_key.upper()}] {r}" for r in recs])
            elif isinstance(recs, str):
                recommendations.append(f"- [{agent_key.upper()}] {recs}")
                
    if recommendations:
        return "\n".join(recommendations)
    return "- Proceed with prioritized domain recommendations as highlighted in agent findings."


# --- Build LangGraph State Graph ---
workflow = StateGraph(OrchestratorState)

workflow.add_node("planner", planner_node)
workflow.add_node("apm", node_apm)
workflow.add_node("qms", node_qms)
workflow.add_node("supply_chain", node_supply_chain)
workflow.add_node("fleet", node_fleet)
workflow.add_node("maintenance", node_maintenance)
workflow.add_node("carbon", node_carbon)
workflow.add_node("apm_then_maintenance", node_apm_then_maintenance)
workflow.add_node("pre_aggregation_validator", pre_aggregation_validator_node)
workflow.add_node("aggregator", aggregator_node)
workflow.add_node("response_builder", response_builder_node)

# Conditional Fan-out from Planner
workflow.add_conditional_edges(
    "planner",
    route_to_parallel_agents,
    ["apm", "qms", "supply_chain", "fleet", "maintenance", "carbon", "apm_then_maintenance", "pre_aggregation_validator"]
)

# Fan-in to pre_aggregation_validator
for agent in ["apm", "qms", "supply_chain", "fleet", "maintenance", "carbon", "apm_then_maintenance"]:
    workflow.add_edge(agent, "pre_aggregation_validator")

# Pre-aggregation validator -> aggregator -> response_builder -> END
workflow.add_edge("pre_aggregation_validator", "aggregator")
workflow.add_edge("aggregator", "response_builder")
workflow.add_edge("response_builder", END)

workflow.set_entry_point("planner")
orchestrator_app = workflow.compile()
