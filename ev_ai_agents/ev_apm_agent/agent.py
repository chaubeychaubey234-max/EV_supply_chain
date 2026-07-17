import os
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List

from .state import APMState
from .tools.battery_tools import fetch_battery_health, predict_battery_health
from .tools.thermal_tools import fetch_thermal_events
from .tools.charging_tools import fetch_charging_patterns

# Define expected structured output from the LLM
class APMAnalysisOutput(BaseModel):
    recommendations: List[str] = Field(description="List of optimal charge-discharge recommendations.")
    maintenance_triggers: List[str] = Field(description="List of predictive maintenance triggers.")

def fetch_data_node(state: APMState):
    """Fetches all telemetry data for the EV or runs ML prediction."""
    
    # Check if we have raw telemetry data for prediction
    if state.get("avg_temperature_c") is not None:
        battery_data = predict_battery_health.invoke({
            "avg_temperature_c": state["avg_temperature_c"],
            "fast_charge_ratio_pct": state["fast_charge_ratio_pct"],
            "deep_discharge_cycles": state["deep_discharge_cycles"],
            "avg_charge_duration_hours": state["avg_charge_duration_hours"],
            "max_temperature_c": state["max_temperature_c"]
        })
        # Mock thermal/charging data for prediction pathway
        thermal_data = {"max_recorded_temperature_celsius": state["max_temperature_c"]}
        charging_data = {"fast_charging_ratio_percentage": state["fast_charge_ratio_pct"]}
    else:
        # Historical lookup
        ev_id = state.get("ev_id", "Unknown")
        battery_data = fetch_battery_health.invoke({"ev_id": ev_id})
        thermal_data = fetch_thermal_events.invoke({"ev_id": ev_id})
        charging_data = fetch_charging_patterns.invoke({"ev_id": ev_id})
    
    return {
        "telemetry_data": charging_data,
        "battery_analysis": battery_data,
        "safety_analysis": thermal_data
    }

def analyze_health_node(state: APMState):
    """Uses LLM to analyze the data and generate recommendations and triggers."""
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
    structured_llm = llm.with_structured_output(APMAnalysisOutput)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert AI for Industrial EV Supply Chain & Asset Intelligence. Your task is to analyze EV telemetry data and generate precise predictive maintenance triggers and charge-discharge recommendations."),
        ("user", "Analyze the following data for EV ID: {ev_id}\n\nBattery Health: {battery_analysis}\n\nThermal Safety: {safety_analysis}\n\nCharging Patterns: {telemetry_data}\n\nProvide actionable recommendations and predictive maintenance triggers.")
    ])
    
    chain = prompt | structured_llm
    
    result = chain.invoke({
        "ev_id": state["ev_id"],
        "battery_analysis": state["battery_analysis"],
        "safety_analysis": state["safety_analysis"],
        "telemetry_data": state["telemetry_data"]
    })
    
    return {
        "recommendations": result.recommendations,
        "maintenance_triggers": result.maintenance_triggers
    }

# Build the graph
workflow = StateGraph(APMState)

workflow.add_node("fetch_data", fetch_data_node)
workflow.add_node("analyze_health", analyze_health_node)

workflow.set_entry_point("fetch_data")
workflow.add_edge("fetch_data", "analyze_health")
workflow.add_edge("analyze_health", END)

apm_app = workflow.compile()
