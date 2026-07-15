import os
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List

from .state import QMSState
from .tools.material_tools import fetch_material_data
from .tools.process_tools import fetch_process_data
from .tools.inspection_tools import fetch_inspection_data

class QMSAnalysisOutput(BaseModel):
    quality_drift_analysis: str = Field(description="Analysis of any quality drift over time.")
    root_cause_analysis: str = Field(description="Root cause analysis based on correlations across data.")
    alerts: List[str] = Field(description="Actionable alerts or warnings for operators.")

def fetch_qms_data_node(state: QMSState):
    """Fetches all manufacturing data for the batch."""
    batch_id = state["batch_id"]
    
    # We call the underlying function of the tool directly
    material = fetch_material_data.invoke({"batch_id": batch_id})
    process = fetch_process_data.invoke({"batch_id": batch_id})
    inspection = fetch_inspection_data.invoke({"batch_id": batch_id})
    
    return {
        "material_data": material,
        "process_data": process,
        "inspection_data": inspection
    }

def analyze_drift_node(state: QMSState):
    """Uses LLM to detect quality drift and find root causes."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    structured_llm = llm.with_structured_output(QMSAnalysisOutput)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI Quality Agent for EV component manufacturing. Correlate process parameters, incoming material data, and inspection results to detect quality drift and perform root-cause analysis."),
        ("user", "Analyze the following manufacturing batch: {batch_id}\n\nMaterial Data: {material}\n\nProcess Data: {process}\n\nInspection Data: {inspection}\n\nProvide a quality drift analysis, root cause, and alerts.")
    ])
    
    chain = prompt | structured_llm
    
    result = chain.invoke({
        "batch_id": state["batch_id"],
        "material": state["material_data"],
        "process": state["process_data"],
        "inspection": state["inspection_data"]
    })
    
    return {
        "quality_drift_analysis": result.quality_drift_analysis,
        "root_cause_analysis": result.root_cause_analysis,
        "alerts": result.alerts
    }

# Build the graph
workflow = StateGraph(QMSState)

workflow.add_node("fetch_data", fetch_qms_data_node)
workflow.add_node("analyze_drift", analyze_drift_node)

workflow.set_entry_point("fetch_data")
workflow.add_edge("fetch_data", "analyze_drift")
workflow.add_edge("analyze_drift", END)

qms_app = workflow.compile()
