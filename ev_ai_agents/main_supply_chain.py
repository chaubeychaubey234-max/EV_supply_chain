import os
import sys
import json
from dotenv import load_dotenv

# Ensure the workspace root is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ev_ai_agents.ev_supply_chain_agent.agent import supply_chain_app

def print_banner(title: str):
    print("=" * 60)
    print(f" {title} ".center(60, "■"))
    print("=" * 60)

def run_agent_audit(query: str):
    print_banner("RUNNING SUPPLY CHAIN INTELLIGENCE AUDIT")
    print(f"Query: \"{query}\"\n")
    
    # Setup initial graph state
    initial_state = {
        "query": query,
        "supplier_id": None,
        "batch_id": None,
        "material": None,
        "country": None,
        "supplier_analysis": None,
        "traceability_analysis": None,
        "risk_analysis": None,
        "quality_analysis": None,
        "unified_report": None,
        "messages": []
    }
    
    # Invoke LangGraph multi-agent orchestration
    final_state = supply_chain_app.invoke(initial_state)
    
    # Display intermediate agent steps and trace logs
    print("\n--- [AGENT ORCHESTRATION LOGS] ---")
    for msg in final_state.get("messages", []):
        role = msg.get("role", "system").upper()
        content = msg.get("content", "")
        print(f"[{role}]: {content}")
    print("-" * 34)
    
    # Display the final unified report
    print_banner("UNIFIED INTELLIGENCE REPORT")
    print(final_state.get("unified_report"))
    print("=" * 60)
    print("\n")

def main():
    load_dotenv()
    
    # Check if there is an argument passed
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        run_agent_audit(query)
    else:
        # Default demo scenarios
        demo_queries = [
            "Audit supplier SUP-001 (Ganfeng Lithium Co.) and review geopolitical risk and quality deviations.",
            "Run a traceability and risk review for cobalt batch BAT-2024-002 from Glencore Congo.",
        ]
        
        print("Starting EV Supply Chain Risk & Traceability Multi-Agent System Demo...")
        print(f"No custom query provided. Running {len(demo_queries)} default demo queries.\n")
        
        for q in demo_queries:
            run_agent_audit(q)

if __name__ == "__main__":
    main()
