import os
import json
from dotenv import load_dotenv
from ev_qms_agent.agent import qms_app

def main():
    load_dotenv()
    
    if "OPENAI_API_KEY" not in os.environ:
        print("WARNING: OPENAI_API_KEY not found in environment. Please set it or add to a .env file.")
        return

    print("Starting EV Manufacturing Quality Intelligence (QMS) Agent...")
    
    # Initialize state for a specific manufacturing batch
    initial_state = {
        "batch_id": "BATCH-2026-X1",
        "material_data": {},
        "process_data": {},
        "inspection_data": {},
        "quality_drift_analysis": "",
        "root_cause_analysis": "",
        "alerts": [],
        "messages": []
    }
    
    # Run the graph
    print(f"\nAnalyzing manufacturing data for {initial_state['batch_id']}...\n")
    final_state = qms_app.invoke(initial_state)
    
    # Print the results beautifully
    print("="*50)
    print("📦 MATERIAL DATA:")
    print(json.dumps(final_state["material_data"], indent=2))
    
    print("\n⚙️ PROCESS DATA:")
    print(json.dumps(final_state["process_data"], indent=2))
    
    print("\n🔍 INSPECTION DATA:")
    print(json.dumps(final_state["inspection_data"], indent=2))
    
    print("\n📉 QUALITY DRIFT ANALYSIS:")
    print(final_state["quality_drift_analysis"])
    
    print("\n🔬 ROOT CAUSE ANALYSIS:")
    print(final_state["root_cause_analysis"])
    
    print("\n⚠️ ALERTS:")
    for alert in final_state["alerts"]:
        print(f" - {alert}")
    print("="*50)

if __name__ == "__main__":
    main()
