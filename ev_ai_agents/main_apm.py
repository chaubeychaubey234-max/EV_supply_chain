import os
import json
from dotenv import load_dotenv
from ev_apm_agent.agent import apm_app

def main():
    load_dotenv()
    
    if "GROQ_API_KEY" not in os.environ:
        print("WARNING: GROQ_API_KEY not found in environment. Please set it or add to a .env file.")
        return

    print("Starting EV Asset Performance Management (APM) Agent...")
    
    # Initialize state for a specific EV
    # Assuming EV_1002 exists in the dataset
    initial_state = {
        "ev_id": "EV_1002",
        "telemetry_data": {},
        "battery_analysis": {},
        "safety_analysis": {},
        "recommendations": [],
        "maintenance_triggers": [],
        "messages": []
    }
    
    print(f"\nAnalyzing telemetry data for {initial_state['ev_id']}...\n")
    final_state = apm_app.invoke(initial_state)
    
    # Print the results beautifully
    print("="*50)
    print("🔋 BATTERY ANALYSIS:")
    print(json.dumps(final_state.get("battery_analysis", {}), indent=2))
    
    print("\n🔥 SAFETY/THERMAL ANALYSIS:")
    print(json.dumps(final_state.get("safety_analysis", {}), indent=2))

    print("\n⚡ CHARGING PATTERNS:")
    print(json.dumps(final_state.get("telemetry_data", {}), indent=2))
    
    print("\n🛠️  AI RECOMMENDATIONS:")
    for rec in final_state.get("recommendations", []):
        print(f"  - {rec}")

    print("\n⚠️  PREDICTIVE MAINTENANCE TRIGGERS:")
    for trigger in final_state.get("maintenance_triggers", []):
        print(f"  - {trigger}")
    print("="*50)

if __name__ == "__main__":
    main()
