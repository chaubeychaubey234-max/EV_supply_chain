import os
import json
from dotenv import load_dotenv
from ev_apm_agent.agent import apm_app

def main():
    # Load environment variables (e.g. OPENAI_API_KEY)
    load_dotenv()
    
    if "OPENAI_API_KEY" not in os.environ:
        print("WARNING: OPENAI_API_KEY not found in environment. Please set it or add to a .env file.")
        return

    print("Starting EV Asset Performance Management (APM) Agent...")
    
    # Initialize state for a specific EV
    initial_state = {
        "ev_id": "EV-9001",
        "telemetry_data": {},
        "battery_analysis": {},
        "safety_analysis": {},
        "recommendations": [],
        "maintenance_triggers": [],
        "messages": []
    }
    
    # Run the graph
    print(f"\nRunning analysis for {initial_state['ev_id']}...\n")
    final_state = apm_app.invoke(initial_state)
    
    # Print the results beautifully
    print("="*50)
    print("🔋 BATTERY HEALTH:")
    print(json.dumps(final_state["battery_analysis"], indent=2))
    
    print("\n🌡️ THERMAL SAFETY:")
    print(json.dumps(final_state["safety_analysis"], indent=2))
    
    print("\n⚡ CHARGING PATTERNS:")
    print(json.dumps(final_state["telemetry_data"], indent=2))
    
    print("\n🛠️ PREDICTIVE MAINTENANCE TRIGGERS:")
    for trigger in final_state["maintenance_triggers"]:
        print(f" - {trigger}")
        
    print("\n💡 CHARGE-DISCHARGE RECOMMENDATIONS:")
    for rec in final_state["recommendations"]:
        print(f" - {rec}")
    print("="*50)

if __name__ == "__main__":
    main()
