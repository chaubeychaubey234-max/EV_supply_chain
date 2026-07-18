import os
import json
import traceback
from dotenv import load_dotenv
from ev_ai_agents.ev_apm_agent.agent import apm_app
from ev_ai_agents.ev_qms_agent.agent import qms_app
from ev_ai_agents.ev_fleet_electrification_agent.agent import run_agent as run_fleet_agent

def test_apm_agent():
    print("\n" + "="*50)
    print("TESTING APM AGENT")
    print("="*50)
    
    test_cases = [
        {"name": "Conceptual Query", "state": {"user_query": "Why is battery health decreasing?"}},
        {"name": "Statistical Query", "state": {"user_query": "What is the average battery health across the fleet?"}},
        {"name": "Asset Query", "state": {"user_query": "Analyze EV-102."}},
        {"name": "Hybrid Query", "state": {"user_query": "Analyze EV-102 and compare it with the fleet average."}},
        {"name": "Invalid Asset", "state": {"user_query": "Analyze EV-999999."}},
        {"name": "Empty Query", "state": {"user_query": ""}},
        {"name": "No Query (Legacy/Supervisor Mode)", "state": {"ev_id": "EV-9005"}}
    ]
    
    for case in test_cases:
        print(f"\n[APM Test] {case['name']}")
        print(f"Input state: {case['state']}")
        try:
            res = apm_app.invoke(case["state"])
            print("Status: Success")
            print("Keys returned:", list(res.keys()))
            print("Detected intent:", res.get("detected_intent"))
            print("Confidence:", res.get("confidence"))
            print("Recommendations summary:", res.get("recommendations", [])[:2])
            print("Battery analysis status:", res.get("battery_analysis", {}).get("status"))
        except Exception as e:
            print("Status: Failed")
            traceback.print_exc()

def test_qms_agent():
    print("\n" + "="*50)
    print("TESTING QMS AGENT")
    print("="*50)
    
    test_cases = [
        {"name": "Conceptual Query", "state": {"user_query": "What causes process drift?"}},
        {"name": "Statistical Query", "state": {"user_query": "What is the scrap rate of cells?"}},
        {"name": "Asset Query", "state": {"user_query": "Inspect Batch BATCH-2026-X1."}},
        {"name": "Hybrid Query", "state": {"user_query": "Analyze BATCH-2026-X1 and compare it against historical quality."}},
        {"name": "Invalid Asset", "state": {"user_query": "Analyze BATCH-999999."}},
        {"name": "Empty Query", "state": {"user_query": ""}},
        {"name": "No Query (Legacy/Supervisor Mode)", "state": {"batch_id": "BATCH-2026-X1"}}
    ]
    
    for case in test_cases:
        print(f"\n[QMS Test] {case['name']}")
        print(f"Input state: {case['state']}")
        try:
            res = qms_app.invoke(case["state"])
            print("Status: Success")
            print("Keys returned:", list(res.keys()))
            print("Detected intent:", res.get("detected_intent"))
            print("Confidence:", res.get("confidence"))
            print("Quality drift analysis snippet:", res.get("quality_drift_analysis", "")[:100] + "...")
            print("Batch metrics total inspected:", res.get("batch_metrics", {}).get("total_inspected"))
        except Exception as e:
            print("Status: Failed")
            traceback.print_exc()

def test_fleet_agent():
    print("\n" + "="*50)
    print("TESTING FLEET ELECTRIFICATION AGENT")
    print("="*50)
    
    test_cases = [
        {"name": "Conceptual Query", "query": "What is fleet electrification?"},
        {"name": "Readiness Query", "query": "Is my fleet ready for electrification?"},
        {"name": "ROI Query", "query": "What is my expected ROI?"},
        {"name": "Matching Query", "query": "Which EV should replace my current vehicle?"},
        {"name": "Procurement Query", "query": "What procurement strategy do you recommend?"},
        {"name": "Hybrid Query", "query": "Recommend EV replacements and estimate ROI."},
        {"name": "Invalid Asset", "query": "Inspect an unknown vehicle.", "vehicle_id": "VEH-INVALID-999"},
        {"name": "Empty Query", "query": ""},
        {"name": "No/Missing Query", "query": None}
    ]
    
    for case in test_cases:
        print(f"\n[Fleet Test] {case['name']}")
        query = case.get("query")
        vehicle_id = case.get("vehicle_id", "VEH-002")
        print(f"Input: query='{query}', vehicle_id='{vehicle_id}'")
        try:
            res = run_fleet_agent(query, vehicle_id=vehicle_id)
            print("Status: Success")
            print("Response fields:", list(res.keys()))
            print("Status returned:", res.get("status"))
            print("Summary snippet:", res.get("summary")[:100] + ("..." if len(res.get("summary", "")) > 100 else ""))
            print("Recommendations:", res.get("recommendations"))
            print("Next Steps:", res.get("next_steps"))
            print("Executed tools:", res.get("selected_tools"))
        except Exception as e:
            print("Status: Failed")
            traceback.print_exc()

if __name__ == "__main__":
    load_dotenv()
    test_apm_agent()
    test_qms_agent()
    test_fleet_agent()
