import os
import json
import traceback
from dotenv import load_dotenv
from ev_ai_agents.ev_apm_agent.agent import apm_app
from ev_ai_agents.ev_qms_agent.agent import qms_app

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

if __name__ == "__main__":
    load_dotenv()
    test_apm_agent()
    test_qms_agent()
