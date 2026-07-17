import os
import sys

# Ensure workspace root is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from dotenv import load_dotenv

# Try loading env from root or ev_ai_agents
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
load_dotenv(os.path.join(PROJECT_ROOT, "ev_ai_agents", ".env"))

from ev_ai_agents.carbon_agent.agent.carbon_agent import get_agent_executor

def run_tests():
    print("="*60)
    print("🍀 STARTING CARBON INTELLIGENCE AGENT VERIFICATION TESTS")
    print("="*60)

    try:
        agent_executor = get_agent_executor()
    except Exception as e:
        print(f"❌ Initialization Error: {e}")
        print("\nPlease set your GEMINI_API_KEY or GOOGLE_API_KEY environment variable and try again.")
        print("Example: export GEMINI_API_KEY='your-api-key'")
        return

    test_cases = [
        "Which routes should be electrified first?",
        "Calculate Scope 1 and Scope 3 emissions.",
        "How much emission reduction can EV adoption provide?",
        "Are we progressing toward net zero?",
        "Show high emission routes on map"
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n📝 TEST CASE {i}: {test}")
        print("-" * 40)
        try:
            response = agent_executor.invoke({"input": test, "chat_history": []})
            print("🤖 Agent Response:")
            print(response.get("output", "No response output."))
        except Exception as e:
            print(f"💥 Failed to run test: {e}")
        print("="*60)

if __name__ == "__main__":
    run_tests()
