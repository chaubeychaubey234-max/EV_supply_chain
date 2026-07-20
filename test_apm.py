from ev_ai_agents.ev_apm_agent.agent import apm_app
import os

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

res = apm_app.invoke({"user_query": "What is the battery health status of EV-9005?"})
print("APM Result Battery Analysis:", res.get("battery_analysis"))
print("Summary:", res.get("messages", ["None"])[-1])
