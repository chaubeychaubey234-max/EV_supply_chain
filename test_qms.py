from ev_ai_agents.ev_qms_agent.agent import qms_app
import os

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

res1 = qms_app.invoke({"user_query": "What is the current scrap rate and which production line has the highest defect rate?"})
print("QMS Result 1 Analysis:", res1.get("material_data"))
print("Summary 1:", res1.get("messages", ["None"])[-1])

res2 = qms_app.invoke({"user_query": "Check batch BATCH-002"})
print("QMS Result 2 Analysis:", res2.get("material_data"))
print("Summary 2:", res2.get("messages", ["None"])[-1])
