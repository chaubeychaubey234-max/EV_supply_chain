import os
import sys
from dotenv import load_dotenv

# Ensure the workspace root is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from ev_ai_agents.carbon_agent.tools import (
    calculate_emissions_reduction,
    track_scope_emissions,
    analyze_route_emissions,
    identify_high_impact_routes,
    recommend_electrification,
    generate_and_save_route_map,
    track_net_zero_progress
)

# Load environment variables
load_dotenv()

class SimpleAgentExecutor:
    """A self-contained message-based agent executor for tool calling using Gemini LLM."""
    
    def __init__(self, llm, tools, system_prompt: str):
        self.llm_with_tools = llm.bind_tools(tools)
        self.tools_map = {t.name: t for t in tools}
        self.system_prompt = system_prompt
        
    def invoke(self, inputs: dict) -> dict:
        user_input = inputs["input"]
        chat_history = inputs.get("chat_history", [])
        
        # Build initial messages list
        messages = [SystemMessage(content=self.system_prompt)]
        
        # Append chat history
        for role, text in chat_history:
            if role == "human":
                messages.append(HumanMessage(content=text))
            elif role == "ai":
                messages.append(AIMessage(content=text))
                
        # Append current user question
        messages.append(HumanMessage(content=user_input))
        
        # Run agent reasoning loop (max 10 iterations)
        for i in range(10):
            response = self.llm_with_tools.invoke(messages)
            messages.append(response)
            
            # If no tool calls are requested by the model, return the text content
            if not response.tool_calls:
                return {"output": response.content}
                
            # Otherwise, iterate and execute each tool call
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]
                
                if tool_name in self.tools_map:
                    try:
                        tool_res = self.tools_map[tool_name].invoke(tool_args)
                        # Ensure the output is formatted as a string
                        tool_res_str = str(tool_res)
                    except Exception as e:
                        tool_res_str = f"Error executing tool {tool_name}: {str(e)}"
                else:
                    tool_res_str = f"Error: Tool '{tool_name}' not found."
                    
                # Append the tool message to messages list
                messages.append(ToolMessage(content=tool_res_str, tool_call_id=tool_id))
                
        return {"output": "Error: Maximum agent tool iterations reached without completion."}

def get_agent_executor() -> SimpleAgentExecutor:
    """Instantiate the Gemini LLM, bind tools, and return a SimpleAgentExecutor."""
    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable is required to run the agent."
        )

    # Initialize Gemini model
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        api_key=os.getenv("GROQ_API_KEY")
    )

    # Compile the list of tools
    tools = [
        calculate_emissions_reduction,
        track_scope_emissions,
        analyze_route_emissions,
        identify_high_impact_routes,
        recommend_electrification,
        generate_and_save_route_map,
        track_net_zero_progress
    ]

    # Custom sustainability consultant system prompt
    system_prompt = (
        "You are the Net Zero Carbon Intelligence Agent, a senior sustainability consultant and geospatial analyst.\n"
        "Your goal is to help users track fleet electrification progress against net zero commitments, analyze route emissions, and suggest priorities for electrification.\n\n"
        "Follow these rules for your responses:\n"
        "1. DO NOT return raw numbers, raw database rows, or raw JSON dictionaries directly to the user.\n"
        "2. Convert all raw outputs from tools into rich business insights, clear carbon impact explanations, concrete sustainability recommendations, and priority actions.\n"
        "3. If the user asks to show or view high emission routes on a map, invoke the `generate_and_save_route_map` tool to rebuild the map, explain what the map visualizes (e.g. high-emission corridors in red, medium in orange, low in green), and mention that the interactive map has been updated on the dashboard.\n"
        "4. Maintain a highly professional, analytical, and actionable tone."
    )

    return SimpleAgentExecutor(llm, tools, system_prompt)
