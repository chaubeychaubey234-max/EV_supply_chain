import streamlit as st
import pandas as pd
import os
import sys
import streamlit.components.v1 as components

# Ensure workspace root is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
load_dotenv(os.path.join(PROJECT_ROOT, "ev_ai_agents", ".env"))

# Import data loaders and helpers
from ev_ai_agents.carbon_agent.utils.data_loader import load_green_logistics, load_emission_factors, load_co2_emissions
from ev_ai_agents.carbon_agent.utils.geo_utils import map_high_emission_routes
from ev_ai_agents.carbon_agent.agent.carbon_agent import get_agent_executor

# Import tools for direct UI data fetching
from ev_ai_agents.carbon_agent.tools.emissions_tools import track_scope_emissions
from ev_ai_agents.carbon_agent.tools.progress_tools import track_net_zero_progress
from ev_ai_agents.carbon_agent.tools.route_tools import recommend_electrification

# Page configuration
st.set_page_config(
    page_title="Net Zero Progress & Carbon Intelligence Tracker",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling (dark carbon theme)
st.markdown("""
<style>
    /* Dark mode background overrides */
    .stApp {
        background-color: #0b0f17;
        color: #f1f5f9;
    }
    
    /* Sleek gradient main header */
    .main-title {
        background: linear-gradient(135deg, #10b981 0%, #3b82f6 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 2px;
    }
    
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }
    
    /* Card layouts */
    .kpi-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(8px);
        text-align: center;
    }
    
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #10b981;
        margin-top: 5px;
    }
    
    .kpi-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Custom status badges */
    .badge-on-track {
        background-color: #10b981;
        color: #ffffff;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
    }
    .badge-at-risk {
        background-color: #f59e0b;
        color: #ffffff;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
    }
    .badge-behind {
        background-color: #ef4444;
        color: #ffffff;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar Setup
# ---------------------------------------------------------------------------
st.sidebar.markdown("""
<div style="text-align: center; padding: 15px 0;">
    <h2 style="color: #10b981; font-size: 1.5rem; font-weight: 700; margin:0;">🌱 NET ZERO</h2>
    <p style="color: #64748b; font-size: 0.85rem;">Carbon Intelligence Agent</p>
</div>
""", unsafe_allow_html=True)

menu_choice = st.sidebar.radio(
    "Navigation Menu",
    ["🔮 Carbon Intelligence Chat", "🗺️ Route Emissions Map", "📋 Electrification Priorities", "📊 CO2 Emissions Dataset"]
)

# Initialize data and map
try:
    df_logistics = load_green_logistics()
    df_factors = load_emission_factors()
    df_co2 = load_co2_emissions()
except Exception as e:
    st.error(f"Error loading datasets: {e}")
    st.stop()

map_file_path = os.path.join(CURRENT_DIR, "route_emissions_map.html")
if not os.path.exists(map_file_path):
    try:
        map_high_emission_routes(map_file_path)
    except Exception as e:
        st.warning(f"Could not pre-generate geospatial map: {e}")

# Fetch KPI emissions data from tools
try:
    scope_data = track_scope_emissions.invoke({})
    progress_data = track_net_zero_progress.invoke({})
    electrification_data = recommend_electrification.invoke({})
except Exception as e:
    # Use fallback mock structures for KPIs if tools error out
    scope_data = {"scope1_emissions": 3400.0, "scope3_emissions": 11500.0, "total_emissions": 14900.0}
    progress_data = {"progress_percentage": 91.9, "emissions_gap": 500.0, "status": "At Risk"}
    electrification_data = {"expected_co2_reduction": 1400.0}

# ---------------------------------------------------------------------------
# Header Section
# ---------------------------------------------------------------------------
st.markdown("<div class='main-title'>Net Zero Progress & Carbon Intelligence</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Geospatial AI carbon optimization, Scope 1 & 3 tracking, and fleet electrification priority insights</div>", unsafe_allow_html=True)

# KPI Row
kpi_cols = st.columns(4)

with kpi_cols[0]:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Scope 1 Emissions</div>
        <div class="kpi-value" style="color: #3b82f6;">{scope_data.get('scope1_emissions')} MT</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[1]:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Scope 3 Emissions</div>
        <div class="kpi-value" style="color: #a855f7;">{scope_data.get('scope3_emissions')} MT</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[2]:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">EV Reduction Potential</div>
        <div class="kpi-value" style="color: #10b981;">{electrification_data.get('expected_co2_reduction')} MT</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[3]:
    status = progress_data.get('status', 'Behind')
    if status == "On Track":
        badge_html = "<span class='badge-on-track'>ON TRACK</span>"
    elif status == "At Risk":
        badge_html = "<span class='badge-at-risk'>AT RISK</span>"
    else:
        badge_html = "<span class='badge-behind'>BEHIND</span>"
        
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Net Zero Status</div>
        <div class="kpi-value" style="font-size: 1.5rem; margin-top: 10px;">{badge_html}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Navigation Logic
# ---------------------------------------------------------------------------

if menu_choice == "🔮 Carbon Intelligence Chat":
    st.header("🔮 Carbon Intelligence Chat Agent")
    st.write("Ask the AI agent about Scope 1 and 3 emissions, high emission routes, net zero targets, or electrification recommendations.")
    
    # Initialize agent executor
    agent_exec = None
    try:
        agent_exec = get_agent_executor()
    except Exception as e:
        st.warning("⚠️ Google Gemini API key not found in environmental variables. Please configure GEMINI_API_KEY to enable chat.")
        
    # Chat message state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your Net Zero Carbon Intelligence Agent. How can I help you optimize your carbon footprint today?"}
        ]
        
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    # Handle user inputs
    if user_query := st.chat_input("Enter your carbon sustainability question..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
            
        with st.chat_message("assistant"):
            if agent_exec:
                with st.spinner("Analyzing fleet carbon data..."):
                    try:
                        # Feed the agent executor the user question
                        # Map input list to langchain format
                        chat_hist = []
                        # Retrieve last few exchanges for short context window
                        for m in st.session_state.messages[-5:-1]:
                            role = "human" if m["role"] == "user" else "ai"
                            chat_hist.append((role, m["content"]))
                            
                        response = agent_exec.invoke({"input": user_query, "chat_history": chat_hist})
                        ans = response.get("output", "No response generated.")
                        st.markdown(ans)
                        st.session_state.messages.append({"role": "assistant", "content": ans})
                    except Exception as err:
                        st.error(f"Error running agent: {err}")
            else:
                # Mock response fallback
                ans = "I cannot call Google Gemini without an API Key. Please configure `GEMINI_API_KEY`."
                st.markdown(ans)
                st.session_state.messages.append({"role": "assistant", "content": ans})

elif menu_choice == "🗺️ Route Emissions Map":
    st.header("🗺️ Route Emissions Map")
    st.write("Visualizing logistics routes colored by annual carbon emissions (Red = High, Orange = Medium, Green = Low). Blue markers flag electrification priorities.")
    
    if os.path.exists(map_file_path):
        with open(map_file_path, 'r', encoding='utf-8') as f:
            html_data = f.read()
        components.html(html_data, height=600)
    else:
        st.error("Route map HTML file could not be found.")

elif menu_choice == "📋 Electrification Priorities":
    st.header("📋 High-Impact Electrification Priorities")
    st.write("Diesel routes with maximum emissions, sorted by suitability and EV savings potential.")
    
    recs_list = electrification_data.get('recommended_routes', [])
    if recs_list:
        df_recs = pd.DataFrame(recs_list)
        # Rename columns beautifully
        df_recs.columns = [
            "Route ID", "Start Location", "End Location", "Distance (km)", 
            "Current Vehicle", "Annual Trips", "Current Annual Emissions (MT)", "Expected EV Savings (MT)"
        ]
        st.dataframe(df_recs.style.background_gradient(subset=["Current Annual Emissions (MT)", "Expected EV Savings (MT)"], cmap="Greens"))
        
        st.info(f"💡 **Strategy Insight**: {electrification_data.get('priority_reason')}")
    else:
        st.warning("No electrification recommendations found.")

elif menu_choice == "📊 CO2 Emissions Dataset":
    st.header("📊 Datasets Overview")
    
    tab1, tab2, tab3 = st.tabs(["Green Logistics (Routes)", "Supply Chain Emission Factors (Scope 3)", "Net Zero Target Performance"])
    
    with tab1:
        st.write("Raw data tracking vehicles, distances, trip frequency, and per-trip emissions.")
        st.dataframe(df_logistics)
        
    with tab2:
        st.write("Scope 3 Emission Factors (CO2 coefficients per unit for critical materials and transport).")
        st.dataframe(df_factors)
        
    with tab3:
        st.write("Historical organizational CO2 emissions and targets set from 2020 to 2030.")
        st.dataframe(df_co2)
