import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Ensure workspace root is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ev_ai_agents.ev_supply_chain_agent.agent import supply_chain_app
from ev_ai_agents.ev_supply_chain_agent.datasets import (
    load_supply_chain,
    load_minerals_risk,
    load_battery_quality,
    query_supplier_demographics,
    query_material_traceability,
    query_batch_quality
)
from ev_ai_agents.ev_supply_chain_agent.tools.traceability_tools import (
    trace_material_batch,
    verify_traceability_completeness,
    map_pack_to_vehicle
)

# Page configuration
st.set_page_config(
    page_title="EV Supply Chain Risk & Traceability Intelligence",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium dark theme styling
st.markdown("""
<style>
    /* Dark mode background overrides */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    /* Sleek gradient main header */
    .main-title {
        background: linear-gradient(135deg, #3b82f6 0%, #10b981 50%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }
    
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }
    
    /* Card layouts */
    .card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #10b981;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Custom vertical trace line timeline */
    .trace-node {
        display: flex;
        align-items: center;
        margin-bottom: 12px;
        padding: 10px 15px;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
    }
    .trace-node-badge {
        font-weight: bold;
        color: #3b82f6;
        margin-right: 10px;
        min-width: 80px;
    }
    
    /* Styled labels for levels */
    .badge-high {
        background-color: #ef4444;
        color: white;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .badge-med {
        background-color: #f59e0b;
        color: white;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .badge-low {
        background-color: #10b981;
        color: white;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------------------------
st.sidebar.markdown("""
<div style="text-align: center; padding: 15px 0;">
    <h2 style="color: #3b82f6; font-size: 1.5rem; font-weight: 700; margin:0;">🔋 EV INTEL</h2>
    <p style="color: #64748b; font-size: 0.85rem;">Multi-Agent Supply Chain Risk Agent</p>
</div>
""", unsafe_allow_html=True)

menu_choice = st.sidebar.radio(
    "Navigation Menu",
    ["🔮 Multi-Agent Auditor", "📦 Sourcing Database", "🧬 Traceability Inspector", "⚠️ Geopolitical Risks", "🔬 Quality Analytics"]
)

# Load dataframes
df_chain = load_supply_chain()
df_risk = load_minerals_risk()
df_quality = load_battery_quality()

# High-level aggregated statistics
total_suppliers = int(df_chain["supplier_id"].nunique())
total_materials = int(df_chain["material"].nunique())
total_batches = int(df_chain["batch_id"].nunique())
total_vehicles = int(df_chain["vehicle_id"].nunique())

# Main Title Header
st.markdown('<div class="main-title">EV Supply Chain Risk & Traceability Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Multi-agent diagnostics, geopolitical mineral risk assessments, and EU Battery Passport audits</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 1. Menu Page: Multi-Agent Auditor
# ---------------------------------------------------------------------------
if menu_choice == "🔮 Multi-Agent Auditor":
    st.markdown("### 🔮 Conversational Multi-Agent Supply Chain Auditor")
    st.write("Prompt the intelligence graph to query supplier profiles, trace batches down to VINs, compute geopolitical exposure, and detect quality defects.")
    
    # Preset scenarios for easy demonstration
    col_pre1, col_pre2, col_pre3 = st.columns(3)
    with col_pre1:
        if st.button("Scenario A: Audit Ganfeng Lithium (SUP-001)"):
            st.session_state["query_input"] = "Audit supplier SUP-001 and review material risk and batch deviations."
    with col_pre2:
        if st.button("Scenario B: Trace Cobalt batch BAT-2024-002"):
            st.session_state["query_input"] = "Trace cobalt batch BAT-2024-002 back to vehicles and audit its quality deviations."
    with col_pre3:
        if st.button("Scenario C: Geopolitical Concentration Risk"):
            st.session_state["query_input"] = "Evaluate critical mineral supply concentration and geopolitical risks for China and Russia."

    # Search Query Input
    query_val = st.text_input(
        "Enter Sourcing Intelligence Query:",
        value=st.session_state.get("query_input", "Audit supplier SUP-001 and trace batch BAT-2024-001 for risks and quality deviations.")
    )

    if st.button("Run Multi-Agent Graph", type="primary"):
        with st.spinner("Invoking LangGraph Orchestrator (Supplier, Traceability, Risk, and Quality specialists)..."):
            
            # Form initial state
            initial_state = {
                "query": query_val,
                "supplier_id": None,
                "batch_id": None,
                "material": None,
                "country": None,
                "supplier_analysis": None,
                "traceability_analysis": None,
                "risk_analysis": None,
                "quality_analysis": None,
                "unified_report": None,
                "messages": []
            }
            
            # Execute
            final_state = supply_chain_app.invoke(initial_state)
            
            # Display step-by-step logs from orchestrator
            st.markdown("#### ⚡ Agent Execution Trace Logs")
            log_container = st.container()
            with log_container:
                for msg in final_state.get("messages", []):
                    role = msg.get("role", "system").upper()
                    content = msg.get("content", "")
                    if "warning" in role.lower():
                        st.warning(f"[{role}]: {content}")
                    elif "error" in role.lower():
                        st.error(f"[{role}]: {content}")
                    else:
                        st.info(f"[{role}]: {content}")
                        
            # Display final report
            st.markdown("---")
            st.markdown("### 📋 Unified Intelligence Report")
            
            report_md = final_state.get("unified_report", "No report generated.")
            st.markdown(f'<div class="card">{report_md}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 2. Menu Page: Sourcing Database
# ---------------------------------------------------------------------------
elif menu_choice == "📦 Sourcing Database":
    st.markdown("### 📦 Supply Chain Material & Supplier Database")
    
    # Visual KPI Blocks
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    with col_kpi1:
        st.markdown(f'<div class="card"><div class="metric-value">{total_suppliers}</div><div class="metric-label">Suppliers Audited</div></div>', unsafe_allow_html=True)
    with col_kpi2:
        st.markdown(f'<div class="card"><div class="metric-value">{total_materials}</div><div class="metric-label">Critical Materials</div></div>', unsafe_allow_html=True)
    with col_kpi3:
        st.markdown(f'<div class="card"><div class="metric-value">{total_batches}</div><div class="metric-label">Material Batches</div></div>', unsafe_allow_html=True)
    with col_kpi4:
        st.markdown(f'<div class="card"><div class="metric-value">{total_vehicles}</div><div class="metric-label">Affected Vehicles</div></div>', unsafe_allow_html=True)
        
    st.markdown("#### Raw Supplier Demographics Mapping (`ev_supply_chain.csv`)")
    st.dataframe(df_chain, use_container_width=True)


# ---------------------------------------------------------------------------
# 3. Menu Page: Traceability Inspector
# ---------------------------------------------------------------------------
elif menu_choice == "🧬 Traceability Inspector":
    st.markdown("### 🧬 Battery Traceability & Lineage Auditor")
    st.write("Inspect end-to-end material pathways: Material Sourcing → Cell Manufacture → Pack Assembly → Vehicle Assembly.")
    
    batch_list = sorted(list(df_chain["batch_id"].dropna().unique()))
    selected_batch = st.selectbox("Select Batch ID to Inspect Lineage:", batch_list)
    
    if selected_batch:
        trace_recs = query_material_traceability(selected_batch)
        if trace_recs:
            mat = trace_recs[0]["material"]
            
            st.markdown(f"#### Trace Summary for Batch: `{selected_batch}` ({mat.upper()})")
            
            # Find batch metadata from the tool databases fallback
            try:
                # We can call the tool directly to get a complete structure
                trace_detail = trace_material_batch.invoke({"batch_id": selected_batch})
                comp_detail = verify_traceability_completeness.invoke({"batch_data": trace_detail})
                
                col_tr1, col_tr2, col_tr3 = st.columns(3)
                with col_tr1:
                    status_lbl = comp_detail.get('traceability_status', 'N/A')
                    st.metric("Completeness Status", status_lbl)
                with col_tr2:
                    pct = comp_detail.get('completeness_percent', 0.0)
                    st.metric("Completeness Percentage", f"{pct}%")
                with col_tr3:
                    ready = "READY" if comp_detail.get('eu_battery_passport_ready') else "NON-COMPLIANT"
                    st.metric("EU Battery Passport Ready", ready)
            except Exception as e:
                st.warning(f"Metadata tool resolution: {e}")
                trace_detail = {}
                comp_detail = {}
                
            st.markdown("#### Downstream Trace Pipeline Mappings:")
            for rec in trace_recs:
                cell = rec.get("cell_id")
                pack = rec.get("pack_id")
                vin = rec.get("vehicle_id")
                
                # Fetch vehicle model if pack exists
                model = "N/A"
                if pack:
                    try:
                        pack_map = map_pack_to_vehicle.invoke({"pack_id": pack})
                        model = pack_map.get("vehicle_model", "N/A")
                    except:
                        pass
                
                st.markdown(f"""
                <div class="trace-node">
                    <span class="trace-node-badge">🌍 Sourcing</span> Supplier Material Lot for batch <b>{selected_batch}</b> ({mat})
                </div>
                <div style="margin-left: 25px;" class="trace-node">
                    <span class="trace-node-badge">🧬 Cell</span> Cell Produced: <b>{cell}</b>
                </div>
                <div style="margin-left: 50px;" class="trace-node">
                    <span class="trace-node-badge">📦 Pack</span> Assembled into Battery Pack: <b>{pack or "UNASSIGNED"}</b>
                </div>
                <div style="margin-left: 75px;" class="trace-node">
                    <span class="trace-node-badge">🚗 Vehicle</span> Installed in Vehicle: <b>{vin or "UNASSIGNED"}</b> (Model: {model})
                </div>
                <div style="border-bottom: 1px dashed rgba(255,255,255,0.05); margin: 15px 0;"></div>
                """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 4. Menu Page: Geopolitical Risks
# ---------------------------------------------------------------------------
elif menu_choice == "⚠️ Geopolitical Risks":
    st.markdown("### ⚠️ Geopolitical Sourcing & Material Dependency Exposure")
    
    # Calculate Herfindahl-Hirschman Index (HHI) for the general supply chain
    st.markdown("#### Supplier Sourcing Concentration (HHI Index)")
    st.write("The Herfindahl-Hirschman Index (HHI) measures supplier concentration risk. HHI > 0.3 indicates high geographic concentration, representing single-point-of-failure vulnerabilities.")
    
    # Calculate HHI dynamically from df_chain
    country_counts = df_chain.groupby("country")["supplier_id"].nunique()
    total_sups = country_counts.sum()
    shares = country_counts / total_sups
    hhi = (shares ** 2).sum()
    
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown(f"""
        <div class="card" style="text-align: center;">
            <div class="metric-value" style="color: {'#ef4444' if hhi > 0.4 else '#f59e0b' if hhi > 0.25 else '#10b981'}">{hhi:.4f}</div>
            <div class="metric-label">HHI Sourcing Concentration Index</div>
            <p style="margin-top:10px; color: #94a3b8; font-size:0.9rem;">
                Status: <b>{"HIGH RISK" if hhi > 0.3 else "MODERATE" if hhi > 0.15 else "DIVERSIFIED"}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_g2:
        # Mini bar chart of supplier countries
        fig, ax = plt.subplots(figsize=(6, 2.5))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#0d1117')
        country_counts.plot(kind='barh', color='#3b82f6', ax=ax)
        ax.spines['bottom'].set_color('#475569')
        ax.spines['left'].set_color('#475569')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(colors='#e2e8f0', labelsize=8)
        ax.set_xlabel("Number of Suppliers", color='#e2e8f0', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        
    st.markdown("#### Mineral Geopolitical Exposure Records (`critical_minerals_risk.csv`)")
    st.dataframe(df_risk, use_container_width=True)


# ---------------------------------------------------------------------------
# 5. Menu Page: Quality Analytics
# ---------------------------------------------------------------------------
elif menu_choice == "🔬 Quality Analytics":
    st.markdown("### 🔬 Battery Manufacturing Quality Inspection & Deviation Audits")
    st.write("Examine defect rates across raw material batches, compared against regulatory/internal quality thresholds.")
    
    # Calculate average defect rate
    avg_defect = df_quality["defect_rate"].mean() * 100
    total_defects = df_quality["defects_found"].sum()
    
    col_q1, col_q2 = st.columns(2)
    with col_q1:
        st.markdown(f'<div class="card"><div class="metric-value">{avg_defect:.2f}%</div><div class="metric-label">Average Incoming Defect Rate</div></div>', unsafe_allow_html=True)
    with col_q2:
        st.markdown(f'<div class="card"><div class="metric-value">{total_defects}</div><div class="metric-label">Total Quality Defect Units Found</div></div>', unsafe_allow_html=True)
        
    # Plot Defect Rates per Batch
    st.markdown("#### Quality Metrics by Material Batch")
    
    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    
    x = range(len(df_quality))
    defect_rates_pct = df_quality["defect_rate"] * 100
    
    bars = ax.bar(x, defect_rates_pct, color='#8b5cf6', label="Batch Defect Rate (%)")
    
    # Add a horizontal threshold line (typical 2.0% incoming limit)
    ax.axhline(y=2.0, color='#ef4444', linestyle='--', label="Incoming Quality Threshold (2.0%)")
    
    ax.set_xticks(x)
    ax.set_xticklabels(df_quality["batch_id"], rotation=45, ha='right', color='#e2e8f0', fontsize=8)
    ax.tick_params(colors='#e2e8f0', labelsize=8)
    ax.spines['bottom'].set_color('#475569')
    ax.spines['left'].set_color('#475569')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel("Defect Rate (%)", color='#e2e8f0', fontsize=9)
    ax.legend(facecolor='#0d1117', edgecolor='#475569', labelcolor='#e2e8f0', fontsize=8)
    
    # Highlight bars exceeding the threshold
    for i, bar in enumerate(bars):
        if defect_rates_pct.iloc[i] > 2.0:
            bar.set_color('#ef4444')
            
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("#### Inspection Audit Log Records (`battery_quality.csv`)")
    st.dataframe(df_quality, use_container_width=True)
