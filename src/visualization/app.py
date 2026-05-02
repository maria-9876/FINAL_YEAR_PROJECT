import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Ensure backend imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.env.seaird_env import KeralaCovidEnv
from src.agents.marl_controller import MARLController

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Kerala Command Center", 
    layout="wide", 
    page_icon="🌍",
    initial_sidebar_state="expanded"
)

# Custom CSS for a sophisticated dark-mode aesthetic
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        color: #ffffff;
        font-weight: 300;
        letter-spacing: 1px;
    }
    .metric-box {
        background-color: #1e2530;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #2e3540;
    }
    .nav-header {
        color: #00d4ff;
        font-size: 24px;
        font-weight: bold;
    }
    html {
        scroll-behavior: smooth;
    }
</style>
""", unsafe_allow_html=True)


# ----------------- SESSION STATE & INITIALIZATION -----------------
# We use session_state to hold the editable DataFrames and parameters across pages
if 'app_initialized' not in st.session_state:
    st.session_state.app_initialized = True
    
    # Load defaults
    data_path = os.path.join(os.path.dirname(__file__), '../../data/kerala_districts.json')
    with open(data_path, 'r') as f:
        data = json.load(f)
        
    # Set default District Dataframe
    default_districts = []
    for d in data['districts']:
        default_districts.append({
            "District": d['name'],
            "Population": d['population'],
            "Compliance Rate": d['compliance_rate'],
            "Economic Weight": d['economic_weight'],
            "Total ICU Beds": d['icu_capacity'],
            "Starting Active Cases": int(d['population'] * 0.0001),
            "Starting Vaccines": 1000
        })
    st.session_state.district_df = pd.DataFrame(default_districts)
    
    # Set default disease settings
    st.session_state.disease = {
        "Name": "COVID-19 (Standard)",
        "beta": 0.3, # Transmission
        "alpha": 0.2, # Incubation
        "mu": 0.01, # Mortality
        "sim_days": 100
    }
    
    st.session_state.simulation_done = False
    st.session_state.history_df = pd.DataFrame()
    st.session_state.final_step_df = pd.DataFrame()

@st.cache_data
def load_geo():
    geo_path = os.path.join(os.path.dirname(__file__), '../../data/kerala_geojson.json')
    with open(geo_path, 'r') as f:
        geo = json.load(f)
    return geo

kerala_geo = load_geo()

# Hardcoded approximate district centroids for scatter_mapbox
DISTRICT_CENTROIDS = {
    'Thiruvananthapuram': {'lat': 8.5241, 'lon': 76.9366},
    'Kollam': {'lat': 8.8932, 'lon': 76.6141},
    'Pathanamthitta': {'lat': 9.2648, 'lon': 76.7870},
    'Alappuzha': {'lat': 9.4981, 'lon': 76.3388},
    'Kottayam': {'lat': 9.5916, 'lon': 76.5222},
    'Idukki': {'lat': 9.8500, 'lon': 76.9492},
    'Ernakulam': {'lat': 9.9816, 'lon': 76.2999},
    'Thrissur': {'lat': 10.5276, 'lon': 76.2144},
    'Palakkad': {'lat': 10.7867, 'lon': 76.6548},
    'Malappuram': {'lat': 11.0733, 'lon': 76.0740},
    'Kozhikode': {'lat': 11.2588, 'lon': 75.7804},
    'Wayanad': {'lat': 11.6854, 'lon': 76.1320},
    'Kannur': {'lat': 11.8745, 'lon': 75.3704},
    'Kasaragod': {'lat': 12.4962, 'lon': 74.9869}
}

# ----------------- SIDEBAR NAVIGATION -----------------
st.sidebar.markdown("<div class='nav-header'>Kerala Command</div>", unsafe_allow_html=True)
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    "🏠 Welcome & Overview", 
    "👥 District Demographics", 
    "🧬 Pathogen Parameters", 
    "🚀 Run Simulation"
])


# ==============================================================================
# PAGE 1: 🏠 Welcome & Overview
# ==============================================================================
if page == "🏠 Welcome & Overview":
    st.markdown("<h1 class='main-header'>🏠 KERALA AI PANDEMIC MANAGER</h1>", unsafe_allow_html=True)
    st.markdown("### Welcome to the Multi-Agent Reinforcement Learning Digital Twin.")
    
    col1, col2 = st.columns([6,4])
    with col1:
        st.markdown("""
        #### System Objectives:
        This application serves as a strictly scientific, data-driven sandbox modeling viral outbreaks in Kerala. You have complete control over the demographic starting states and the biological parameters of the disease.
        
        When you run the simulation, **14 independent AI Agents** (one governing each district) will take real-time control of the state. They will communicate through Neural Networks using **Proximal Policy Optimization (PPO)** to balance:
        - 📉 **Infection Containment** (Lockdown Mandates)
        - 🏥 **Hospital Integrity** (Ventilator/ICU availability)
        - 💰 **Economic Growth** (Preventing state bankruptcy)
        
        #### How to use this platform:
        1. Navigate to **👥 District Demographics** to edit populations, ICU beds, and starting case counts.
        2. Navigate to **🧬 Pathogen Parameters** to adjust how lethal or transmissible the disease is.
        3. Navigate to **🚀 Run Simulation** to launch the AI and watch it attempt to save the state structure.
        """)
    with col2:
        st.info("💡 **Architecture Note:** The RL Agents operate using the Decentralized Actor-Centralized Critic (CTDE) paradigm. They only perceive their local district but are evaluated collectively to ensure they don't selfishly crash neighboring economies.")


# ==============================================================================
# PAGE 2: 👥 District Demographics
# ==============================================================================
elif page == "👥 District Demographics":
    st.markdown("<h1 class='main-header'>👥 District Demographics & Resources</h1>", unsafe_allow_html=True)
    st.markdown("Edit the starting dataset for the State of Kerala. You can freely augment populations, adjust compliance rates, and inject massive case influxes into specific districts before launch.")
    
    st.info("📝 **Click any cell in the table below to edit its value.** The AI will instantly read these updated numbers upon simulation execution.")
    
    edited_df = st.data_editor(st.session_state.district_df, use_container_width=True, hide_index=True)
    
    # Save edits back to session state continuously
    st.session_state.district_df = edited_df
    
    st.markdown("---")
    # Quick visual summary of data
    st.subheader("Current Hospital Capacity Spread")
    fig_beds = px.bar(
        edited_df, 
        x="District", 
        y=["Total ICU Beds", "Starting Active Cases"], 
        barmode="group",
        color_discrete_map={"Total ICU Beds": "#00d4ff", "Starting Active Cases": "#ff4b4b"}
    )
    fig_beds.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_beds, use_container_width=True)


# ==============================================================================
# PAGE 3: 🧬 Pathogen Parameters
# ==============================================================================
elif page == "🧬 Pathogen Parameters":
    st.markdown("<h1 class='main-header'>🧬 Pathogen Parameters</h1>", unsafe_allow_html=True)
    st.markdown("Fine-tune the biological realities of the spreading virus.")
    
    col_set, col_info = st.columns([5, 5])
    
    with col_set:
        st.subheader("Preset Contagions")
        disease_type = st.selectbox("Select a base template to auto-fill parameters:", ["COVID-19 (Standard)", "COVID-19 (Delta)", "Measles (High Spread)", "Ebola (High Lethality)"])
        
        if disease_type == "COVID-19 (Standard)":
            base_beta, base_mu = 0.3, 0.01
        elif disease_type == "COVID-19 (Delta)":
            base_beta, base_mu = 0.5, 0.02
        elif disease_type == "Measles (High Spread)":
            base_beta, base_mu = 0.8, 0.005
        else:
            base_beta, base_mu = 0.2, 0.15
            
        st.markdown("---")
        st.subheader("Manual Mathematical Overrides")
        st.session_state.disease["beta"] = st.slider("Transmission Rate (Beta)", 0.0, 1.0, float(base_beta), help="Probability of susceptible individual catching the disease upon contact.")
        st.session_state.disease["mu"] = st.slider("Base Mortality Rate (Mu)", 0.0, 0.5, float(base_mu), help="Baseline chance of death if infected. This will automatically triple if ICU beds overflow.")
        st.session_state.disease["sim_days"] = st.slider("Simulation Horizon (Days)", 30, 365, st.session_state.disease["sim_days"])
        st.session_state.disease["Name"] = disease_type
        
    with col_info:
        st.success("🔬 **SEAIRD Model Integration**")
        st.markdown(f"""
        Your settings are actively passed into the differential **SEAIRD** equations:
        - **Susceptible → Exposed:** Driven by Beta (**{st.session_state.disease["beta"]:.2f}**)
        - **Infected → Deceased:** Driven by Mu (**{st.session_state.disease["mu"]:.3f}**)
        
        *Note: The AI agents do not know these mathematical truths natively. They must learn the transmission rates dynamically through trial and error over thousands of simulated epochs.*
        """)


# ==============================================================================
# PAGE 4: 🚀 Run Simulation
# ==============================================================================
elif page == "🚀 Run Simulation":
    st.markdown("<h1 class='main-header'>🚀 Simulation Command Terminal</h1>", unsafe_allow_html=True)
    
    # Pre-flight Check
    st.markdown("### Pre-Flight Checklist")
    st.write(f"🔬 **Pathogen Profile:** {st.session_state.disease['Name']} (Beta: {st.session_state.disease['beta']}, Mu: {st.session_state.disease['mu']})")
    st.write(f"📅 **Time Horizon:** {st.session_state.disease['sim_days']} predicted days")
    
    run_sim = st.button("EXECUTE NEURAL SIMULATION", type="primary", use_container_width=True)
    
    # ----------------- MAIN PROCESSING LOGIC -----------------
    if run_sim:
        with st.spinner('Engaging Multi-Agent Neural Networks...'):
            sim_days = st.session_state.disease["sim_days"]
            env = KeralaCovidEnv(max_steps=sim_days)
            observations, _ = env.reset()
            
            # Apply user overrides from District Demographics Page
            d_df = st.session_state.district_df
            for idx, row in d_df.iterrows():
                dist = row['District']
                env.state[dist]['I'] = row['Starting Active Cases']
                env.state[dist]['vaccines_avail'] = row['Starting Vaccines']
                env.district_data[dist]['icu_capacity'] = row['Total ICU Beds']
                env.district_data[dist]['population'] = row['Population']
                env.district_data[dist]['compliance_rate'] = row['Compliance Rate']
                env.district_data[dist]['economic_weight'] = row['Economic Weight']
                
            # Apply Pathogen Overrides
            for agent in env.agents:
                env.dynamics_models[agent].beta = st.session_state.disease["beta"]
                env.dynamics_models[agent].mu = st.session_state.disease["mu"]
                
            controller = MARLController(env.agents, 10, 4)
            history = []
            
            for step in range(sim_days):
                actions, _, _ = controller.get_actions(observations)
                observations, _, _, _, _ = env.step(actions)
                
                for agent in env.agents:
                    history.append({
                        "Step": step,
                        "District": agent,
                        "Infected": env.state[agent]['I'],
                        "ICU Occupancy": env.state[agent]['icu_occupancy'],
                        "ICU Limit": env.district_data[agent]['icu_capacity'],
                        "Economy Health": env.state[agent]['economic_health'],
                        "Lockdown Level": int(actions[agent][0] * 3.99),
                        "Vaccine Stock": env.state[agent]['vaccines_avail'],
                        "Lat": DISTRICT_CENTROIDS[agent]['lat'],
                        "Lon": DISTRICT_CENTROIDS[agent]['lon']
                    })
                    
            # Cache results aggressively
            h_df = pd.DataFrame(history)
            max_step = h_df['Step'].max()
            f_df = h_df[h_df['Step'] == max_step].copy()
            
            # Format strings for visualization
            f_df['Lockdown Policy'] = f_df['Lockdown Level'].apply(lambda x: f"Level {x}")
            f_df.rename(columns={
                'Infected': 'Active Cases',
                'ICU Occupancy': 'Beds Used',
                'Vaccine Stock': 'Remaining Vaccines'
            }, inplace=True)
            
            st.session_state['history_df'] = h_df
            st.session_state['final_step_df'] = f_df
            st.session_state['simulation_done'] = True
            st.success("Simulation Sequence Complete! Dashboard Active.")


    # ----------------- DASHBOARD RENDER -----------------
    if not st.session_state.get('simulation_done', False):
        st.info("System standing by. Awaiting execution command.")
    else:
        st.markdown("---")
        df = st.session_state['history_df']
        final_df = st.session_state['final_step_df']
        
        # Render Tabs
        tab1, tab2, tab3 = st.tabs(["📊 Executive Overview", "🗺️ Geospatial Intelligence", "🚑 Logistics & Transfers"])
        
        # --- TAB 1: OVERVIEW ---
        with tab1:
            st.markdown("### 📈 System Diagnostics")
            curr_cases = int(final_df['Active Cases'].sum())
            curr_beds = int(final_df['Beds Used'].sum())
            avg_econ = final_df['Economy Health'].mean() * 100
            
            kpi1, kpi2, kpi3 = st.columns(3)
            with kpi1:
                st.markdown(f"<div class='metric-box'><h3>Total Active Cases</h3><h2>{curr_cases:,}</h2></div>", unsafe_allow_html=True)
            with kpi2:
                st.markdown(f"<div class='metric-box'><h3>Total ICU Resources Engaged</h3><h2>{curr_beds:,}</h2></div>", unsafe_allow_html=True)
            with kpi3:
                st.markdown(f"<div class='metric-box'><h3>State Economic Retention</h3><h2>{avg_econ:.1f}%</h2></div>", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            col_curve, col_table = st.columns([5, 5])
            with col_curve:
                st.subheader("📉 Outbreak Curve Projection")
                agg_infections = df.groupby('Step')['Infected'].sum().reset_index()
                fig_curve = px.line(agg_infections, x='Step', y='Infected', color_discrete_sequence=['#00d4ff'])
                fig_curve.update_layout(margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", xaxis_title="Days", yaxis_title="Active Cases")
                st.plotly_chart(fig_curve, use_container_width=True)
                
            with col_table:
                st.subheader("📋 Final Agent Directives")
                display_df = final_df[['District', 'Lockdown Policy', 'Active Cases']].copy()
                display_df['Active Cases'] = display_df['Active Cases'].astype(int)
                st.dataframe(display_df, use_container_width=True, hide_index=True)

        # --- TAB 2: GEOSPATIAL MAP ---
        with tab2:
            st.markdown("### 🛰️ Live Pathogen Density Tracker")
            st.markdown("Bubble size reflects outbreak scale. Hover over nodes to review containment metrics.")
            
            # Create sophisticated Bubble Map using scatter_mapbox
            fig_map = px.scatter_mapbox(
                final_df, 
                lat="Lat", 
                lon="Lon", 
                size="Active Cases", 
                color="Lockdown Policy",
                hover_name="District",
                hover_data=["Active Cases", "Beds Used", "Remaining Vaccines"],
                color_discrete_map={"Level 0": "#00ff00", "Level 1": "#ffff00", "Level 2": "#ff9900", "Level 3": "#ff0000"},
                mapbox_style="carto-darkmatter",
                zoom=6.0,
                center={"lat": 10.5, "lon": 76.5},
                size_max=50,
                opacity=0.7
            )
            fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_map, use_container_width=True, height=600)

        # --- TAB 3: LOGISTICS & TRANSFERS ---
        with tab3:
            st.markdown("### 🚑 Emergency Supply Chain & Logistics Network")
            st.markdown("Automated AI tracking of District-level surplus and deficits to prevent hospital collapse.")
            
            # Calculate Oxygen and Bed metrics
            logistics_df = final_df.copy()
            logistics_df['Bed Capacity (%)'] = (logistics_df['Beds Used'] / logistics_df['ICU Limit']) * 100
            logistics_df['O2 Cylinders Required'] = logistics_df['Beds Used'] * 2
            logistics_df['O2 Cylinders Stock'] = logistics_df['ICU Limit'] * 2
            logistics_df['O2 Surplus/Deficit'] = logistics_df['O2 Cylinders Stock'] - logistics_df['O2 Cylinders Required']
            
            st.markdown("#### 1. Real-Time Resource Status")
            # Format the dataframe for display
            display_log = logistics_df[['District', 'Bed Capacity (%)', 'Beds Used', 'ICU Limit', 'O2 Surplus/Deficit']].copy()
            display_log['Bed Capacity (%)'] = display_log['Bed Capacity (%)'].apply(lambda x: f"{x:.1f}%")
            display_log['Beds Used'] = display_log['Beds Used'].astype(int)
            display_log['ICU Limit'] = display_log['ICU Limit'].astype(int)
            display_log['O2 Surplus/Deficit'] = display_log['O2 Surplus/Deficit'].astype(int)
            st.dataframe(display_log.style.applymap(lambda x: 'color: red;' if float(x.strip('%')) > 90 else 'color: green;' if isinstance(x, str) and '%' in x else '', subset=['Bed Capacity (%)']), use_container_width=True, hide_index=True)
            
            # Calculate Transfers
            st.markdown("#### 2. AI Transfer Recommendations")
            critical_districts = logistics_df[logistics_df['Beds Used'] >= logistics_df['ICU Limit'] * 0.90]
            safe_districts = logistics_df[logistics_df['Beds Used'] < logistics_df['ICU Limit'] * 0.50]
            
            if critical_districts.empty:
                st.success("✅ **STATUS OPTIMAL:** No districts are currently forecasting resource exhaustion. Health network is stable.")
            else:
                for _, c_row in critical_districts.iterrows():
                    shortfall_beds = int(c_row['Beds Used'] - (c_row['ICU Limit'] * 0.90)) # Try to bring them back to 90%
                    st.markdown(f"<h4 style='color: #ff4b4b;'>🚨 {c_row['District']} OVERFLOW DETECTED</h4>", unsafe_allow_html=True)
                    
                    if not safe_districts.empty:
                        # Sort safe districts by most surplus
                        safe_districts['Surplus'] = safe_districts['ICU Limit'] - safe_districts['Beds Used']
                        donor = safe_districts.sort_values(by='Surplus', ascending=False).iloc[0]
                        
                        st.info(f"👉 **EXECUTE TRANSFER:** Dispatch **{shortfall_beds} ICU Beds & {shortfall_beds*2} O2 Cylinders** from **{donor['District']}** immediately.")
                    else:
                        st.error("💀 **CRITICAL FAILURE:** Entire state network is overwhelmed. No safe donor districts available.")

