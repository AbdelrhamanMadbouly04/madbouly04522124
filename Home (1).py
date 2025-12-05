import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.integrate import odeint
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
from reportlab.lib import colors
import streamlit.components.v1 as components
import os

# --- 0. Disable Developer Hotkeys & Menu ---
def kill_developer_hotkeys():
    if not os.path.exists(".streamlit"):
        os.makedirs(".streamlit")
    
    config_content = """
[client]
toolbarMode = "viewer"
showErrorDetails = false
[ui]
hideTopBar = true
"""
    try:
        with open(".streamlit/config.toml", "w") as f:
            f.write(config_content)
    except:
        pass

kill_developer_hotkeys()

# --- 1. Constants ---
R_GAS = 8.314
CP_BIOMASS = 1300.0
CP_WATER = 4180.0
H_VAPOR = 2260000.0
TEMP_REF_K = 298.15
HHV_INITIAL = {"Wood": 18.0, "Agricultural Waste": 16.5, "Municipal Waste": 15.0}
HHV_ENRICHMENT_FACTOR = 1.3
KINETICS = {"Hemicellulose": [1.5e10, 110000], "Cellulose": [1.0e12, 130000], "Lignin": [2.0e9, 100000]}
BIOMASS_COMPOSITION = {
    "Wood": {"Hemicellulose": 0.35, "Cellulose": 0.45, "Lignin": 0.20, "Ash": 0.02, "Gas_Factor": 0.40},
    "Agricultural Waste": {"Hemicellulose": 0.45, "Cellulose": 0.35, "Lignin": 0.20, "Ash": 0.08, "Gas_Factor": 0.50},
    "Municipal Waste": {"Hemicellulose": 0.30, "Cellulose": 0.40, "Lignin": 0.30, "Ash": 0.15, "Gas_Factor": 0.60}
}
DRYING_RATE_CONST = 0.05
SIZE_FACTOR = {"Fine (<1mm)": 1.0, "Medium (1-5mm)": 0.85, "Coarse (>5mm)": 0.65}
BASE_FC_FACTOR = 0.20

# --- 2. Styles (Black & Burgundy) ---
GLOBAL_CSS = """
<style>
    /* Theme: Black & Burgundy */
    .stApp { background-color: #000000; color: #f0f0f0; font-family: 'Segoe UI', sans-serif; }
    h1, h2, h3, h4, h5, h6 { color: #ffffff !important; }
    .stMarkdown, p, label { color: #e0e0e0 !important; }
    
    section[data-testid="stSidebar"] { background-color: #640d14; }
    section[data-testid="stSidebar"] * { color: #ffffff !important; }
    
    .stSlider > div > div > div > div { background-color: #640d14 !important; }
    .stSelectbox > div > div { color: #ffffff; }

    div[data-testid="stMetric"] {
        background-color: #121212; border: 2px solid #640d14;
        border-radius: 8px; padding: 10px; box-shadow: 0px 4px 10px rgba(100, 13, 20, 0.4);
    }
    div[data-testid="stMetricValue"] { color: #ffffff !important; font-size: 24px !important; }
    div[data-testid="stMetricLabel"] { color: #d97584 !important; font-size: 14px !important; }

    .header-box {
        background: #000000; border: 1px solid #640d14; padding: 20px;
        border-radius: 10px; color: #ffffff; text-align: center;
        margin-bottom: 20px; box-shadow: 0 4px 6px rgba(100, 13, 20, 0.3);
    }
    .header-box h1 { color: #ffffff !important; margin: 0; }
    .header-box p { color: #cccccc !important; margin: 0; }

    div[data-testid="stTabs"] button { color: #cccccc !important; font-weight: bold; background: transparent !important; }
    div[data-testid="stTabs"] button[aria-selected="true"] { color: #ffffff !important; border-bottom: 3px solid #640d14 !important; }
    
    .stButton > button { background-color: #640d14 !important; color: #ffffff !important; border: 1px solid #801119; border-radius: 6px; }
    .stButton > button:hover { background-color: #801119 !important; }

    .bfd-block {
        padding: 10px; border-radius: 8px; text-align: center; background: #121212;
        border: 1px solid #640d14; color: #ffffff; font-weight: bold; font-size: 0.9em;
    }
    .bfd-stream { color: #640d14; font-size: 20px; padding-top: 10px; text-align: center; }
    
    .streamlit-expanderHeader { color: #ffffff !important; background-color: #4a090e !important; border-radius: 5px; }

    #MainMenu, footer, header, .stDeployButton {visibility: hidden;}
</style>
"""

# --- 3. Simulation Logic ---
def simulate_torrefaction(biomass, moisture, temp_C, duration_min, size, initial_mass_kg, reactor_type="N/A"):
    temp_K = temp_C + 273.15
    comp = BIOMASS_COMPOSITION.get(biomass)
    R_GAS_LOCAL = R_GAS 
    initial_moisture_frac = moisture / 100
    initial_ash_frac = comp["Ash"]
    daf_frac = 1.0 - initial_moisture_frac - initial_ash_frac
    m_h_init = comp["Hemicellulose"] * daf_frac
    m_c_init = comp["Cellulose"] * daf_frac
    m_l_init = comp["Lignin"] * daf_frac
    initial_mass_fixed_carbon_daf = daf_frac * BASE_FC_FACTOR 
    k_drying = DRYING_RATE_CONST * SIZE_FACTOR.get(size)
    size_factor_val = SIZE_FACTOR.get(size)
    k_h_eff = KINETICS["Hemicellulose"][0] * np.exp(-KINETICS["Hemicellulose"][1] / (R_GAS_LOCAL * temp_K)) * size_factor_val
    k_c_eff = KINETICS["Cellulose"][0] * np.exp(-KINETICS["Cellulose"][1] / (R_GAS_LOCAL * temp_K)) * size_factor_val
    k_l_eff = KINETICS["Lignin"][0] * np.exp(-KINETICS["Lignin"][1] / (R_GAS_LOCAL * temp_K)) * size_factor_val

    def model(y, t, k_dry, kh, kc, kl):
        m_moist, m_h, m_c, m_l = y
        d_moist = -k_dry * m_moist if m_moist > 0.001 else 0 
        d_h = -kh * m_h
        d_c = -kc * m_c
        d_l = -kl * m_l
        return [d_moist, d_h, d_c, d_l]
    
    t = np.linspace(0, duration_min, 100)
    y0 = [initial_moisture_frac, m_h_init, m_c_init, m_l_init]
    sol = odeint(model, y0, t, args=(k_drying, k_h_eff, k_c_eff, k_l_eff))
    sol[sol < 0] = 0
    
    final_moisture_remaining = sol[:, 0][-1]
    final_h_remaining = sol[:, 1][-1]
    final_c_remaining = sol[:, 2][-1]
    final_l_remaining = sol[:, 3][-1]
    lost_h_frac = m_h_init - final_h_remaining
    lost_c_frac = m_c_init - final_c_remaining
    lost_l_frac = m_l_init - final_l_remaining
    total_volatiles_lost_frac = lost_h_frac + lost_c_frac + lost_l_frac
    mass_ash_kg = initial_mass_kg * initial_ash_frac
    mass_fixed_carbon_kg = initial_mass_kg * initial_mass_fixed_carbon_daf
    mass_remaining_components = (final_h_remaining + final_c_remaining + final_l_remaining) * initial_mass_kg
    mass_biochar_total = mass_fixed_carbon_kg + mass_remaining_components + mass_ash_kg
    final_solid_yield_percent = (mass_biochar_total / initial_mass_kg) * 100
    mass_moisture_loss_kg = (initial_moisture_frac - final_moisture_remaining) * initial_mass_kg
    mass_non_condensable_gas_kg = total_volatiles_lost_frac * initial_mass_kg * comp["Gas_Factor"] 
    mass_bio_oil_kg = total_volatiles_lost_frac * initial_mass_kg * (1 - comp["Gas_Factor"]) 

    yields_percent = pd.DataFrame({"Yield (%)": [final_solid_yield_percent, (mass_moisture_loss_kg / initial_mass_kg) * 100, (mass_bio_oil_kg / initial_mass_kg) * 100, (mass_non_condensable_gas_kg / initial_mass_kg) * 100]}, index=["Biochar", "Water Vapor", "Bio-Oil", "Gases"])
    yields_mass = pd.DataFrame({"Mass (kg)": [mass_biochar_total, mass_moisture_loss_kg, mass_bio_oil_kg, mass_non_condensable_gas_kg]}, index=["Biochar", "Water Vapor", "Bio-Oil", "Gases"])
    solid_composition = pd.DataFrame({"Mass (kg)": [mass_fixed_carbon_kg, mass_remaining_components, mass_ash_kg]}, index=["Fixed Carbon", "Volatile Matter", "Ash"])
    final_ash_percent = (mass_ash_kg / mass_biochar_total) * 100
    initial_hhv_mj_kg = HHV_INITIAL.get(biomass, 17.0) 
    biochar_hhv_mj_kg = initial_hhv_mj_kg * HHV_ENRICHMENT_FACTOR 
    initial_energy_mj = initial_mass_kg * initial_hhv_mj_kg * (1 - initial_moisture_frac)
    final_biochar_energy_mj = mass_biochar_total * biochar_hhv_mj_kg
    energy_yield_percent = (final_biochar_energy_mj / initial_energy_mj) * 100
    
    return {
        "yields_percent": yields_percent, "yields_mass": yields_mass, "solid_composition": solid_composition,
        "final_ash_percent": final_ash_percent, "initial_hhv": initial_hhv_mj_kg,
        "biochar_hhv": biochar_hhv_mj_kg, "energy_yield_percent": energy_yield_percent,
        "parameters": {"biomass": biomass, "moisture": moisture, "temperature": temp_C, "duration": duration_min, "size": size, "initial_mass": initial_mass_kg, "reactor": reactor_type},
        "mass_moisture_loss_kg": mass_moisture_loss_kg, "mass_dry_biomass_kg": initial_mass_kg * daf_frac, 
    }

def calculate_thermal_balance(p, results):
    T_K = p['temperature'] + 273.15
    Q_sensible_biomass = (p['initial_mass'] * (1 - p['moisture']/100) * CP_BIOMASS * (T_K - TEMP_REF_K)) / 1000
    Q_sensible_water = (p['initial_mass'] * (p['moisture']/100) * CP_WATER * (T_K - TEMP_REF_K)) / 1000
    Q_latent_water = (results.get('mass_moisture_loss_kg', 0.0) * H_VAPOR) / 1000
    Q_torrefaction = results.get('mass_dry_biomass_kg', 0.0) * 100.0
    Q_total_required_kJ = Q_sensible_biomass + Q_sensible_water + Q_latent_water + Q_torrefaction
    return {'Q_total_required_kJ': Q_total_required_kJ, 'Q_latent_water': Q_latent_water, 'Q_total_per_kg': Q_total_required_kJ / p['initial_mass']}

@st.cache_data
def run_sensitivity_analysis(biomass, moisture, size, initial_mass_kg, reactor_type):
    T_range = np.linspace(220, 320, 8)
    D_range = np.linspace(20, 90, 8)
    results_T = [simulate_torrefaction(biomass, moisture, T, 60, size, initial_mass_kg, reactor_type)["yields_percent"].iloc[0,0] for T in T_range]
    results_D = [simulate_torrefaction(biomass, moisture, 275, D, size, initial_mass_kg, reactor_type)["yields_percent"].iloc[0,0] for D in D_range]
    return pd.DataFrame({"Temp": T_range, "Yield": results_T}), pd.DataFrame({"Duration": D_range, "Yield": results_D})

def create_pdf(results, thermal, profit):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph("Chemisco Simulation Report", styles['Title']), Spacer(1, 12)]
    data = [["Metric", "Value"], ["Yield", f"{results['yields_percent'].iloc[0,0]:.1f}%"], ["Energy Eff.", f"{results['energy_yield_percent']:.1f}%"], ["Profit", f"${profit:.2f}"]]
    t = Table(data, colWidths=[200, 200])
    t.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor('#640d14')), ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke), ('GRID', (0,0), (-1,-1), 1, colors.black)]))
    story.append(t)
    doc.build(story)
    buffer.seek(0)
    return buffer

# --- 5. Main App ---
def main():
    st.set_page_config(page_title="Chemisco Pro", layout="wide", initial_sidebar_state="expanded")
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    
    # *** Inject Botpress Globally ***
    js_code = """
    <script>
        if (!window.parent.document.getElementById('botpress-inject')) {
            var script1 = window.parent.document.createElement('script');
            script1.id = 'botpress-inject';
            script1.src = 'https://cdn.botpress.cloud/webchat/v3.4/inject.js';
            window.parent.document.head.appendChild(script1);
            
            script1.onload = function() {
                var script2 = window.parent.document.createElement('script');
                script2.src = 'https://files.bpcontent.cloud/2025/11/28/23/20251128230307-F5JAD1ML.js';
                script2.defer = true;
                window.parent.document.body.appendChild(script2);
            };
        }
    </script>
    """
    components.html(js_code, height=0, width=0)

    if 'target_yield' not in st.session_state: st.session_state.update({'target_yield': 75, 'cost_biomass': 30.0, 'cost_energy': 5.0, 'price_char': 1.20})

    with st.sidebar:
        st.markdown("""<div class="header-box"><h1>CHEMISCO</h1><p>Torrefaction Simulator</p></div>""", unsafe_allow_html=True)
        st.header("⚙️ Inputs")
        reactor = st.selectbox("Reactor", ["Rotary Drum", "Fluidized Bed", "Screw Reactor", "Fixed Bed"])
        with st.expander("🌲 Feedstock", expanded=True):
            mass = st.number_input("Mass (kg)", 1.0, 1000.0, 100.0, 10.0)
            biomass = st.selectbox("Type", list(BIOMASS_COMPOSITION.keys()))
            moisture = st.slider("Moisture (%)", 0.0, 50.0, 10.0)
            size = st.selectbox("Size", list(SIZE_FACTOR.keys()))
        with st.expander("🔥 Process", expanded=True):
            temp = st.slider("Temp (°C)", 200, 350, 275)
            time_min = st.slider("Time (min)", 10, 120, 45)
        with st.expander("💰 Economics", expanded=False):
            st.session_state.cost_biomass = st.number_input("Feed ($/ton)", value=st.session_state.cost_biomass)
            st.session_state.cost_energy = st.number_input("Energy ($/hr)", value=st.session_state.cost_energy)
            st.session_state.price_char = st.number_input("Price ($/kg)", value=st.session_state.price_char)
        game_mode = st.checkbox("🎮 Plant Manager Mode")

    res = simulate_torrefaction(biomass, moisture, temp, time_min, size, mass, reactor)
    therm = calculate_thermal_balance(res['parameters'], res)
    cost_feed = (mass/1000)*st.session_state.cost_biomass
    cost_ops = (time_min/60)*st.session_state.cost_energy
    rev = res['yields_mass'].iloc[0,0] * st.session_state.price_char
    profit = rev - (cost_feed + cost_ops)

    st.title("CHEMISCO: Advanced Dashboard")
    st.markdown("---")
    
    c1, c2, c3, c4, c5 = st.columns([1.5, 0.5, 1.5, 0.5, 1.5])
    with c1: st.markdown(f'<div class="bfd-block">FEED<br>{mass} kg</div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="bfd-stream">➜</div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="bfd-block">{reactor.upper()}<br>{temp}°C | {time_min}min</div>', unsafe_allow_html=True)
    with c4: st.markdown('<div class="bfd-stream">➜</div>', unsafe_allow_html=True)
    with c5: st.markdown(f'<div class="bfd-block">BIOCHAR<br>{res["yields_mass"].iloc[0,0]:.1f} kg</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Mass Yield", f"{res['yields_percent'].iloc[0,0]:.1f}%", f"{res['yields_mass'].iloc[0,0]:.1f} kg")
    k2.metric("Energy Yield", f"{res['energy_yield_percent']:.1f}%", f"HHV: {res['biochar_hhv']:.1f}")
    k3.metric("Heat Req", f"{therm['Q_total_per_kg']:.0f} kJ/kg", "Thermal Load")
    k4.metric("Profit", f"${profit:.2f}", "Net Balance")
    
    st.markdown("---")

    # --- TABS (Removed AI Expert Tab) ---
    t1, t2, t3, t4 = st.tabs(["📊 Charts", "🌡️ Sensitivity", "📄 Report", "🎮 Game"])
    
    color_scale = ["#640d14", "#8c2f39", "#b3525e", "#d97584"]
    plot_bg = '#000000'
    txt_col = '#ffffff'
    
    with t1:
        cc1, cc2 = st.columns(2)
        with cc1:
            st.subheader("Product Distribution")
            fig = px.pie(res['yields_percent'].reset_index(), values='Yield (%)', names='index', color_discrete_sequence=color_scale, hole=0.4)
            fig.update_layout(paper_bgcolor=plot_bg, plot_bgcolor=plot_bg, font_color=txt_col)
            st.plotly_chart(fig, use_container_width=True)
        with cc2:
            st.subheader("Solid Composition")
            fig2 = px.bar(res['solid_composition'].reset_index(), x='index', y='Mass (kg)', color='index', color_discrete_sequence=color_scale)
            fig2.update_layout(paper_bgcolor=plot_bg, plot_bgcolor=plot_bg, font_color=txt_col, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    with t2:
        df_T, df_D = run_sensitivity_analysis(biomass, moisture, size, mass, reactor)
        c_sens1, c_sens2 = st.columns(2)
        with c_sens1:
            fig_t = px.line(df_T, x="Temp", y="Yield", title="Yield vs Temp", markers=True)
            fig_t.update_traces(line_color="#d97584")
            fig_t.update_layout(paper_bgcolor=plot_bg, plot_bgcolor=plot_bg, font_color=txt_col)
            st.plotly_chart(fig_t, use_container_width=True)
        with c_sens2:
            fig_d = px.line(df_D, x="Duration", y="Yield", title="Yield vs Time", markers=True)
            fig_d.update_traces(line_color="#d97584")
            fig_d.update_layout(paper_bgcolor=plot_bg, plot_bgcolor=plot_bg, font_color=txt_col)
            st.plotly_chart(fig_d, use_container_width=True)

    with t3:
        st.write("Generate official technical PDF report.")
        pdf = create_pdf(res, therm, profit)
        st.download_button("Download PDF", pdf, "report.pdf", "application/pdf")

    with t4:
        if game_mode:
            st.info("🎯 Target: Yield > 80% AND Energy Eff > 90%")
            if res['yields_percent'].iloc[0,0] > 80 and res['energy_yield_percent'] > 90:
                st.balloons()
                st.success("🏆 You Won! Perfect Optimization.")
            else:
                st.warning(f"Current: Yield {res['yields_percent'].iloc[0,0]:.1f}%, Energy {res['energy_yield_percent']:.1f}%")
        else:
            st.write("Activate Game Mode in Sidebar.")

if __name__ == "__main__":
    main()
