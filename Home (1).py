import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
from reportlab.lib import colors
import streamlit.components.v1 as components
import math

# --- 1. Constants & Defaults ---
R_GAS = 8.314
CP_BIOMASS = 1500.0  # J/kg.K
CP_WATER = 4180.0    # J/kg.K
H_VAPOR = 2260000.0  # J/kg (Latent heat)
TEMP_REF_K = 298.15
HHV_DRY_INITIAL_DEFAULT = 18.0 # MJ/kg

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

    /* Hide Hamburger Menu & Footer only - Header stays visible for sidebar toggle */
    #MainMenu, footer, .stDeployButton {visibility: hidden;}
</style>
"""

# --- 3. Mathematical Models (Corrected Logic) ---

def moisture_evap_linear(initial_moisture_kg, T_C, t_min, k_f=0.02):
    """(A) Linear moisture evaporation model (starts > 100°C)"""
    if T_C <= 100:
        return 0.0
    evap_kg = k_f * (T_C - 100) * t_min * initial_moisture_kg
    return min(initial_moisture_kg, max(0.0, evap_kg))

def Y_solid_empirical(T_C, t_min, a=0.35, b=0.004):
    """(C) Empirical solid yield loss model"""
    severity = max(0.0, T_C - 200) * t_min
    expo = math.exp(-b * severity)
    return 1.0 - a * (1.0 - expo)

def m_oil(dry_mass_kg, T_C, t_min, C_oil=0.25):
    """(D) Bio-oil production"""
    k_oil = 0.0008 * max(0.0, T_C - 200)
    return dry_mass_kg * C_oil * (1.0 - math.exp(-k_oil * t_min))

def m_gas(dry_mass_kg, T_C, t_min, C_gas=0.20):
    """(D) Gas production"""
    k_gas = 0.0015 * max(0.0, T_C - 180)
    return dry_mass_kg * C_gas * (1.0 - math.exp(-k_gas * t_min))

def hh_increase_fraction(Y_solid):
    """(F) HHV increase"""
    return 0.25 * (1.0 - Y_solid)

def run_simulation(mass_in, moisture_pct, ash_pct_dry, temp_c, time_min, params):
    moisture_frac = moisture_pct / 100.0
    ash_frac_dry = ash_pct_dry / 100.0
    
    M0_water = mass_in * moisture_frac
    M0_dry = mass_in * (1.0 - moisture_frac)
    M_ash = M0_dry * ash_frac_dry
    
    # Calculations
    w_evap = moisture_evap_linear(M0_water, temp_c, time_min, k_f=params['k_f'])
    w_remaining = M0_water - w_evap
    
    oil_kg = m_oil(M0_dry, temp_c, time_min, C_oil=params['C_oil'])
    gas_kg = m_gas(M0_dry, temp_c, time_min, C_gas=params['C_gas'])
    
    # Mass Balance
    char_dry = max(0, M0_dry - oil_kg - gas_kg) 
    char_total_mass = char_dry + w_remaining
    
    # Energy
    y_solid_val = Y_solid_empirical(temp_c, time_min, a=params['a_solid'], b=params['b_solid'])
    hhv_inc_frac = hh_increase_fraction(y_solid_val)
    hhv_final = HHV_DRY_INITIAL_DEFAULT * (1.0 + hhv_inc_frac)
    
    energy_in = M0_dry * HHV_DRY_INITIAL_DEFAULT
    energy_out = char_dry * hhv_final
    energy_yield = (energy_out / energy_in) * 100 if energy_in > 0 else 0
    
    # Thermal Load
    T_K = temp_c + 273.15
    Q_sensible_bio = (M0_dry * CP_BIOMASS * (T_K - TEMP_REF_K)) / 1000 
    Q_sensible_water = (M0_water * CP_WATER * (373.15 - TEMP_REF_K)) / 1000 
    Q_latent = (w_evap * H_VAPOR) / 1000 
    Q_total_kJ = Q_sensible_bio + Q_sensible_water + Q_latent
    
    return {
        "mass_in": mass_in,
        "char_kg": char_total_mass,
        "water_evap_kg": w_evap,
        "oil_kg": oil_kg,
        "gas_kg": gas_kg,
        "ash_kg": M_ash,
        "hhv_final": hhv_final,
        "mass_yield_pct": (char_total_mass / mass_in) * 100,
        "energy_yield_pct": energy_yield,
        "hhv_increase_pct": hhv_inc_frac * 100,
        "Q_total_kJ": Q_total_kJ,
        "params": params
    }

def get_time_series(mass_in, moisture_pct, ash_pct_dry, temp_c, time_min, params):
    times = np.linspace(0, time_min, 50)
    data = []
    for t in times:
        res = run_simulation(mass_in, moisture_pct, ash_pct_dry, temp_c, t, params)
        data.append({
            "Time (min)": t,
            "Char (kg)": res['char_kg'],
            "Bio-Oil (kg)": res['oil_kg'],
            "Gases (kg)": res['gas_kg'],
            "Water Vapor (kg)": res['water_evap_kg'],
            "HHV Increase (%)": res['hhv_increase_pct']
        })
    return pd.DataFrame(data)

def create_pdf(res, profit):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph("Chemisco Simulation Report", styles['Title']), Spacer(1, 12)]
    data = [
        ["Metric", "Value"], 
        ["Mass Yield", f"{res['mass_yield_pct']:.1f}%"], 
        ["Final HHV", f"{res['hhv_final']:.2f} MJ/kg"],
        ["Energy Yield", f"{res['energy_yield_pct']:.1f}%"],
        ["Bio-Oil Produced", f"{res['oil_kg']:.2f} kg"],
        ["Profit Est.", f"${profit:.2f}"]
    ]
    t = Table(data, colWidths=[200, 200])
    t.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor('#640d14')), 
                           ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke), 
                           ('GRID', (0,0), (-1,-1), 1, colors.black)]))
    story.append(t)
    doc.build(story)
    buffer.seek(0)
    return buffer

# --- 4. Main Streamlit App ---
def main():
    st.set_page_config(page_title="Chemisco Pro", layout="wide", initial_sidebar_state="expanded")
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    
    # --- OPTIONAL: Botpress Inject (Commented out to prevent loading lag) ---
    # Uncomment lines below if you need the chatbot back
    # js_code = """<script>if(!window.parent.document.getElementById('botpress-inject')){var s=window.parent.document.createElement('script');s.id='botpress-inject';s.src='https://cdn.botpress.cloud/webchat/v3.4/inject.js';window.parent.document.head.appendChild(s);s.onload=function(){var s2=window.parent.document.createElement('script');s2.src='https://files.bpcontent.cloud/2025/11/28/23/20251128230307-F5JAD1ML.js';s2.defer=true;window.parent.document.body.appendChild(s2);}}</script>"""
    # components.html(js_code, height=0, width=0)

    if 'cost_biomass' not in st.session_state: 
        st.session_state.update({'cost_biomass': 30.0, 'cost_energy': 0.15, 'price_char': 1.20})

    # --- Sidebar ---
    with st.sidebar:
        st.markdown("""<div class="header-box"><h1>CHEMISCO</h1><p>Torrefaction Simulator</p></div>""", unsafe_allow_html=True)
        st.header("⚙️ Inputs")
        reactor = st.selectbox("Reactor Type", ["Rotary Drum", "Fluidized Bed", "Screw Reactor"])
        
        with st.expander("🌲 Feedstock", expanded=True):
            mass = st.number_input("Mass (kg)", 1.0, 10000.0, 100.0, 10.0)
            moisture = st.slider("Moisture (%)", 0.0, 60.0, 15.0)
            ash = st.slider("Ash (Dry Basis %)", 0.0, 30.0, 5.0)
            
        with st.expander("🔥 Process", expanded=True):
            temp = st.slider("Temp (°C)", 150, 350, 275)
            time_min = st.slider("Time (min)", 10, 120, 30)

        with st.expander("🔧 Advanced Model Params", expanded=False):
            p_kf = st.number_input("Drying rate (k_f)", 0.0, 0.1, 0.02, format="%.3f")
            p_Coil = st.number_input("Max Oil frac (C_oil)", 0.0, 0.5, 0.25)
            p_Cgas = st.number_input("Max Gas frac (C_gas)", 0.0, 0.5, 0.20)
            p_a = st.number_input("Solid Yield Factor (a)", 0.1, 0.5, 0.35)
            p_b = st.number_input("Degradation (b)", 0.001, 0.01, 0.004, format="%.4f")
            
        params = {"k_f": p_kf, "C_oil": p_Coil, "C_gas": p_Cgas, "a_solid": p_a, "b_solid": p_b}

        with st.expander("💰 Economics", expanded=False):
            st.session_state.cost_biomass = st.number_input("Feed ($/ton)", value=st.session_state.cost_biomass)
            st.session_state.cost_energy = st.number_input("Energy ($/kWh)", value=st.session_state.cost_energy)
            st.session_state.price_char = st.number_input("Char Price ($/kg)", value=st.session_state.price_char)
        
        game_mode = st.checkbox("🎮 Optimization Mode")

    # --- Calculations ---
    res = run_simulation(mass, moisture, ash, temp, time_min, params)
    
    # Economics
    cost_feed = (mass / 1000) * st.session_state.cost_biomass
    energy_kwh = res['Q_total_kJ'] / 3600.0
    cost_ops = energy_kwh * st.session_state.cost_energy
    revenue = res['char_kg'] * st.session_state.price_char
    profit = revenue - (cost_feed + cost_ops)

    # --- Dashboard View ---
    st.title("CHEMISCO: Advanced Dashboard")
    st.markdown("---")
    
    # Flow Visualization
    c1, c2, c3, c4, c5 = st.columns([1.5, 0.5, 1.5, 0.5, 1.5])
    with c1: st.markdown(f'<div class="bfd-block">FEED<br>{mass} kg<br>{moisture}% H2O</div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="bfd-stream">➜</div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="bfd-block">{reactor.upper()}<br>{temp}°C | {time_min}min</div>', unsafe_allow_html=True)
    with c4: st.markdown('<div class="bfd-stream">➜</div>', unsafe_allow_html=True)
    with c5: st.markdown(f'<div class="bfd-block">BIOCHAR<br>{res["char_kg"]:.1f} kg</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    # Metrics
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Mass Yield", f"{res['mass_yield_pct']:.1f}%", f"{res['char_kg']:.1f} kg")
    k2.metric("Energy Density (HHV)", f"{res['hhv_final']:.2f} MJ/kg", f"+{res['hhv_increase_pct']:.1f}% Increase")
    k3.metric("Bio-Oil Output", f"{res['oil_kg']:.1f} kg", "Condensable Volatiles")
    k4.metric("Est. Profit", f"${profit:.2f}", f"Energy Cost: ${cost_ops:.2f}")
    
    st.markdown("---")

    # --- Tabs ---
    t1, t2, t3, t4 = st.tabs(["📊 Charts", "📈 Time Analysis", "📄 Report", "🎮 Game"])
    
    plot_bg = '#000000'
    txt_col = '#ffffff'
    colors_seq = ["#2ecc71", "#3498db", "#e67e22", "#e74c3c"] # Char, Water, Oil, Gas
    
    with t1:
        cc1, cc2 = st.columns(2)
        with cc1:
            st.subheader("Mass Balance Distribution")
            df_pie = pd.DataFrame({
                "Component": ["Biochar", "Water Vapor", "Bio-Oil", "Gases"],
                "Mass (kg)": [res['char_kg'], res['water_evap_kg'], res['oil_kg'], res['gas_kg']]
            })
            fig = px.pie(df_pie, values='Mass (kg)', names='Component', hole=0.4, color_discrete_sequence=colors_seq)
            fig.update_layout(paper_bgcolor=plot_bg, plot_bgcolor=plot_bg, font_color=txt_col)
            st.plotly_chart(fig, use_container_width=True)
            
        with cc2:
            st.subheader("Solid Composition")
            organic_char = res['char_kg'] - res['ash_kg']
            df_bar = pd.DataFrame({
                "Type": ["Organic Carbon", "Ash"],
                "Mass (kg)": [organic_char, res['ash_kg']]
            })
            fig2 = px.bar(df_bar, x='Type', y='Mass (kg)', color='Type', color_discrete_sequence=['#2ecc71', '#7f8c8d'])
            fig2.update_layout(paper_bgcolor=plot_bg, plot_bgcolor=plot_bg, font_color=txt_col, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    with t2:
        st.subheader("Process Kinetics (Simulation)")
        df_time = get_time_series(mass, moisture, ash, temp, time_min, params)
        
        fig_area = go.Figure()
        fig_area.add_trace(go.Scatter(x=df_time['Time (min)'], y=df_time['Char (kg)'], stackgroup='one', name='Char', line=dict(width=0, color='#2ecc71')))
        fig_area.add_trace(go.Scatter(x=df_time['Time (min)'], y=df_time['Bio-Oil (kg)'], stackgroup='one', name='Bio-Oil', line=dict(width=0, color='#e67e22')))
        fig_area.add_trace(go.Scatter(x=df_time['Time (min)'], y=df_time['Gases (kg)'], stackgroup='one', name='Gases', line=dict(width=0, color='#e74c3c')))
        fig_area.add_trace(go.Scatter(x=df_time['Time (min)'], y=df_time['Water Vapor (kg)'], stackgroup='one', name='Water Vapor', line=dict(width=0, color='#3498db')))
        fig_area.update_layout(paper_bgcolor=plot_bg, plot_bgcolor=plot_bg, font_color=txt_col, title="Product Evolution Over Time", xaxis_title="Time (min)", yaxis_title="Mass (kg)")
        st.plotly_chart(fig_area, use_container_width=True)
        
        fig_hhv = px.line(df_time, x="Time (min)", y="HHV Increase (%)", title="Energy Density Increase")
        fig_hhv.update_traces(line_color="#d97584")
        fig_hhv.update_layout(paper_bgcolor=plot_bg, plot_bgcolor=plot_bg, font_color=txt_col)
        st.plotly_chart(fig_hhv, use_container_width=True)

    with t3:
        st.write("Generate official technical PDF report.")
        pdf = create_pdf(res, profit)
        st.download_button("Download Report PDF", pdf, "chemisco_report.pdf", "application/pdf")

    with t4:
        if game_mode:
            st.info("🎯 Goal: Yield > 70% AND HHV > 22 MJ/kg")
            if res['mass_yield_pct'] > 70 and res['hhv_final'] > 22:
                st.balloons()
                st.success(f"🏆 WINNER! Yield: {res['mass_yield_pct']:.1f}%, HHV: {res['hhv_final']:.2f}")
            else:
                st.warning(f"Current: Yield {res['mass_yield_pct']:.1f}%, HHV {res['hhv_final']:.2f}")
        else:
            st.write("Tick 'Optimization Mode' in sidebar to play.")

if __name__ == "__main__":
    main()
