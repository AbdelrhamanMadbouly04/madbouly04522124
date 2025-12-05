import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from io import BytesIO
from reportlab.lib import colors
from datetime import datetime
import math
import streamlit.components.v1 as components # Import components for AI Injection

# --- 1. Constants & Defaults ---
R_GAS = 8.314
CP_BIOMASS = 1500.0
CP_WATER = 4180.0
H_VAPOR = 2260000.0
HHV_DRY_INITIAL_DEFAULT = 18.0

# --- 2. Styles (Professional Clean Theme) ---
GLOBAL_CSS = """
<style>
    /* Main Background */
    .stApp { background-color: #f4f6f9; font-family: 'Segoe UI', sans-serif; }
    
    /* Sidebar Styles */
    section[data-testid="stSidebar"] { background-color: #1a3c34; color: #ffffff; }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] label { color: #e0f2f1 !important; }
    section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] p { color: #b2dfdb !important; }

    /* Text Colors */
    h1, h2, h3 { color: #1a3c34 !important; font-weight: 700; }
    .stMarkdown, p, div, span, li { color: #2c3e50; }

    /* Metrics Cards */
    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #d1d5db; border-left: 6px solid #1a3c34;
        border-radius: 8px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetricValue"] { color: #1a3c34 !important; }
    div[data-testid="stMetricLabel"] { color: #546e7a !important; font-weight: 600; }

    /* Buttons */
    .stButton > button { 
        background-color: #1a3c34 !important; color: white !important; 
        border: none; font-weight: bold; border-radius: 6px;
    }

    /* Flow Visualization Blocks */
    .bfd-block {
        padding: 15px; border-radius: 8px; text-align: center; background: #ffffff;
        border: 2px solid #1a3c34; color: #37474f; font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .bfd-stream { color: #1a3c34; font-size: 24px; padding-top: 10px; font-weight: bold; }

    /* Sidebar Header - Logo Box Style */
    .header-box {
        background-color: #2e7d32; 
        padding: 15px; 
        border-radius: 8px; 
        text-align: center; 
        margin-bottom: 25px;
        border: 1px solid #4caf50;
    }
    .header-box h1 { color: #ffffff !important; margin: 0; font-size: 1.8rem; font-weight: 800; letter-spacing: 2px; }
    .header-box p { color: #e8f5e9 !important; margin: 0; font-size: 0.9rem; margin-top: 5px; }

    /* Tabs */
    div[data-testid="stTabs"] button { color: #546e7a; font-weight: 600; }
    div[data-testid="stTabs"] button[aria-selected="true"] { color: #1a3c34 !important; border-bottom: 3px solid #1a3c34 !important; }
    
    #MainMenu, footer, .stDeployButton {visibility: hidden;}
</style>
"""

# --- 3. Mathematical Models ---
def moisture_evap_linear(initial_moisture_kg, T_C, t_min, k_f=0.02):
    if T_C <= 100: return 0.0
    evap_kg = k_f * (T_C - 100) * t_min * initial_moisture_kg
    return min(initial_moisture_kg, max(0.0, evap_kg))

def Y_solid_empirical(T_C, t_min, a=0.35, b=0.004):
    severity = max(0.0, T_C - 200) * t_min
    return 1.0 - a * (1.0 - math.exp(-b * severity))

def m_oil(dry_mass_kg, T_C, t_min, C_oil=0.25):
    k_oil = 0.0008 * max(0.0, T_C - 200)
    return dry_mass_kg * C_oil * (1.0 - math.exp(-k_oil * t_min))

def m_gas(dry_mass_kg, T_C, t_min, C_gas=0.20):
    k_gas = 0.0015 * max(0.0, T_C - 180)
    return dry_mass_kg * C_gas * (1.0 - math.exp(-k_gas * t_min))

def hhv_improved_model(Y_solid, temp_c, enhancement_factor=0.85):
    mass_loss_fraction = 1.0 - Y_solid
    base_increase = mass_loss_fraction * enhancement_factor
    temp_bonus = 0.0
    if temp_c > 280: temp_bonus = 0.02 * ((temp_c - 280) / 50.0)
    return base_increase + temp_bonus

def run_simulation(mass_in, moisture_pct, ash_pct_dry, temp_c, time_min, params):
    moisture_frac = moisture_pct / 100.0
    M0_water = mass_in * moisture_frac
    M0_dry = mass_in * (1.0 - moisture_frac)
    M_ash = M0_dry * (ash_pct_dry / 100.0)
    
    w_evap = moisture_evap_linear(M0_water, temp_c, time_min, k_f=params['k_f'])
    oil_kg = m_oil(M0_dry, temp_c, time_min, C_oil=params['C_oil'])
    gas_kg = m_gas(M0_dry, temp_c, time_min, C_gas=params['C_gas'])
    char_dry = max(0, M0_dry - oil_kg - gas_kg) 
    char_total_mass = char_dry + (M0_water - w_evap)
    
    y_solid_val = Y_solid_empirical(temp_c, time_min, a=params['a_solid'], b=params['b_solid'])
    hhv_inc_frac = hhv_improved_model(y_solid_val, temp_c, enhancement_factor=params.get('energy_factor', 0.85))
    hhv_final = HHV_DRY_INITIAL_DEFAULT * (1.0 + hhv_inc_frac)
    
    energy_in = M0_dry * HHV_DRY_INITIAL_DEFAULT
    energy_out = char_dry * hhv_final
    
    T_K = temp_c + 273.15
    Q_total_kJ = ((M0_dry * CP_BIOMASS * (T_K - 298.15)) + (M0_water * CP_WATER * (373.15 - 298.15)) + (w_evap * H_VAPOR)) / 1000
    
    return {
        "mass_in": mass_in, "char_kg": char_total_mass, "water_evap_kg": w_evap,
        "oil_kg": oil_kg, "gas_kg": gas_kg, "ash_kg": M_ash,
        "hhv_final": hhv_final, "mass_yield_pct": (char_total_mass / mass_in) * 100,
        "energy_yield_pct": (energy_out / energy_in) * 100 if energy_in > 0 else 0,
        "hhv_increase_pct": hhv_inc_frac * 100, "Q_total_kJ": Q_total_kJ
    }

def get_time_series(mass_in, moisture_pct, ash_pct_dry, temp_c, time_min, params):
    times = np.linspace(0, time_min, 50)
    data = []
    for t in times:
        res = run_simulation(mass_in, moisture_pct, ash_pct_dry, temp_c, t, params)
        data.append({
            "Time (min)": t, "Char (kg)": res['char_kg'], "Bio-Oil (kg)": res['oil_kg'],
            "Gases (kg)": res['gas_kg'], "Water Vapor (kg)": res['water_evap_kg'],
            "HHV Increase (%)": res['hhv_increase_pct']
        })
    return pd.DataFrame(data)

# --- 4. Professional PDF Generator ---
def create_pdf(res, profit, fig1, fig2):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    CHEMISCO_GREEN = colors.HexColor('#1a3c34')
    LOGO_BLUE_GREEN = colors.HexColor('#2e7d32')
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Header with Logo
    logo_style = ParagraphStyle(name='LogoText', fontName='Helvetica-Bold', fontSize=18, textColor=colors.white, alignment=1)
    sub_logo_style = ParagraphStyle(name='SubLogo', fontName='Helvetica', fontSize=8, textColor=colors.whitesmoke, alignment=1)
    
    logo_content = [[Paragraph("CHEMISCO", logo_style)], [Paragraph("Torrefaction Simulator", sub_logo_style)]]
    t_logo = Table(logo_content, colWidths=[2.5*inch])
    t_logo.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), LOGO_BLUE_GREEN),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOX', (0,0), (-1,-1), 1, colors.white), ('TOPPADDING', (0,0), (-1,-1), 6), ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))

    info_text = Paragraph(f"<b>Date:</b> {current_time}<br/><b>Status:</b> Success", styles['Normal'])
    header_layout = [[t_logo, info_text]]
    t_header_main = Table(header_layout, colWidths=[3*inch, 3*inch])
    t_header_main.setStyle(TableStyle([('ALIGN', (0,0), (0,0), 'LEFT'), ('ALIGN', (1,0), (1,0), 'RIGHT'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
    
    story.append(t_header_main)
    story.append(Spacer(1, 25))
    
    # Title & Metrics
    story.append(Paragraph("Technical Engineering Report", ParagraphStyle(name='Title', parent=styles['Heading2'], textColor=CHEMISCO_GREEN, fontSize=16)))
    story.append(Spacer(1, 10))
    story.append(Paragraph("This document summarizes the simulation results for the biomass torrefaction process.", styles['Normal']))
    story.append(Spacer(1, 20))

    data = [
        ["Metric", "Value"], 
        ["Mass Yield", f"{res['mass_yield_pct']:.1f} %"], 
        ["Energy Density (HHV)", f"{res['hhv_final']:.2f} MJ/kg"],
        ["Energy Yield", f"{res['energy_yield_pct']:.1f} %"],
        ["Bio-Oil Produced", f"{res['oil_kg']:.2f} kg"],
        ["Profit Estimate", f"${profit:.2f}"]
    ]
    t = Table(data, colWidths=[3.5*inch, 2.5*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), CHEMISCO_GREEN), ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12), ('BACKGROUND', (0,1), (-1,-1), colors.whitesmoke),
        ('GRID', (0,0), (-1,-1), 1, colors.grey)
    ]))
    story.append(t); story.append(Spacer(1, 30))

    # Charts
    def add_plot(fig, title):
        try:
            fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", font=dict(color="black"))
            img_bytes = fig.to_image(format="png", width=800, height=450, scale=2)
            story.append(Paragraph(f"<b>{title}</b>", styles['Heading3']))
            story.append(Image(BytesIO(img_bytes), width=6*inch, height=3.5*inch))
            story.append(Spacer(1, 20))
        except Exception:
            story.append(Paragraph(f"<font color=red>Error rendering chart: {title}. Ensure 'kaleido==0.2.1' is installed.</font>", styles['Normal']))

    add_plot(fig1, "Figure 1: Mass Balance Distribution")
    add_plot(fig2, "Figure 2: Solid Composition")
    
    story.append(Spacer(1, 30))
    story.append(Paragraph("<font color=grey size=8>Chemisco Simulator v3.4 | Confidential & Proprietary</font>", styles['Normal']))
    doc.build(story)
    buffer.seek(0)
    return buffer

# --- 5. Main Streamlit App ---
def main():
    st.set_page_config(page_title="Chemisco Pro", layout="wide", initial_sidebar_state="expanded")
    
    # *** 🚀 INJECT BOTPRESS (EXACT CODE FROM HOME (9) (1).PY) ***
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
    # ***************************************************************

    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    
    if 'cost_biomass' not in st.session_state: 
        st.session_state.update({'cost_biomass': 30.0, 'cost_energy': 0.15, 'price_char': 1.20})

    # Sidebar
    with st.sidebar:
        st.markdown("""
            <div class="header-box">
                <h1>CHEMISCO</h1>
                <p>TORREFACTION SIMULATOR</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.header("⚙️ Inputs")
        reactor = st.selectbox("Reactor Type", ["Rotary Drum", "Fluidized Bed", "Screw Reactor"])
        
        with st.expander("🌲 Feedstock", expanded=True):
            mass = st.number_input("Mass (kg)", 1.0, 10000.0, 100.0, 10.0)
            moisture = st.slider("Moisture (%)", 0.0, 60.0, 15.0)
            ash = st.slider("Ash (Dry %)", 0.0, 30.0, 5.0)
            
        with st.expander("🔥 Process", expanded=True):
            temp = st.slider("Temp (°C)", 150, 350, 275)
            time_min = st.slider("Time (min)", 10, 120, 30)

        with st.expander("🔧 Model Params", expanded=False):
            p_kf = st.number_input("Drying rate", 0.0, 0.1, 0.02)
            p_Coil = st.number_input("Max Oil frac", 0.0, 0.5, 0.25)
            p_Cgas = st.number_input("Max Gas frac", 0.0, 0.5, 0.20)
            p_a = st.number_input("Solid Yield Factor", 0.1, 0.5, 0.35)
            p_b = st.number_input("Degradation", 0.001, 0.01, 0.004, format="%.4f")
            st.markdown("---")
            p_enh = st.slider("Energy Factor", 0.2, 1.5, 0.85)
            
        params = {"k_f": p_kf, "C_oil": p_Coil, "C_gas": p_Cgas, "a_solid": p_a, "b_solid": p_b, "energy_factor": p_enh}

        with st.expander("💰 Economics", expanded=False):
            st.session_state.cost_biomass = st.number_input("Feed ($/ton)", value=st.session_state.cost_biomass)
            st.session_state.cost_energy = st.number_input("Energy ($/kWh)", value=st.session_state.cost_energy)
            st.session_state.price_char = st.number_input("Char Price ($/kg)", value=st.session_state.price_char)
        
        game_mode = st.checkbox("🎮 Optimization Challenge")

    # Calculations
    res = run_simulation(mass, moisture, ash, temp, time_min, params)
    cost_feed = (mass / 1000) * st.session_state.cost_biomass
    energy_kwh = res['Q_total_kJ'] / 3600.0
    cost_ops = energy_kwh * st.session_state.cost_energy
    revenue = res['char_kg'] * st.session_state.price_char
    profit = revenue - (cost_feed + cost_ops)

    # Visualization
    APP_TXT_COLOR = "#000000"
    APP_BG_COLOR = "#f4f6f9"
    colors_seq = ["#1a3c34", "#5c6bc0", "#ffa726", "#ef5350"]

    # 1. Pie Chart
    df_pie = pd.DataFrame({
        "Component": ["Biochar", "Water Vapor", "Bio-Oil", "Gases"],
        "Mass (kg)": [res['char_kg'], res['water_evap_kg'], res['oil_kg'], res['gas_kg']]
    })
    fig1 = px.pie(df_pie, values='Mass (kg)', names='Component', hole=0.5, color_discrete_sequence=colors_seq, title="Mass Balance")
    fig1.update_layout(paper_bgcolor=APP_BG_COLOR, plot_bgcolor=APP_BG_COLOR, font=dict(color=APP_TXT_COLOR, size=14))

    # 2. Bar Chart
    organic_char = res['char_kg'] - res['ash_kg']
    df_bar = pd.DataFrame({
        "Type": ["Organic Carbon", "Ash"],
        "Mass (kg)": [organic_char, res['ash_kg']]
    })
    fig2 = px.bar(df_bar, x='Type', y='Mass (kg)', color='Type', color_discrete_sequence=['#1a3c34', '#b0bec5'], title="Solid Composition")
    fig2.update_layout(
        paper_bgcolor=APP_BG_COLOR, plot_bgcolor=APP_BG_COLOR, font=dict(color=APP_TXT_COLOR, size=14),
        xaxis=dict(title_font=dict(color=APP_TXT_COLOR), tickfont=dict(color=APP_TXT_COLOR)),
        yaxis=dict(title_font=dict(color=APP_TXT_COLOR), tickfont=dict(color=APP_TXT_COLOR)),
        showlegend=False
    )

    # Dashboard
    st.title("CHEMISCO: Process Dashboard")
    st.markdown("---")
    
    c1, c2, c3, c4, c5 = st.columns([1.5, 0.5, 1.5, 0.5, 1.5])
    with c1: st.markdown(f'<div class="bfd-block">FEED<br>{mass} kg<br>{moisture}% H2O</div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="bfd-stream">➜</div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="bfd-block">{reactor.upper()}<br>{temp}°C | {time_min}min</div>', unsafe_allow_html=True)
    with c4: st.markdown('<div class="bfd-stream">➜</div>', unsafe_allow_html=True)
    with c5: st.markdown(f'<div class="bfd-block">BIOCHAR<br>{res["char_kg"]:.1f} kg</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Mass Yield", f"{res['mass_yield_pct']:.1f}%", f"{res['char_kg']:.1f} kg")
    k2.metric("Energy Density (HHV)", f"{res['hhv_final']:.2f} MJ/kg", f"+{res['hhv_increase_pct']:.1f}% Increase")
    k3.metric("Bio-Oil Output", f"{res['oil_kg']:.1f} kg", "Condensable Volatiles")
    k4.metric("Est. Profit", f"${profit:.2f}", f"Energy Cost: ${cost_ops:.2f}")
    st.markdown("---")

    t1, t2, t3, t4 = st.tabs(["📊 Analytics", "📈 Kinetics", "📄 Export", "🎯 Challenge"])
    
    with t1:
        cc1, cc2 = st.columns(2)
        with cc1: st.plotly_chart(fig1, use_container_width=True)
        with cc2: st.plotly_chart(fig2, use_container_width=True)

    with t2:
        df_time = get_time_series(mass, moisture, ash, temp, time_min, params)
        fig_area = go.Figure()
        fig_area.add_trace(go.Scatter(x=df_time['Time (min)'], y=df_time['Char (kg)'], stackgroup='one', name='Char', line=dict(width=0, color='#1a3c34')))
        fig_area.add_trace(go.Scatter(x=df_time['Time (min)'], y=df_time['Bio-Oil (kg)'], stackgroup='one', name='Bio-Oil', line=dict(width=0, color='#ffa726')))
        fig_area.add_trace(go.Scatter(x=df_time['Time (min)'], y=df_time['Gases (kg)'], stackgroup='one', name='Gases', line=dict(width=0, color='#ef5350')))
        fig_area.add_trace(go.Scatter(x=df_time['Time (min)'], y=df_time['Water Vapor (kg)'], stackgroup='one', name='Water Vapor', line=dict(width=0, color='#5c6bc0')))
        fig_area.update_layout(
            paper_bgcolor=APP_BG_COLOR, plot_bgcolor=APP_BG_COLOR, title="Product Evolution", 
            font=dict(color=APP_TXT_COLOR),
            xaxis=dict(title="Time (min)", tickfont=dict(color="black"), title_font=dict(color="black")),
            yaxis=dict(title="Mass (kg)", tickfont=dict(color="black"), title_font=dict(color="black"))
        )
        st.plotly_chart(fig_area, use_container_width=True)

    with t3:
        st.markdown("### 📄 Professional Report Generation")
        try:
            import kaleido
            pdf = create_pdf(res, profit, fig1, fig2)
            st.download_button("Download PDF Report", pdf, f"Chemisco_Report.pdf", "application/pdf")
        except ImportError:
            st.error("⚠️ Library Missing: Please ensure 'kaleido==0.2.1' is in requirements.txt")

    with t4:
        if game_mode:
            TARGET_HHV, MIN_YIELD, TARGET_PROFIT = 22.0, 55.0, 0.0
            st.markdown("### 🎯 Engineering Challenge"); st.markdown("---")
            col_g1, col_g2, col_g3 = st.columns(3)
            delta_hhv = res['hhv_final'] - TARGET_HHV
            col_g1.metric("HHV (>22)", f"{res['hhv_final']:.2f}", f"{delta_hhv:.2f}", delta_color="normal" if res['hhv_final'] >= TARGET_HHV else "inverse")
            delta_yield = res['mass_yield_pct'] - MIN_YIELD
            col_g2.metric("Yield (>55%)", f"{res['mass_yield_pct']:.1f}%", f"{delta_yield:.1f}%", delta_color="normal" if res['mass_yield_pct'] >= MIN_YIELD else "inverse")
            col_g3.metric("Profit (>$0)", f"${profit:.2f}", "Net", delta_color="normal" if profit > 0 else "inverse")
            if res['hhv_final'] >= TARGET_HHV and res['mass_yield_pct'] >= MIN_YIELD and profit > TARGET_PROFIT:
                st.balloons(); st.success("🏆 Success!")
            else:
                st.warning("Optimization Failed. Adjust Temp/Time.")
        else:
            st.info("Enable 'Optimization Challenge' in sidebar.")

if __name__ == "__main__":
    main()
