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

# --- 2. Styles (Green & Light Gray) ---
GLOBAL_CSS = """
<style>
    /* Theme: Green & Light Gray */
    .stApp { background-color: #e9e9e9; color: #1a1a1a; font-family: 'Segoe UI', sans-serif; }
    
    /* Headings */
    h1, h2, h3, h4, h5, h6 { color: #00743c !important; }
    
    /* General Text */
    .stMarkdown, p, label, li { color: #333333 !important; }
    
    /* Sidebar (Green Background) */
    section[data-testid="stSidebar"] { background-color: #00743c; }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] label, 
    section[data-testid="stSidebar"] p { color: #ffffff !important; }
    
    /* Inputs in Sidebar */
    .stSlider > div > div > div > div { background-color: #ffffff !important; }
    .stSelectbox > div > div { color: #ffffff; }

    /* Metrics (Cards) - White bg on Light Gray */
    div[data-testid="stMetric"] {
        background-color: #ffffff; 
        border: 2px solid #00743c;
        border-radius: 8px; padding: 10px; 
        box-shadow: 0px 4px 10px rgba(0, 116, 60, 0.2);
    }
    div[data-testid="stMetricValue"] { color: #000000 !important; font-size: 24px !important; }
    div[data-testid="stMetricLabel"] { color: #00743c !important; font-size: 14px !important; font-weight: bold; }

    /* Header Box */
    .header-box {
        background: #ffffff; border: 2px solid #00743c; padding: 20px;
        border-radius: 10px; text-align: center;
        margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .header-box h1 { color: #00743c !important; margin: 0; }
    .header-box p { color: #666666 !important; margin: 0; }

    /* Tabs */
    div[data-testid="stTabs"] button { color: #555555 !important; font-weight: bold; background: transparent !important; }
    div[data-testid="stTabs"] button[aria-selected="true"] { color: #00743c !important; border-bottom: 3px solid #00743c !important; }
    
    /* Buttons */
    .stButton > button { background-color: #00743c !important; color: #ffffff !important; border: none; border-radius: 6px; }
    .stButton > button:hover { background-color: #005a2e !important; }

    /* Flow Visualization Blocks */
    .bfd-block {
        padding: 10px; border-radius: 8px; text-align: center; 
        background: #ffffff; /* White Block */
        border: 2px solid #00743c; 
        color: #333333; font-weight: bold; font-size: 0.9em;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .bfd-stream { color: #00743c; font-size: 20px; padding-top: 10px; text-align: center; font-weight: bold; }
    
    /* Expander Header */
    .streamlit-expanderHeader { color: #000000 !important; background-color: #f0f0f0 !important; border-radius: 5px; }

    /* Hide Hamburger Menu & Footer */
    #MainMenu, footer, .stDeployButton {visibility: hidden;}
</style>
"""

# --- 3. Mathematical Models ---

def moisture_evap_linear(initial_moisture_kg, T_C, t_min, k_f=0.02):
    """(A) Linear moisture evaporation"""
    if T_C <= 100:
        return 0.0
    evap_kg = k_f * (T_C - 100) * t_min * initial_moisture_kg
    return min(initial_moisture_kg, max(0.0, evap_kg))

def Y_solid_empirical(T_C, t_min, a=0.35, b=0.004):
    """(C) Solid Yield Model"""
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

def hhv_improved_model(Y_solid, temp_c, enhancement_factor=0.85):
    """(F) Improved HHV Model"""
    mass_loss_fraction = 1.0 - Y_solid
    base_increase = mass_loss_fraction * enhancement_factor
    temp_bonus = 0.0
    if temp_c > 280:
        temp_bonus = 0.02 * ((temp_c - 280) / 50.0)
    return base_increase + temp_bonus

def run_simulation(mass_in, moisture_pct, ash_pct_dry, temp_c, time_min, params):
    moisture_frac = moisture_pct / 100.0
    ash_frac_dry = ash_pct_dry / 100.0
    
    M0_water = mass_in * moisture_frac
    M0_dry = mass_in * (1.0 - moisture_frac)
    M_ash = M0_dry * ash_frac_dry
    
    w_evap = moisture_evap_linear(M0_water, temp_c, time_min, k_f=params['k_f'])
    w_remaining = M0_water - w_evap
    
    oil_kg = m_oil(M0_dry, temp_c, time_min, C_oil=params['C_oil'])
    gas_kg = m_gas(M0_dry, temp_c, time_min, C_gas=params['C_gas'])
    
    char_dry = max(0, M0_dry - oil_kg - gas_kg) 
    char_total_mass = char_dry + w_remaining
    
    y_solid_val = Y_solid_empirical(temp_c, time_min, a=params['a_solid'], b=params['b_solid'])
    enh_factor = params.get('energy_factor', 0.85)
    hhv_inc_frac = hhv_improved_model(y_solid_val, temp_c, enhancement_factor=enh_factor)
    hhv_final = HHV_DRY_INITIAL_DEFAULT * (1.0 + hhv_inc_frac)
    
    energy_in = M0_dry * HHV_DRY_INITIAL_DEFAULT
    energy_out = char_dry * hhv_final
    energy_yield = (energy_out / energy_in) * 100 if energy_in > 0 else 0
    
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
    # Changed table color to Green to match theme
    t.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor('#00743c')), 
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
            st.markdown("---")
            p_enh = st.slider("Energy Factor", 0.2, 1.5, 0.85)
            
        params = {"k_f": p_kf, "C_oil": p_Coil, "C_gas": p_Cgas, "a_solid": p_a, "b_solid": p_b, "energy_factor": p_enh}

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
    
    # Update Plot Styles for Light Theme
    plot_bg = '#e9e9e9' # Light Gray
    txt_col = '#000000' # Black
    # Update colors: Green for Char, Blue for Water
    colors_seq = ["#00743c", "#3498db", "#e67e22", "#e74c3c"] 
    
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
            fig2 = px.bar(df_bar, x='Type', y='Mass (kg)', color='Type', color_discrete_sequence=['#00743c', '#7f8c8d'])
            fig2.update_layout(paper_bgcolor=plot_bg, plot_bgcolor=plot_bg, font_color=txt_col, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    with t2:
        st.subheader("Process Kinetics (Simulation)")
        df_time = get_time_series(mass, moisture, ash, temp, time_min, params)
        
        fig_area = go.Figure()
        fig_area.add_trace(go.Scatter(x=df_time['Time (min)'], y=df_time['Char (kg)'], stackgroup='one', name='Char', line=dict(width=0, color='#00743c')))
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
            st.markdown("### 🎯 Engineering Challenge: Find the Sweet Spot")
            st.markdown("""
            **مهمتك كمهندس:** المستثمر عايز منتج مواصفاته عالية (Bio-Coal) وفي نفس الوقت يحقق ربح.
            عشان تكسب لازم تحقق الـ 3 شروط دول في نفس الوقت:
            1.  **الجودة:** كثافة الطاقة (HHV) لازم تكون **أعلى من 22.0 MJ/kg**.
            2.  **الإنتاجية:** صافي الوزن (Yield) لازم يكون **أعلى من 55%**.
            3.  **الاقتصاد:** لازم تحقق **صافي ربح موجب (> $0)**.
            """)
            
            st.markdown("---")

            # Goals
            TARGET_HHV = 22.0
            MIN_YIELD = 55.0
            TARGET_PROFIT = 0.0

            col_g1, col_g2, col_g3 = st.columns(3)
            
            # HHV Check
            delta_hhv = res['hhv_final'] - TARGET_HHV
            col_g1.metric("Energy Density (HHV)", f"{res['hhv_final']:.2f} MJ/kg", f"{delta_hhv:.2f} (Target: >22)", 
                          delta_color="normal" if res['hhv_final'] >= TARGET_HHV else "inverse")
            
            # Yield Check
            delta_yield = res['mass_yield_pct'] - MIN_YIELD
            col_g2.metric("Mass Yield", f"{res['mass_yield_pct']:.1f}%", f"{delta_yield:.1f}% (Target: >55%)",
                          delta_color="normal" if res['mass_yield_pct'] >= MIN_YIELD else "inverse")
            
            # Profit Check
            col_g3.metric("Net Profit", f"${profit:.2f}", "Must be Positive",
                          delta_color="normal" if profit > 0 else "inverse")

            st.markdown("---")

            success_hhv = res['hhv_final'] >= TARGET_HHV
            success_yield = res['mass_yield_pct'] >= MIN_YIELD
            success_profit = profit > TARGET_PROFIT

            if success_hhv and success_yield and success_profit:
                st.balloons()
                st.success("🏆 **مبــــروك! لقد وجدت التوازن المثالي (The Sweet Spot)**")
                score = (res['hhv_final'] * res['mass_yield_pct']) + profit
                st.metric("🌟 Engineering Score", f"{int(score)}")
            else:
                st.error("❌ **محاولة غير ناجحة.. جرب تاني!**")
                st.markdown("#### 💡 نصائح المهندس الاستشاري:")
                if not success_hhv:
                    st.warning("🔸 **جودة منخفضة:** المنتج لسه خشب خام. **زود الحرارة أو الوقت**.")
                if not success_yield:
                    st.warning("🔸 **حرق زائد:** إنت حرقت كمية كبيرة. **قلل الحرارة أو الوقت**.")
                if not success_profit:
                    st.warning("🔸 **خسارة مادية:** تكلفة الطاقة عالية. حاول تظبط المعادلة.")

        else:
            st.info("👈 قم بتفعيل **'Optimization Mode'** من القائمة الجانبية لبدء اللعبة.")

if __name__ == "__main__":
    main()
