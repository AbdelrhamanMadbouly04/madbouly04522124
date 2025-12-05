import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
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

# --- 1. Constants (تم التحديث بناءً على طلبك) ---
R_GAS = 8.314
CP_BIOMASS = 1500.0 # Average for wood J/kg.K
CP_WATER = 4180.0
H_VAPOR = 2260000.0
TEMP_REF_K = 298.15

# افتراضات النموذج الجديد
HHV_DRY_INITIAL = 18.0  # MJ/kg
ASH_PERCENT = 0.05      # 5% من الكتلة الجافة

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

# --- 3. Simulation Logic (المعادلات الجديدة) ---
def simulate_torrefaction(biomass, moisture, temp_C, duration_min, size, initial_mass_kg, reactor_type="N/A"):
    # 1. تحضير كتل البداية
    moisture_frac = moisture / 100.0
    mass_water_init = initial_mass_kg * moisture_frac
    mass_dry_init = initial_mass_kg * (1.0 - moisture_frac)
    
    # الرماد ثابت 5% من الكتلة الجافة
    mass_ash = mass_dry_init * ASH_PERCENT
    
    # التأكد من درجة الحرارة للدوال الأسية (لتجنب القيم السالبة في المعادلات إذا كانت T < 200)
    # المعادلة مصممة لـ T >= 200
    T_eff = max(temp_C, 200) 
    
    # ---------------------------------------------------------
    # 2. نموذج تبخر الماء (Water Evaporation Model)
    # k_m(T) = 0.1 * e^(0.03 * (T - 200))
    # ---------------------------------------------------------
    k_m = 0.1 * np.exp(0.03 * (T_eff - 200))
    moist_evap_frac = 1.0 - np.exp(-k_m * duration_min)
    
    mass_water_evaporated = mass_water_init * moist_evap_frac
    mass_water_remaining = mass_water_init - mass_water_evaporated
    
    # ---------------------------------------------------------
    # 3. نموذج الطيارات/الزيوت (Volatiles Model)
    # V_max(T) = 0.45 * (1 - e^(-0.01 * (T - 200)))
    # k_v(T) = 0.05 * e^(0.02 * (T - 200))
    # ---------------------------------------------------------
    V_max = 0.45 * (1.0 - np.exp(-0.01 * (T_eff - 200)))
    k_v = 0.05 * np.exp(0.02 * (T_eff - 200))
    
    # كسر الطيارات المتحررة من الكتلة الجافة
    volatile_frac_released = V_max * (1.0 - np.exp(-k_v * duration_min))
    
    mass_volatiles_released = mass_dry_init * volatile_frac_released
    
    # ---------------------------------------------------------
    # 4. حساب الكتلة الصلبة المتبقية (Char Mass Calculation)
    # Char = Dry_Mass - Volatiles (Ash is included in Dry Mass and stays in Char)
    # ---------------------------------------------------------
    mass_char_dry = mass_dry_init - mass_volatiles_released
    
    # الكتلة الصلبة النهائية (تشمل أي ماء لم يتبخر + الرماد + المادة العضوية المتبقية)
    mass_product_total = mass_char_dry + mass_water_remaining
    
    # ---------------------------------------------------------
    # 5. نموذج الطاقة (Energy / HHV Model)
    # Severity = (T - 200) * t
    # Increase = 0.05 + 0.30 * (1 - e^(-0.0008 * Severity))
    # ---------------------------------------------------------
    severity = (T_eff - 200) * duration_min
    energy_increase_frac = 0.05 + 0.30 * (1.0 - np.exp(-0.0008 * severity))
    
    hhv_final_dry = HHV_DRY_INITIAL * (1.0 + energy_increase_frac)
    
    # ---------------------------------------------------------
    # 6. تجهيز النتائج للعرض
    # ---------------------------------------------------------
    
    # حساب نسب النواتج (Yields)
    # سنقسم الطيارات افتراضياً لغرض الرسم البياني فقط (مثلاً 80% زيوت مكثفة، 20% غازات)
    mass_bio_oil = mass_volatiles_released * 0.8
    mass_gases = mass_volatiles_released * 0.2
    
    yields_mass = pd.DataFrame({
        "Mass (kg)": [mass_product_total, mass_water_evaporated, mass_bio_oil, mass_gases]
    }, index=["Biochar", "Water Vapor", "Bio-Oil (Est.)", "Gases (Est.)"])
    
    yields_percent = pd.DataFrame({
        "Yield (%)": (yields_mass["Mass (kg)"] / initial_mass_kg) * 100
    })
    
    # تكوين المادة الصلبة (للشارت الثاني)
    # Organic Part of Char = Total Dry Char - Ash
    mass_organic_char = mass_char_dry - mass_ash
    solid_composition = pd.DataFrame({
        "Mass (kg)": [mass_organic_char, mass_ash]
    }, index=["Organic Matter", "Ash"])
    
    # كفاءة الطاقة
    energy_in_total = mass_dry_init * HHV_DRY_INITIAL
    energy_out_char = mass_char_dry * hhv_final_dry
    energy_yield_percent = (energy_out_char / energy_in_total) * 100
    
    final_ash_percent_in_product = (mass_ash / mass_char_dry) * 100 if mass_char_dry > 0 else 0

    return {
        "yields_percent": yields_percent,
        "yields_mass": yields_mass,
        "solid_composition": solid_composition,
        "final_ash_percent": final_ash_percent_in_product,
        "initial_hhv": HHV_DRY_INITIAL,
        "biochar_hhv": hhv_final_dry,
        "energy_yield_percent": energy_yield_percent,
        "parameters": {
            "biomass": biomass, "moisture": moisture, "temperature": temp_C, 
            "duration": duration_min, "size": size, "initial_mass": initial_mass_kg, 
            "reactor": reactor_type
        },
        "mass_moisture_loss_kg": mass_water_evaporated,
        "mass_dry_biomass_kg": mass_dry_init,
        "mass_char_dry": mass_char_dry 
    }

def calculate_thermal_balance(p, results):
    # حساب تقريبي للطاقة الحرارية المطلوبة
    T_K = p['temperature'] + 273.15
    
    # 1. تسخين المادة الصلبة (Sensible Heat)
    # Q = m * Cp * dT
    Q_sensible_biomass = (p['initial_mass'] * (1 - p['moisture']/100) * CP_BIOMASS * (T_K - TEMP_REF_K)) / 1000
    
    # 2. تسخين الماء (Sensible Heat)
    Q_sensible_water = (p['initial_mass'] * (p['moisture']/100) * CP_WATER * (373.15 - TEMP_REF_K)) / 1000 # To 100C
    
    # 3. تبخير الماء (Latent Heat)
    Q_latent_water = (results.get('mass_moisture_loss_kg', 0.0) * H_VAPOR) / 1000
    
    # 4. حرارة التفاعل (Torrefaction Heat)
    # التفاعل غالباً طارد للحرارة قليلاً (Exothermic) أو متعادل عند درجات منخفضة
    # سنفترض استهلاك بسيط للطاقة كاحتياط (Endothermic assumption for safety factor) أو 0
    Q_reaction_safety = results.get('mass_dry_biomass_kg', 0.0) * 50.0 # kJ assumption
    
    Q_total_required_kJ = Q_sensible_biomass + Q_sensible_water + Q_latent_water + Q_reaction_safety
    
    return {
        'Q_total_required_kJ': Q_total_required_kJ, 
        'Q_latent_water': Q_latent_water, 
        'Q_total_per_kg': Q_total_required_kJ / p['initial_mass']
    }

@st.cache_data
def run_sensitivity_analysis(biomass, moisture, size, initial_mass_kg, reactor_type):
    T_range = np.linspace(220, 320, 10)
    D_range = np.linspace(10, 90, 10)
    
    # تحليل الحرارة (تثبيت الزمن 30 دقيقة)
    results_T = [simulate_torrefaction(biomass, moisture, T, 30, size, initial_mass_kg, reactor_type)["yields_percent"].iloc[0,0] for T in T_range]
    
    # تحليل الزمن (تثبيت الحرارة 275 درجة)
    results_D = [simulate_torrefaction(biomass, moisture, 275, D, size, initial_mass_kg, reactor_type)["yields_percent"].iloc[0,0] for D in D_range]
    
    return pd.DataFrame({"Temp": T_range, "Yield": results_T}), pd.DataFrame({"Duration": D_range, "Yield": results_D})

def create_pdf(results, thermal, profit):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph("Chemisco Simulation Report", styles['Title']), Spacer(1, 12)]
    data = [
        ["Metric", "Value"], 
        ["Mass Yield", f"{results['yields_percent'].iloc[0,0]:.1f}%"], 
        ["Final HHV", f"{results['biochar_hhv']:.2f} MJ/kg"],
        ["Energy Yield", f"{results['energy_yield_percent']:.1f}%"], 
        ["Profit", f"${profit:.2f}"]
    ]
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
            # نوع الكتلة الحيوية لم يعد يؤثر على الكينتيكا (النموذج موحد الآن) لكن تركناه للعرض
            biomass = st.selectbox("Type", ["Wood Mix", "Agri Waste"]) 
            moisture = st.slider("Moisture (%)", 0.0, 50.0, 15.0) # Default changed to 15% as per example
            size = st.selectbox("Size", ["Fine", "Medium", "Coarse"])
        with st.expander("🔥 Process", expanded=True):
            temp = st.slider("Temp (°C)", 200, 350, 275) # Default 275
            time_min = st.slider("Time (min)", 10, 120, 30) # Default 30
        with st.expander("💰 Economics", expanded=False):
            st.session_state.cost_biomass = st.number_input("Feed ($/ton)", value=st.session_state.cost_biomass)
            st.session_state.cost_energy = st.number_input("Energy ($/hr)", value=st.session_state.cost_energy)
            st.session_state.price_char = st.number_input("Price ($/kg)", value=st.session_state.price_char)
        game_mode = st.checkbox("🎮 Plant Manager Mode")

    # تشغيل المحاكاة بالدوال الجديدة
    res = simulate_torrefaction(biomass, moisture, temp, time_min, size, mass, reactor)
    therm = calculate_thermal_balance(res['parameters'], res)
    
    # حسابات اقتصادية بسيطة
    cost_feed = (mass/1000)*st.session_state.cost_biomass
    cost_ops = (time_min/60)*st.session_state.cost_energy
    rev = res['yields_mass'].iloc[0,0] * st.session_state.price_char
    profit = rev - (cost_feed + cost_ops)

    st.title("CHEMISCO: Advanced Dashboard")
    st.markdown("---")
    
    # عرض التدفق (Flow)
    c1, c2, c3, c4, c5 = st.columns([1.5, 0.5, 1.5, 0.5, 1.5])
    with c1: st.markdown(f'<div class="bfd-block">FEED<br>{mass} kg<br>{moisture}% H2O</div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="bfd-stream">➜</div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="bfd-block">{reactor.upper()}<br>{temp}°C | {time_min}min</div>', unsafe_allow_html=True)
    with c4: st.markdown('<div class="bfd-stream">➜</div>', unsafe_allow_html=True)
    with c5: st.markdown(f'<div class="bfd-block">BIOCHAR<br>{res["yields_mass"].iloc[0,0]:.1f} kg</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    # المقاييس الرئيسية (Metrics)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Mass Yield (Solid)", f"{res['yields_percent'].iloc[0,0]:.1f}%", f"{res['yields_mass'].iloc[0,0]:.1f} kg")
    k2.metric("HHV (Energy Density)", f"{res['biochar_hhv']:.2f} MJ/kg", f"Energy Yield: {res['energy_yield_percent']:.1f}%")
    k3.metric("Total Heat Req", f"{therm['Q_total_required_kJ']/1000:.1f} MJ", f"{therm['Q_total_per_kg']:.0f} kJ/kg")
    k4.metric("Profit Estimate", f"${profit:.2f}", "Based on inputs")
    
    st.markdown("---")

    # --- TABS ---
    t1, t2, t3, t4 = st.tabs(["📊 Charts", "🌡️ Sensitivity", "📄 Report", "🎮 Game"])
    
    color_scale = ["#640d14", "#8c2f39", "#b3525e", "#d97584"]
    plot_bg = '#000000'
    txt_col = '#ffffff'
    
    with t1:
        cc1, cc2 = st.columns(2)
        with cc1:
            st.subheader("Mass Balance Distribution")
            # رسم الفطيرة للمكونات الخارجة (فحم، ماء، زيوت وغازات)
            fig = px.pie(res['yields_percent'].reset_index(), values='Yield (%)', names='index', 
                         color_discrete_sequence=color_scale, hole=0.4)
            fig.update_layout(paper_bgcolor=plot_bg, plot_bgcolor=plot_bg, font_color=txt_col)
            st.plotly_chart(fig, use_container_width=True)
        with cc2:
            st.subheader("Solid Product Composition")
            # رسم العمود لمكونات الفحم (مادة عضوية vs رماد)
            fig2 = px.bar(res['solid_composition'].reset_index(), x='index', y='Mass (kg)', 
                          color='index', color_discrete_sequence=['#d97584', '#640d14'])
            fig2.update_layout(paper_bgcolor=plot_bg, plot_bgcolor=plot_bg, font_color=txt_col, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    with t2:
        st.markdown("### How Temperature & Time affect Yield")
        df_T, df_D = run_sensitivity_analysis(biomass, moisture, size, mass, reactor)
        c_sens1, c_sens2 = st.columns(2)
        with c_sens1:
            fig_t = px.line(df_T, x="Temp", y="Yield", title="Effect of Temperature (at 30 min)", markers=True)
            fig_t.update_traces(line_color="#d97584")
            fig_t.update_layout(paper_bgcolor=plot_bg, plot_bgcolor=plot_bg, font_color=txt_col, xaxis_title="Temperature (°C)", yaxis_title="Mass Yield (%)")
            st.plotly_chart(fig_t, use_container_width=True)
        with c_sens2:
            fig_d = px.line(df_D, x="Duration", y="Yield", title="Effect of Time (at 275°C)", markers=True)
            fig_d.update_traces(line_color="#d97584")
            fig_d.update_layout(paper_bgcolor=plot_bg, plot_bgcolor=plot_bg, font_color=txt_col, xaxis_title="Time (min)", yaxis_title="Mass Yield (%)")
            st.plotly_chart(fig_d, use_container_width=True)

    with t3:
        st.write("Generate official technical PDF report.")
        pdf = create_pdf(res, therm, profit)
        st.download_button("Download PDF", pdf, "simulation_report.pdf", "application/pdf")

    with t4:
        if game_mode:
            st.info("🎯 Target: Yield > 70% AND HHV > 22 MJ/kg")
            if res['yields_percent'].iloc[0,0] > 70 and res['biochar_hhv'] > 22:
                st.balloons()
                st.success("🏆 Excellent Operation Parameters!")
            else:
                st.warning(f"Current: Yield {res['yields_percent'].iloc[0,0]:.1f}%, HHV {res['biochar_hhv']:.2f} MJ/kg")
        else:
            st.write("Activate Game Mode in Sidebar.")

if __name__ == "__main__":
    main()
