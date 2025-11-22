import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportImage, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors

# --- 1. الثوابت الكيميائية والبيانات التجريبية ---
R_GAS = 8.314  # ثابت الغازات العام

EMPIRICAL_DATA = {
    "Wood": {
        "A": 2.5e10, "Ea": 135000, "Ash": 0.01, "Gas_Factor": 0.35
    },
    "Agricultural Waste": {
        "A": 5.0e11, "Ea": 150000, "Ash": 0.08, "Gas_Factor": 0.45
    },
    "Municipal Waste": {
        "A": 1.0e12, "Ea": 165000, "Ash": 0.15, "Gas_Factor": 0.55
    }
}

SIZE_FACTOR = {
    "Fine (<1mm)": 1.0,
    "Medium (1-5mm)": 0.85,
    "Coarse (>5mm)": 0.65
}

# --- 2. دالة المحاكاة (Simulate Torrefaction) ---
def simulate_torrefaction(biomass, moisture, temp_C, duration_min, size, initial_mass_kg):
    """
    منطق المحاكاة المعدل:
    1. الماء يتبخر بمعدل يعتمد على الحرارة ويقف عند الصفر.
    2. المواد المتطايرة تتحلل بناء على معادلة أرهينيوس.
    """
    temp_K = temp_C + 273.15
    data = EMPIRICAL_DATA.get(biomass)
    
    # 1. حساب معدل التجفيف (Drying Rate)
    # الماء يتبخر أسرع بكثير عند درجات حرارة التوريفاكشن (>200C)
    # هذه معادلة تقريبية لزيادة سرعة التبخير مع الحرارة
    if temp_C < 100:
        k_drying = 0.05 # بطيء جدا تحت 100
    else:
        # كلما زادت الحرارة فوق 100، زاد المعدل بشكل كبير
        # هذا يضمن أن الماء يتبخر كلياً في وقت قصير عند 250 درجة مثلاً
        k_drying = 0.1 + (temp_C - 100) * 0.005 

    # 2. ثوابت التفاعل الكيميائي (Devolatilization)
    k_devol_arrhenius = data["A"] * np.exp(-data["Ea"] / (R_GAS * temp_K))
    k_devol_eff = k_devol_arrhenius * SIZE_FACTOR.get(size, 1.0)
    
    ash_content = data["Ash"]

    # تحويل النسب المئوية إلى كسور (Fraction 0-1)
    initial_moisture_fraction = moisture / 100.0
    
    # كتلة المواد المتطايرة الأولية = (1 - الرماد - الماء)
    initial_volatiles_fraction = 1.0 - initial_moisture_fraction - ash_content
    
    # حماية من القيم السالبة
    if initial_volatiles_fraction < 0:
        initial_moisture_fraction = 1.0 - ash_content
        initial_volatiles_fraction = 0.0

    # نظام المعادلات التفاضلية (ODEs)
    def model(y, t):
        m, v = y # m: الرطوبة المتبقية، v: المواد المتطايرة المتبقية
        
        # معادلة التجفيف: تتوقف تماماً إذا وصلت الرطوبة للصفر
        if m <= 0.001: # إذا كانت الرطوبة شبه منعدمة
            d_m = 0.0
            m = 0.0 # تثبيت القيمة عند الصفر
        else:
            d_m = -k_drying * m
            
        # معادلة التحلل الحراري
        if v <= 0:
            d_v = 0.0
        else:
            d_v = -k_devol_eff * v
            
        return [d_m, d_v]

    # الزمن
    t = np.linspace(0, duration_min, 200)
    y0 = [initial_moisture_fraction, initial_volatiles_fraction]
    
    # حل المعادلات
    sol = odeint(model, y0, t)
    
    # تنظيف النتائج (منع الأرقام السالبة الصغيرة جداً الناتجة عن الحساب الرقمي)
    sol[:, 0] = np.maximum(sol[:, 0], 0.0) # Moisture Profile
    sol[:, 1] = np.maximum(sol[:, 1], 0.0) # Volatiles Profile

    # حساب المتبقيات النهائية
    final_moisture_fraction = sol[-1, 0]
    final_volatiles_fraction = sol[-1, 1]

    # --- حساب كتلة الفحم الحيوي (Biochar) ---
    # الفحم الحيوي يتكون من: الرماد (ثابت) + كربون ثابت (نتج عن التفاعل) + أي مواد متطايرة لم تتحلل بعد
    # لكن للتبسيط في هذا النموذج: الكتلة الصلبة المتبقية = الكتلة الكلية - (الماء المتبخر + الغازات المتطايرة)
    
    # أولاً نحسب ما تم فقده
    moisture_lost_fraction = initial_moisture_fraction - final_moisture_fraction
    volatiles_lost_fraction = initial_volatiles_fraction - final_volatiles_fraction
    
    # الكتلة المتبقية الصلبة (Biochar Yield)
    # المعادلة: 1.0 (الكل) - الماء المفقود - الغازات المفقودة
    final_solid_fraction = 1.0 - moisture_lost_fraction - volatiles_lost_fraction
    
    # تحويل الكسور إلى كتل (kg) ونسب (%)
    yields_data = {
        "Biochar (Solid)": final_solid_fraction * initial_mass_kg,
        "Gases (Volatiles)": volatiles_lost_fraction * initial_mass_kg,
        "Water Vapor": moisture_lost_fraction * initial_mass_kg,
        "Ash (Inside Biochar)": ash_content * initial_mass_kg # للعلم فقط، هي جزء من البيوتشار
    }
    
    # إنشاء DataFrames للعرض
    yields_df = pd.DataFrame({
        "Mass (kg)": list(yields_data.values()),
        "Percentage (%)": [x / initial_mass_kg * 100 for x in yields_data.values()]
    }, index=yields_data.keys())

    # بروفايل الكتلة مع الزمن (للرسم البياني)
    # نحسب كتلة البيوتشار لحظياً
    # Biochar_t = 1 - (Moisture_lost_t + Volatiles_lost_t)
    moisture_lost_t = initial_moisture_fraction - sol[:, 0]
    volatiles_lost_t = initial_volatiles_fraction - sol[:, 1]
    solid_mass_fraction_t = 1.0 - moisture_lost_t - volatiles_lost_t
    
    mass_profile = pd.DataFrame({
        "Time (min)": t,
        "Moisture Fraction": sol[:, 0],
        "Volatiles Fraction": sol[:, 1],
        "Solid Product (Biochar)": solid_mass_fraction_t
    }).set_index("Time (min)")

    # تكوين الغازات (افتراضي بناء على البيانات)
    if yields_data["Gases (Volatiles)"] > 0:
        gas_total = yields_data["Gases (Volatiles)"]
        gas_comp = {
            "CO2": 0.45 * gas_total,
            "CO": 0.35 * gas_total,
            "CH4": 0.15 * gas_total,
            "H2": 0.05 * gas_total
        }
    else:
        gas_comp = {"CO2": 0, "CO": 0, "CH4": 0, "H2": 0}
        
    gas_df = pd.DataFrame.from_dict(gas_comp, orient='index', columns=['Mass (kg)'])

    return {
        "yields": yields_df,
        "mass_profile": mass_profile,
        "gas_composition": gas_df,
        "parameters": {
            "biomass": biomass,
            "temp": temp_C,
            "time": duration_min,
            "mass": initial_mass_kg,
            "k_drying": k_drying
        }
    }

# --- 3. واجهة التطبيق (Main App) ---
def main():
    st.set_page_config(page_title="Torrefaction Simulator", layout="wide")
    
    st.title("🔥 Torrefaction Process Simulator")
    st.markdown("Simulation of biomass torrefaction with physical mass balance.")
    
    # Sidebar Inputs
    st.sidebar.header("Settings")
    
    initial_mass = st.sidebar.number_input("Initial Biomass Mass (kg)", value=100.0, min_value=1.0)
    biomass_type = st.sidebar.selectbox("Biomass Type", list(EMPIRICAL_DATA.keys()))
    moisture_content = st.sidebar.slider("Initial Moisture (%)", 0.0, 60.0, 15.0)
    particle_size = st.sidebar.selectbox("Particle Size", list(SIZE_FACTOR.keys()))
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Operating Conditions")
    temperature = st.sidebar.slider("Temperature (°C)", 150, 350, 250)
    duration = st.sidebar.slider("Duration (min)", 10, 180, 60)
    
    # التحقق من المدخلات
    if (moisture_content/100 + EMPIRICAL_DATA[biomass_type]["Ash"]) > 1.0:
        st.error("Error: Moisture + Ash content cannot exceed 100%.")
        return

    # تشغيل المحاكاة
    results = simulate_torrefaction(biomass_type, moisture_content, temperature, duration, particle_size, initial_mass)
    
    # --- عرض النتائج ---
    
    # 1. Key Metrics
    st.subheader("Results Summary")
    col1, col2, col3 = st.columns(3)
    
    biochar_mass = results["yields"].loc["Biochar (Solid)", "Mass (kg)"]
    water_mass = results["yields"].loc["Water Vapor", "Mass (kg)"]
    gas_mass = results["yields"].loc["Gases (Volatiles)", "Mass (kg)"]
    
    col1.metric("Solid Biochar Yield", f"{biochar_mass:.2f} kg")
    col2.metric("Water Evaporated", f"{water_mass:.2f} kg")
    col3.metric("Gases Produced", f"{gas_mass:.2f} kg")
    
    # تنبيه إذا كان الماء لم يتبخر بالكامل
    if water_mass < (initial_mass * moisture_content / 100) * 0.99:
        st.warning(f"⚠️ Note: Drying is incomplete. Consider increasing time or temperature. (Evaporated: {water_mass:.1f} kg / Total Moisture: {initial_mass * moisture_content / 100:.1f} kg)")
    else:
        st.success("✅ Drying Complete: All initial moisture has evaporated.")

    st.markdown("---")

    # 2. Tabs for Details
    tab1, tab2, tab3 = st.tabs(["📊 Mass Balance", "📈 Process Kinetics", "📄 Report"])
    
    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Yields Data")
            st.dataframe(results["yields"].style.format("{:.2f}"))
        
        with col_b:
            st.subheader("Product Distribution")
            fig1, ax1 = plt.subplots()
            # نرسم فقط المكونات الرئيسية
            plot_data = results["yields"].loc[["Biochar (Solid)", "Gases (Volatiles)", "Water Vapor"]]
            ax1.pie(plot_data["Mass (kg)"], labels=plot_data.index, autopct='%1.1f%%', startangle=90)
            st.pyplot(fig1)
            
    with tab2:
        st.subheader("Mass Conversion Over Time")
        st.line_chart(results["mass_profile"])
        st.caption("Notice how Moisture drops to zero quickly, while Volatiles decrease more slowly based on reaction kinetics.")
        
    with tab3:
        st.subheader("Download Report")
        if st.button("Generate PDF"):
            pdf_file = generate_simple_pdf(results)
            st.download_button("Download PDF", data=pdf_file, file_name="torrefaction_report.pdf", mime="application/pdf")

# --- 4. PDF Generator (Simplified) ---
def generate_simple_pdf(results):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    
    elements.append(Paragraph("Torrefaction Simulation Report", styles['Title']))
    elements.append(Spacer(1, 12))
    
    # Parameters
    p = results['parameters']
    text = f"""
    <b>Biomass:</b> {p['biomass']}<br/>
    <b>Temperature:</b> {p['temp']} C<br/>
    <b>Duration:</b> {p['time']} min<br/>
    <b>Initial Mass:</b> {p['mass']} kg<br/>
    """
    elements.append(Paragraph(text, styles['Normal']))
    elements.append(Spacer(1, 12))
    
    # Yields Table
    data = [["Component", "Mass (kg)", "Percentage (%)"]]
    for idx, row in results['yields'].iterrows():
        data.append([idx, f"{row['Mass (kg)']:.2f}", f"{row['Percentage (%)']:.2f}"])
        
    t = Table(data, style=[
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
    ])
    elements.append(t)
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

if __name__ == "__main__":
    main()
