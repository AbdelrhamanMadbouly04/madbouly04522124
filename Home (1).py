import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportImage, Table
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
from reportlab.lib.units import inch
from reportlab.lib import colors

# --- 1. Chemical and Empirical Constants (UNCHANGED) ---
R_GAS = 8.314
EMPIRICAL_DATA = {
    "Wood": {"A": 2.5e10, "Ea": 135000, "k_drying_base": 0.05, "Ash": 0.02, "Gas_Factor": 0.35},
    "Agricultural Waste": {"A": 5.0e11, "Ea": 150000, "k_drying_base": 0.07, "Ash": 0.08, "Gas_Factor": 0.45},
    "Municipal Waste": {"A": 1.0e12, "Ea": 165000, "k_drying_base": 0.10, "Ash": 0.15, "Gas_Factor": 0.55}
}
SIZE_FACTOR = {"Fine (<1mm)": 1.0, "Medium (1-5mm)": 0.85, "Coarse (>5mm)": 0.65}

# --- 2. Static UI Components (UNCHANGED) ---
GLOBAL_CSS = """
<style>
    .stApp { padding-top: 20px; }
    .main-banner {
        background-color: #388E3C;
        padding: 30px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    .main-banner h1 { color: #FFFFFF; margin: 0; font-size: 2.5em; }
    .main-banner p { color: #C8E6C9; margin-top: 5px; font-size: 1.1em; }
    .st-emotion-cache-1na6f8g, .st-emotion-cache-1d391kg { background-color: #F0F8FF; }
    .st-emotion-cache-p5m8m8 { 
        border-radius: 10px;
        border-left: 5px solid #4CAF50; 
        padding: 10px;
        margin-bottom: 15px;
        background-color: #FFFFFF;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    [data-testid="stMetricValue"] {
        font-size: 28px;
        color: #388E3C; 
    }
    .bfd-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 30px 0 60px 0;
        position: relative;
    }
    .bfd-block {
        padding: 15px 25px;
        border: 3px solid #4CAF50; 
        border-radius: 6px;
        text-align: center;
        background-color: #E8F5E9; 
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        font-weight: bold;
        color: #1B5E20; 
        position: relative;
        min-width: 180px;
    }
    .bfd-block p { margin: 5px 0 0; font-size: 12px; font-weight: normal; }
    .bfd-stream {
        width: 70px;
        height: 3px;
        background-color: #4CAF50;
        position: relative;
    }
    .bfd-stream::before { 
        content: '';
        position: absolute;
        right: -10px;
        top: -5px;
        border-top: 6px solid transparent;
        border-bottom: 6px solid transparent;
        border-left: 10px solid #4CAF50;
    }
    .side-stream {
        position: absolute;
        left: 50%;
        transform: translateX(-50%);
        width: 3px;
        height: 40px;
        background-color: #FF9800; 
        bottom: -40px;
    }
    .side-stream-label {
        position: absolute;
        bottom: -65px;
        left: 50%;
        transform: translateX(-50%);
        font-size: 11px;
        white-space: nowrap;
        color: #FF9800;
    }
</style>
"""

# --- 3. Simulation Core Logic (UNCHANGED) ---
def simulate_torrefaction(biomass, moisture, temp_C, duration_min, size, initial_mass_kg):
    temp_K = temp_C + 273.15
    data = EMPIRICAL_DATA.get(biomass)
    k_devol_arrhenius = data["A"] * np.exp(-data["Ea"] / (R_GAS * temp_K))
    k_devol_eff = k_devol_arrhenius * SIZE_FACTOR.get(size)
    k_drying = data["k_drying_base"] 
    initial_moisture_frac = moisture / 100
    initial_ash_frac = data["Ash"]
    initial_volatiles_frac = 1.0 - initial_moisture_frac - initial_ash_frac
    mass_ash_kg = initial_mass_kg * initial_ash_frac
    
    def model(y, t, k1, k2):
        m_moist, m_vol = y
        d_moist = -k1 * m_moist if m_moist > 0.001 else 0
        d_vol = -k2 * m_vol
        return [d_moist, d_vol]
    
    t = np.linspace(0, duration_min, 100)
    y0 = [initial_moisture_frac, initial_volatiles_frac]
    sol = odeint(model, y0, t, args=(k_drying, k_devol_eff))
    sol[sol < 0] = 0
    moisture_curve = sol[:, 0] 
    volatiles_curve = sol[:, 1]
    fixed_carbon_frac_initial = 1.0 - initial_moisture_frac - initial_volatiles_frac - initial_ash_frac
    current_total_mass_fraction = moisture_curve + volatiles_curve + fixed_carbon_frac_initial + initial_ash_frac
    ash_concentration_percent = (initial_ash_frac / current_total_mass_fraction) * 100
    
    final_moisture_loss = initial_moisture_frac
    final_volatiles_remaining = volatiles_curve[-1]
    final_volatiles_lost = initial_volatiles_frac - final_volatiles_remaining
    final_solid_fraction = 1.0 - final_moisture_loss - final_volatiles_lost
    mass_biochar_total = final_solid_fraction * initial_mass_kg
    final_ash_percent = (mass_ash_kg / mass_biochar_total) * 100

    yields_percent = pd.DataFrame({
        "Yield (%)": [final_solid_fraction * 100, final_volatiles_lost * 100, final_moisture_loss * 100, initial_ash_frac * 100]},
        index=["Biochar (Solid Product)", "Non-Condensable Gases", "Moisture Loss (Water Vapor)", "Original Ash Content"]
    )
    yields_mass = yields_percent.copy()
    yields_mass["Mass (kg)"] = yields_percent["Yield (%)"] * initial_mass_kg / 100
    yields_mass.drop(columns=["Yield (%)"], inplace=True)

    mass_volatiles_remaining = final_volatiles_remaining * initial_mass_kg
    mass_fixed_carbon = fixed_carbon_frac_initial * initial_mass_kg
    
    solid_composition = pd.DataFrame({
        "Mass (kg)": [mass_fixed_carbon, mass_volatiles_remaining, mass_ash_kg]
    }, index=["Fixed Carbon", "Remaining Volatiles", "Ash"])

    gas_fraction = final_volatiles_lost * data["Gas_Factor"]
    gas_comp_mass = {
        "CO2": 0.45 * gas_fraction * initial_mass_kg,
        "CO": 0.35 * gas_fraction * initial_mass_kg,
        "CH4": 0.15 * gas_fraction * initial_mass_kg,
        "H2": 0.05 * gas_fraction * initial_mass_kg
    }
    gas_composition_molar = pd.DataFrame.from_dict(
        {k: v * 100 / final_volatiles_lost for k, v in gas_comp_mass.items() if final_volatiles_lost > 0.001}, 
        orient="index", columns=["Molar % in Dry Gas"]
    ).fillna(0)

    mass_profile = pd.DataFrame({
        "Time (min)": t,
        "Total Mass Yield (%)": current_total_mass_fraction * 100,
        "Ash Concentration in Solid (%)": ash_concentration_percent
    }).set_index("Time (min)")
    
    return {
        "yields_percent": yields_percent,
        "yields_mass": yields_mass,
        "solid_composition": solid_composition,
        "final_ash_percent": final_ash_percent,
        "gas_composition_molar": gas_composition_molar,
        "mass_profile": mass_profile,
        "k_devol_eff": k_devol_eff,
        "parameters": {
            "biomass": biomass, "moisture": moisture, "temperature": temp_C, 
            "duration": duration_min, "size": size, "initial_mass": initial_mass_kg
        }
    }

# --- 4. Main Streamlit App ---
def main():
    st.set_page_config(page_title="Chemisco Pro Torrefaction Simulator", layout="wide", initial_sidebar_state="expanded")
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    # 4.1. Sidebar (UNCHANGED)
    with st.sidebar:
        st.markdown("""
            <div style='text-align: center; padding: 15px; border-radius: 8px; background-color: #1B5E20;'>
                <h1 style='color: white; margin: 0; font-size: 1.8em;'>CHEMISCO PRO</h1>
                <p style='color: #A5D6A7; margin: 0; font-size: 0.9em;'>Torrefaction Process Simulator</p>
            </div>
            """, unsafe_allow_html=True)
        st.header("⚙️ Input Parameters")
        with st.expander("🌲 Biomass Properties", expanded=True):
            initial_mass_kg = st.number_input("Initial Biomass Mass (kg)", min_value=1.0, value=100.0, step=10.0)
            biomass_type = st.selectbox("Biomass Type", list(EMPIRICAL_DATA.keys()))
            moisture_content = st.slider("Initial Moisture Content (%)", 0.0, 50.0, 10.0, step=1.0)
            particle_size = st.selectbox("Particle Size", list(SIZE_FACTOR.keys()))
        with st.expander("🌡️ Process Conditions", expanded=True):
            temperature = st.slider("Torrefaction Temperature (°C)", 200, 350, 275, step=5)
            duration = st.slider("Process Duration (min)", 10, 120, 45, step=5)
            ash_percent_init = EMPIRICAL_DATA[biomass_type]["Ash"] * 100
            st.info(f"Initial Ash Content: **{ash_percent_init:.1f}%**")

    # 4.2. Main Content
    st.markdown("""
        <div class="main-banner">
            <h1>🔥 Advanced Torrefaction Simulator</h1>
            <p>Enhanced Kinetic Model for Process Optimization</p>
        </div>
        """, unsafe_allow_html=True)
    
    # BFD (UNCHANGED)
    st.subheader("Process Flow Block Diagram (BFD)")
    bfd_html = f"""
    <div class="bfd-container">
        <div class="bfd-block">
            FEED PREPARATION
            <p style="color: #1565C0;">Initial Mass: {initial_mass_kg:.0f} kg</p>
            <p style="color: #0277BD;">Moisture: {moisture_content:.1f}%</p>
        </div>
        <div class="bfd-stream"></div>
        <div class="bfd-block">
            DRYING & PREHEATING
            <p>100 °C - 200 °C</p>
            <div class="side-stream"></div>
            <div class="side-stream-label">Water Vapor</div>
        </div>
        <div class="bfd-stream"></div>
        <div class="bfd-block" style="border-color: #D32F2F; background-color: #FFCDD2; color: #B71C1C;">
            TORREFACTION REACTOR
            <p style="color: #B71C1C;">Temp: {temperature} °C</p>
            <p style="color: #B71C1C;">Duration: {duration} min</p>
            <div class="side-stream" style="background-color: #FFC107;"></div>
            <div class="side-stream-label" style="color: #FFC107;">Volatile Gases</div>
        </div>
        <div class="bfd-stream"></div>
        <div class="bfd-block" style="border-color: #388E3C; background-color: #C8E6C9; color: #1B5E20;">
            COOLING & PRODUCT
            <p>Torrefied Biochar</p>
        </div>
    </div>
    <div style="height: 40px;"></div>
    """
    st.markdown(bfd_html, unsafe_allow_html=True)
    
    if moisture_content / 100 + EMPIRICAL_DATA[biomass_type]["Ash"] > 1:
        st.error("**Input Error:** Initial Moisture and Ash content exceed 100%.")
        return 
        
    results = simulate_torrefaction(biomass_type, moisture_content, temperature, duration, particle_size, initial_mass_kg)
    
    # --- Display Results ---
    st.header("📊 Simulation Results & Analysis")
    tab1, tab2, tab3, tab4 = st.tabs(["Yields & Ash Enrichment", "Ash & Mass Kinetics", "Gas Composition", "PDF Report"])
    
    with tab1:
        st.subheader(f"Product Yields & Ash Enrichment")
        col_m1, col_m2, col_m3 = st.columns(3)
        
        biochar_mass = results["yields_mass"].loc["Biochar (Solid Product)", "Mass (kg)"]
        col_m1.metric("⚖️ Total Biochar Mass", f"{biochar_mass:.2f} kg")
        
        final_ash = results["final_ash_percent"]
        ash_increase = final_ash - ash_percent_init
        col_m2.metric("⚗️ Final Ash Concentration", f"{final_ash:.2f} %", delta=f"+{ash_increase:.2f}% (Enrichment)")
        
        moisture_loss = results["yields_mass"].loc["Moisture Loss (Water Vapor)", "Mass (kg)"]
        col_m3.metric("💧 Moisture Removed", f"{moisture_loss:.2f} kg")

        st.markdown("---")
        
        col_t1, col_t2 = st.columns(2)
        
        # --- MODIFIED SECTION FOR PIE CHARTS ---
        with col_t1:
            st.markdown("##### Final Biochar Composition (Solid Product Only)")
            st.caption("Notice how the Ash portion is larger here than in the raw material due to mass loss.")
            # Create fig with equal size (6x6) and transparent background
            fig_solid, ax_solid = plt.subplots(figsize=(6, 6))
            fig_solid.patch.set_alpha(0.0)
            ax_solid.patch.set_alpha(0.0)
            
            colors_solid = ['#4E342E', '#8D6E63', '#BDBDBD']
            ax_solid.pie(
                results["solid_composition"]["Mass (kg)"], 
                labels=results["solid_composition"].index, 
                autopct='%1.1f%%', 
                startangle=140, 
                colors=colors_solid,
                explode=(0, 0, 0.1)
            )
            # Set title color to be visible on dark background if needed, or rely on Streamlit theme
            ax_solid.set_title(f"Composition of the {biochar_mass:.1f} kg Biochar Produced", color='white')
            # Use facecolor='none' for transparency
            st.pyplot(fig_solid, facecolor='none')
        
        with col_t2:
            st.markdown("##### Global Mass Balance (Initial vs Output)")
            # Create fig with equal size (6x6) and transparent background
            fig1, ax1 = plt.subplots(figsize=(6, 6))
            fig1.patch.set_alpha(0.0)
            ax1.patch.set_alpha(0.0)
            
            filtered_yields = results["yields_percent"].iloc[[0, 1, 2]] 
            ax1.pie(filtered_yields["Yield (%)"].values, labels=filtered_yields.index, autopct='%1.1f%%', startangle=90, colors=['#795548', '#CFD8DC', '#B3E5FC'])
            ax1.set_title("Overall Process Mass Balance", color='white')
            # Use facecolor='none' for transparency
            st.pyplot(fig1, facecolor='none')

    with tab2:
        st.subheader("Ash Concentration & Mass Depletion Logic")
        # Create fig with transparent background
        fig_k, ax_mass = plt.subplots(figsize=(10, 5))
        fig_k.patch.set_alpha(0.0)
        ax_mass.patch.set_alpha(0.0)

        color_mass = 'tab:green'
        ax_mass.set_xlabel('Time (min)', color='white')
        ax_mass.set_ylabel('Total Mass Remaining (%)', color=color_mass)
        ax_mass.plot(results["mass_profile"].index, results["mass_profile"]["Total Mass Yield (%)"], color=color_mass, linewidth=2.5, label="Total Mass %")
        ax_mass.tick_params(axis='y', labelcolor=color_mass)
        ax_mass.tick_params(axis='x', labelcolor='white') # X-axis ticks white
        ax_mass.grid(True, linestyle='--', alpha=0.5)
        # Change spines color for better visibility
        for spine in ax_mass.spines.values(): spine.set_color('white')

        ax_ash = ax_mass.twinx()
        color_ash = 'tab:grey'
        # Lighter color for better visibility on dark background
        display_ash_color = '#E0E0E0' 
        ax_ash.set_ylabel('Ash Concentration in Solid (%)', color=display_ash_color, fontsize=12, weight='bold')
        ax_ash.plot(results["mass_profile"].index, results["mass_profile"]["Ash Concentration in Solid (%)"], color=color_ash, linewidth=3, linestyle='-', label="Ash % (Enrichment)")
        ax_ash.tick_params(axis='y', labelcolor=display_ash_color)
        for spine in ax_ash.spines.values(): spine.set_color('white')

        plt.title("Kinetic Logic: As Mass Decreases, Ash Concentration Increases", color='white')
        # Use facecolor='none' for transparency
        st.pyplot(fig_k, facecolor='none')
        st.caption("""
        **Interpretation:** The Green line shows the total mass of the biomass decreasing due to drying and devolatilization. 
        The **Grey Line** represents the **Ash Concentration**. As you can see, it rises according to the logical equation: 
        $Ash_{\%} = Mass_{Ash} / Mass_{CurrentSolid}$.
        """)

    with tab3:
        st.subheader("Gas Composition")
        st.bar_chart(results["gas_composition_molar"])

    with tab4:
        st.subheader("Download Report")
        if st.button("⬇️ Download PDF Report"):
            pdf_buffer = generate_pdf_report(results)
            st.download_button(
                label="Download Report",
                data=pdf_buffer,
                file_name=f"Torrefaction_Report.pdf",
                mime="application/pdf"
            )

# --- 5. PDF Report Generation Function (UNCHANGED) ---
def generate_pdf_report(results):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    elements.append(Paragraph("Torrefaction Simulation Report", styles["Title"]))
    elements.append(Spacer(1, 0.2*inch))
    p = results["parameters"]
    elements.append(Paragraph(f"Biomass: {p['biomass']} | Temp: {p['temperature']} C | Time: {p['duration']} min", styles["Normal"]))
    elements.append(Spacer(1, 0.2*inch))
    data = [["Component", "Mass (kg)", "Yield (%)"]]
    for idx, row in results["yields_percent"].iterrows():
        mass = results["yields_mass"].loc[idx, "Mass (kg)"]
        data.append([idx, f"{mass:.2f}", f"{row['Yield (%)']:.2f}"])
    t = Table(data, style=[('GRID', (0,0), (-1,-1), 1, colors.black)])
    elements.append(t)
    elements.append(Spacer(1, 0.5*inch))
    elements.append(Paragraph(f"Final Ash Concentration: {results['final_ash_percent']:.2f}%", styles["h3"]))
    doc.build(elements)
    buffer.seek(0)
    return buffer

if __name__ == "__main__":
    main()
