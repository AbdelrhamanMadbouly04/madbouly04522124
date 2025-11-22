import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

# --- 2. Global CSS (UNCHANGED) ---
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
    /* BFD Styles */
    .bfd-container { display: flex; justify-content: center; align-items: center; margin: 30px 0 60px 0; position: relative; }
    .bfd-block { padding: 15px 25px; border: 3px solid #4CAF50; border-radius: 6px; text-align: center; background-color: #E8F5E9; box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15); font-weight: bold; color: #1B5E20; position: relative; min-width: 180px; }
    .bfd-block p { margin: 5px 0 0; font-size: 12px; font-weight: normal; }
    .bfd-stream { width: 70px; height: 3px; background-color: #4CAF50; position: relative; }
    .bfd-stream::before { content: ''; position: absolute; right: -10px; top: -5px; border-top: 6px solid transparent; border-bottom: 6px solid transparent; border-left: 10px solid #4CAF50; }
    .side-stream { position: absolute; left: 50%; transform: translateX(-50%); width: 3px; height: 40px; background-color: #FF9800; bottom: -40px; }
    .side-stream-label { position: absolute; bottom: -65px; left: 50%; transform: translateX(-50%); font-size: 11px; white-space: nowrap; color: #FF9800; }
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
    st.set_page_config(page_title="Chemisco Pro", layout="wide", initial_sidebar_state="expanded")
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    # Sidebar
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

    # Main Banner
    st.markdown("""
        <div class="main-banner">
            <h1>🔥 Advanced Torrefaction Simulator</h1>
            <p>Enhanced Kinetic Model for Process Optimization</p>
        </div>
        """, unsafe_allow_html=True)
    
    # BFD
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
        
        # --- PLOTLY CHARTS (Donut Style) ---
        with col_t1:
            st.markdown("##### Final Biochar Composition")
            st.caption("Solid Product Breakdown")
            
            df_solid = results["solid_composition"].reset_index()
            df_solid.columns = ["Component", "Mass (kg)"]
            
            fig1 = px.pie(df_solid, values='Mass (kg)', names='Component', hole=0.5,
                          color='Component',
                          color_discrete_map={
                              "Fixed Carbon": "#3E2723", 
                              "Remaining Volatiles": "#8D6E63",
                              "Ash": "#B0BEC5"
                          })
            
            fig1.update_layout(
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                margin=dict(t=20, b=50, l=10, r=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            fig1.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig1, use_container_width=True)

        with col_t2:
            st.markdown("##### Global Mass Balance")
            st.caption("Initial Input vs. Final Output")
            
            filtered_yields = results["yields_percent"].iloc[[0, 1, 2]].reset_index()
            filtered_yields.columns = ["Component", "Yield (%)"]
            
            fig2 = px.pie(filtered_yields, values='Yield (%)', names='Component', hole=0.5,
                          color='Component',
                          color_discrete_map={
                              "Biochar (Solid Product)": "#5D4037",
                              "Non-Condensable Gases": "#78909C",
                              "Moisture Loss (Water Vapor)": "#81D4FA"
                          })
            
            fig2.update_layout(
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                margin=dict(t=20, b=50, l=10, r=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            fig2.update_traces(textposition='inside', textinfo='percent')
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.subheader("Ash Concentration & Mass Depletion Kinetics")
        
        # --- FIXED DUAL-AXIS CHART (No secondary_y argument) ---
        fig_dual = go.Figure()

        # Line 1: Total Mass (Left Axis)
        fig_dual.add_trace(go.Scatter(
            x=results["mass_profile"].index,
            y=results["mass_profile"]["Total Mass Yield (%)"],
            name="Total Mass %",
            line=dict(color="#4CAF50", width=3), 
            yaxis="y1"
        ))

        # Line 2: Ash Concentration (Right Axis)
        fig_dual.add_trace(go.Scatter(
            x=results["mass_profile"].index,
            y=results["mass_profile"]["Ash Concentration in Solid (%)"],
            name="Ash Concentration %",
            line=dict(color="#FF5252", width=3, dash='dot'),
            yaxis="y2"
        ))

        # Corrected Layout - All Grid settings moved inside update_layout
        fig_dual.update_layout(
            title="Dynamic Ash Enrichment Logic",
            xaxis=dict(title="Time (min)", showgrid=False),
            
            # Left Axis (Primary)
            yaxis=dict(
                title=dict(text="Total Mass Remaining (%)", font=dict(color="#4CAF50")),
                tickfont=dict(color="#4CAF50"),
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(255,255,255,0.1)'
            ),
            
            # Right Axis (Secondary)
            yaxis2=dict(
                title=dict(text="Ash Concentration (%)", font=dict(color="#FF5252")),
                tickfont=dict(color="#FF5252"),
                overlaying="y",
                side="right",
                showgrid=False
            ),
            
            legend=dict(x=0.1, y=1.1, orientation="h"),
            hovermode="x unified",
            paper_bgcolor='rgba(0,0,0,0)', # Transparent
            plot_bgcolor='rgba(0,0,0,0)',
            height=450
        )

        st.plotly_chart(fig_dual, use_container_width=True)
        
        st.info("""
        **Logic Explanation:** The green line drops as moisture and volatiles leave the biomass. 
        Since Ash is inert (does not react), its *concentration* (Red Dotted Line) must mathematically increase as the total mass decreases.
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
