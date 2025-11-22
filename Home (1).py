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

# --- 1. Chemical and Empirical Constants ---
R_GAS = 8.314

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

# --- 2. Simulation Core Logic (Fixed) ---
def simulate_torrefaction(biomass, moisture, temp_C, duration_min, size, initial_mass_kg):
    """
    Simulation logic updated:
    1. Drying rate depends on Temperature (fast > 100C).
    2. Hard stop when moisture hits zero.
    """
    temp_K = temp_C + 273.15
    data = EMPIRICAL_DATA.get(biomass)
    
    # 1. Dynamic Drying Rate:
    # If temp > 100C, water evaporates fast. We model this linearily increasing with Temp overshoot.
    # This ensures water is gone quickly at 250C, unlike a fixed low rate.
    if temp_C < 100:
        k_drying = 0.05 
    else:
        k_drying = 0.1 + (temp_C - 100) * 0.005 

    # 2. Devolatilization Rate (Arrhenius)
    k_devol_arrhenius = data["A"] * np.exp(-data["Ea"] / (R_GAS * temp_K))
    k_devol_eff = k_devol_arrhenius * SIZE_FACTOR.get(size)
    
    ash_content = data["Ash"]

    # Define ODE System
    def model(y, t):
        m, v = y # m: moisture fraction, v: volatiles fraction
        
        # --- FIX: Drying Logic ---
        # If moisture is effectively zero, stop evaporating.
        if m <= 0.0001:
            d_m = 0.0
            m = 0.0 # Clamp to zero
        else:
            d_m = -k_drying * m # Exponential decay
            
        # --- Devolatilization Logic ---
        if v <= 0.0001:
            d_v = 0.0
        else:
            d_v = -k_devol_eff * v
            
        return [d_m, d_v]
    
    t = np.linspace(0, duration_min, 200)
    
    # Initial Fractions (Normalized)
    initial_moisture_fraction = moisture / 100.0
    initial_volatiles_fraction = 1.0 - initial_moisture_fraction - ash_content
    
    # Safety check
    if initial_volatiles_fraction < 0:
        initial_volatiles_fraction = 0
        initial_moisture_fraction = 1.0 - ash_content

    y0 = [initial_moisture_fraction, initial_volatiles_fraction]
        
    # Solve ODE
    sol = odeint(model, y0, t)
    
    # Clamp negative results from numerical noise
    sol[:, 0] = np.maximum(sol[:, 0], 0.0) # Moisture
    sol[:, 1] = np.maximum(sol[:, 1], 0.0) # Volatiles

    # Final States
    final_moisture_fraction = sol[-1, 0]
    final_volatiles_remaining = sol[-1, 1]
    
    # --- Calculate Mass Balance (The Fixed Part) ---
    # Lost fractions (Evaporated/Gassed out)
    # Since final_moisture stops at 0, moisture_lost cannot exceed initial_moisture
    moisture_lost_fraction = initial_moisture_fraction - final_moisture_fraction
    volatiles_lost_fraction = initial_volatiles_fraction - final_volatiles_remaining
    
    # Solid Yield (Biochar) = Everything strictly remaining solid
    # 1.0 - (Water Lost + Gas Lost)
    final_biochar_fraction = 1.0 - moisture_lost_fraction - volatiles_lost_fraction
    
    # Create DataFrames
    yields_data = {
        "Biochar (Solid)": final_biochar_fraction * initial_mass_kg,
        "Gases (Volatiles)": volatiles_lost_fraction * initial_mass_kg,
        "Water Vapor": moisture_lost_fraction * initial_mass_kg,
        "Ash (In Solid)": ash_content * initial_mass_kg
    }

    yields_mass = pd.DataFrame(
        {"Mass (kg)": list(yields_data.values())}, 
        index=yields_data.keys()
    )
    
    yields_percent = pd.DataFrame(
        {"Yield (%)": [x / initial_mass_kg * 100 for x in yields_data.values()]}, 
        index=yields_data.keys()
    )

    # Gas Composition
    if yields_data["Gases (Volatiles)"] > 0:
        gas_total = yields_data["Gases (Volatiles)"]
        gas_comp_mass = {
            "CO2": 0.45 * gas_total, "CO": 0.35 * gas_total,
            "CH4": 0.15 * gas_total, "H2": 0.05 * gas_total
        }
    else:
        gas_comp_mass = {"CO2": 0, "CO": 0, "CH4": 0, "H2": 0}

    gas_composition_molar = pd.DataFrame.from_dict(gas_comp_mass, orient='index', columns=['Mass (kg)'])
    # Normalize for simple molar view (approximate)
    if gas_composition_molar['Mass (kg)'].sum() > 0:
        gas_composition_molar['%'] = gas_composition_molar['Mass (kg)'] / gas_composition_molar['Mass (kg)'].sum() * 100

    # Mass Profile over time (for chart)
    # We calculate the solid mass at every time step
    moisture_t = sol[:, 0]
    volatiles_t = sol[:, 1]
    # Solid = Initial - Lost.  Lost = Initial - Current. 
    # So Solid = Current Moisture + Current Volatiles + Ash (Wait, no)
    # Actually: Solid Product includes Ash + Char + Remaining Volatiles + Remaining Moisture (wet basis)
    # But usually we plot Dry Biochar formation. Let's plot the phases:
    
    mass_profile = pd.DataFrame({
        "Time (min)": t,
        "Moisture": moisture_t,
        "Volatiles": volatiles_t,
        "Solid (Char+Ash)": 1.0 - moisture_t - volatiles_t # Simplified formation curve
    }).set_index("Time (min)")
    
    return {
        "yields_mass": yields_mass,
        "yields_percent": yields_percent,
        "gas_composition": gas_composition_molar,
        "mass_profile": mass_profile,
        "parameters": {
            "biomass": biomass, "moisture": moisture, "temperature": temp_C, 
            "duration": duration_min, "size": size, "initial_mass": initial_mass_kg
        }
    }

# --- 3. PDF Generator (Simplified) ---
def generate_pdf_report(results):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    
    elements.append(Paragraph("Torrefaction Simulation Report", styles['Title']))
    elements.append(Spacer(1, 12))
    
    # Parameters
    p = results['parameters']
    text = f"""
    <b>Biomass Type:</b> {p['biomass']}<br/>
    <b>Temperature:</b> {p['temperature']} C<br/>
    <b>Duration:</b> {p['duration']} min<br/>
    <b>Initial Mass:</b> {p['initial_mass']} kg<br/>
    """
    elements.append(Paragraph(text, styles['Normal']))
    elements.append(Spacer(1, 12))
    
    # Table
    data = [["Component", "Mass (kg)"]]
    for idx, row in results['yields_mass'].iterrows():
        data.append([idx, f"{row['Mass (kg)']:.2f}"])
        
    t = Table(data, style=[
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
    ])
    elements.append(t)
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

# --- 4. Main Streamlit App (Clean UI) ---
def main():
    st.set_page_config(page_title="Torrefaction Sim", layout="wide")
    
    st.title("🔥 Torrefaction Process Simulator")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        initial_mass = st.number_input("Initial Mass (kg)", 100.0, step=10.0)
        biomass_type = st.selectbox("Biomass Type", list(EMPIRICAL_DATA.keys()))
        moisture_content = st.slider("Moisture Content (%)", 0.0, 60.0, 15.0)
        particle_size = st.selectbox("Particle Size", list(SIZE_FACTOR.keys()))
        
        st.markdown("---")
        temperature = st.slider("Temperature (°C)", 150, 350, 250)
        duration = st.slider("Duration (min)", 10, 180, 60)

    # Validation
    if (moisture_content/100 + EMPIRICAL_DATA[biomass_type]["Ash"]) > 1.0:
        st.error("Error: Moisture + Ash > 100%")
        return

    # Simulation
    results = simulate_torrefaction(biomass_type, moisture_content, temperature, duration, particle_size, initial_mass)

    # Results Display
    st.subheader("Results Summary")
    
    col1, col2, col3 = st.columns(3)
    biochar_mass = results["yields_mass"].loc["Biochar (Solid)", "Mass (kg)"]
    water_mass = results["yields_mass"].loc["Water Vapor", "Mass (kg)"]
    gas_mass = results["yields_mass"].loc["Gases (Volatiles)", "Mass (kg)"]
    
    col1.metric("Solid Product", f"{biochar_mass:.2f} kg")
    col2.metric("Water Evaporated", f"{water_mass:.2f} kg")
    col3.metric("Gases Released", f"{gas_mass:.2f} kg")

    # Check Drying Status
    total_initial_water = initial_mass * moisture_content / 100
    if water_mass >= total_initial_water * 0.99:
        st.success("✅ Drying Complete: All moisture has been evaporated.")
    else:
        st.warning(f"⚠️ Drying Incomplete: Only {water_mass:.1f}kg of {total_initial_water:.1f}kg evaporated.")

    tab1, tab2, tab3 = st.tabs(["Mass Balance", "Kinetics Chart", "Download Report"])
    
    with tab1:
        st.dataframe(results["yields_mass"].style.format("{:.2f}"), use_container_width=True)
        
        # Simple Pie Chart
        fig, ax = plt.subplots()
        y = results["yields_mass"]["Mass (kg)"]
        # Remove Ash duplicate for chart clarity if needed, or keep
        # Here we plot the main outputs
        ax.pie(y, labels=y.index, autopct='%1.1f%%', startangle=90)
        st.pyplot(fig)
        
    with tab2:
        st.line_chart(results["mass_profile"])
        st.caption("Simulation of fraction decay over time. Notice Moisture hits zero and stops.")

    with tab3:
        if st.button("Generate PDF Report"):
            pdf = generate_pdf_report(results)
            st.download_button("Download PDF", pdf, "report.pdf", "application/pdf")

if __name__ == "__main__":
    main()
