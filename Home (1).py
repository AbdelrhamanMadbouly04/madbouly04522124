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

# --- 1. Chemical and Empirical Constants ---
R_GAS = 8.314  # Universal Gas Constant (J/mol·K)

EMPIRICAL_DATA = {
    "Wood": {
        "A": 2.5e10, "Ea": 135000, "k_drying_base": 0.05,
        "Ash": 0.02, "Gas_Factor": 0.35
    },
    "Agricultural Waste": {
        "A": 5.0e11, "Ea": 150000, "k_drying_base": 0.07,
        "Ash": 0.08, "Gas_Factor": 0.45
    },
    "Municipal Waste": {
        "A": 1.0e12, "Ea": 165000, "k_drying_base": 0.10,
        "Ash": 0.15, "Gas_Factor": 0.55
    }
}

SIZE_FACTOR = {
    "Fine (<1mm)": 1.0,
    "Medium (1-5mm)": 0.85,
    "Coarse (>5mm)": 0.65
}

# --- 2. Simulation Function (simulate_torrefaction) ---
def simulate_torrefaction(biomass, moisture, temp_C, duration_min, size, initial_mass_kg):
    """Core torrefaction simulation logic using Arrhenius and particle size correction.
    Revised to enforce physical constraints: moisture cannot go below zero (hard stop),
    no negative fractions, and mass balance enforced at the end."""
    temp_K = temp_C + 273.15
    data = EMPIRICAL_DATA.get(biomass)
    if data is None:
        raise ValueError(f"Unknown biomass type: {biomass}")

    # kinetic constants
    k_devol_arrhenius = data["A"] * np.exp(-data["Ea"] / (R_GAS * temp_K))
    k_devol_eff = k_devol_arrhenius * SIZE_FACTOR.get(size, 1.0)
    k_drying = data["k_drying_base"]
    ash_content = data["Ash"]

    # safe guards for initial fractions
    initial_moisture_fraction = float(np.clip(moisture / 100.0, 0.0, 0.999))
    initial_volatiles_fraction = 1.0 - initial_moisture_fraction - ash_content
    if initial_volatiles_fraction < 0:
        # if ash + moisture exceed 1, renormalize moisture (keep ash fixed)
        initial_moisture_fraction = max(0.0, 1.0 - ash_content - 1e-6)
        initial_volatiles_fraction = 1.0 - initial_moisture_fraction - ash_content

    # ODE system: drying + devolatilization
    def model(y, t, k1, k2):
        m, v = y
        # hard stop behaviour: when moisture nearly zero, no further drying
        if m <= 0.0:
            d_m = 0.0
        else:
            d_m = -k1 * m
            # don't allow step to overshoot too far negative (helps numeric)
            if m + d_m * 1.0 < -1e-9:
                d_m = -m / 1.0  # conservative step, will be clipped later

        d_v = -k2 * v if v > 0.0 else 0.0
        return [d_m, d_v]

    # time vector (minutes)
    t = np.linspace(0, duration_min, 200)
    y0 = [initial_moisture_fraction, initial_volatiles_fraction]
    sol = odeint(model, y0, t, args=(k_drying, k_devol_eff))
    # clip small negatives from numerical solver
    sol[:, 0] = np.clip(sol[:, 0], 0.0, 1.0)
    sol[:, 1] = np.clip(sol[:, 1], 0.0, 1.0)

    # enforce monotonic non-increase of moisture (physically drying only)
    moisture_profile = np.minimum.accumulate(sol[:, 0])

    # recompute biochar profile and clip
    biochar_profile = 1.0 - moisture_profile - sol[:, 1] - ash_content
    biochar_profile = np.clip(biochar_profile, 0.0, 1.0)

    # final steady-state at end of process
    final_moisture = float(np.clip(moisture_profile[-1], 0.0, 1.0))
    final_volatiles_remaining = float(np.clip(sol[-1, 1], 0.0, 1.0))

    # enforce mass-balance: ash is fixed, ensure sum = 1
    final_biochar_fraction = 1.0 - ash_content - final_moisture - final_volatiles_remaining
    if final_biochar_fraction < 0:
        # if negative due to numerical or unrealistic inputs, set biochar=0 and
        # reassign what's left to volatiles_remaining (can't change ash)
        final_biochar_fraction = 0.0
        final_volatiles_remaining = 1.0 - ash_content - final_moisture
        final_volatiles_remaining = max(final_volatiles_remaining, 0.0)

    # clip again to be safe
    final_biochar_fraction = float(np.clip(final_biochar_fraction, 0.0, 1.0))
    final_volatiles_remaining = float(np.clip(final_volatiles_remaining, 0.0, 1.0))

    # compute lost fractions (bounded)
    initial_volatiles_fraction = float(initial_volatiles_fraction)
    final_volatiles_lost_fraction = float(np.clip(initial_volatiles_fraction - final_volatiles_remaining, 0.0, initial_volatiles_fraction))
    moisture_lost_fraction = float(np.clip(initial_moisture_fraction - final_moisture, 0.0, initial_moisture_fraction))

    # Build yields (percent & mass)
    yields_percent = pd.DataFrame({
        "Yield (%)": [
            (final_biochar_fraction + ash_content) * 100.0,
            final_volatiles_lost_fraction * 100.0,
            moisture_lost_fraction * 100.0,
            ash_content * 100.0
        ]},
        index=["Biochar (Solid) & Ash", "Non-Condensable Gases", "Moisture Loss (Water Vapor)", "Initial Ash Content"]
    )

    yields_mass = yields_percent.copy()
    yields_mass["Mass (kg)"] = yields_percent["Yield (%)"] * initial_mass_kg / 100.0
    yields_mass.drop(columns=["Yield (%)"], inplace=True)

    # Gas fraction (mass of gas produced relative to initial mass) using Gas_Factor
    gas_fraction = final_volatiles_lost_fraction * data.get("Gas_Factor", 1.0)
    gas_total_mass = gas_fraction * initial_mass_kg

    # if no gas produced, produce zero composition
    if gas_total_mass <= 0 or final_volatiles_lost_fraction < 1e-6:
        gas_comp_mass = {"CO2": 0.0, "CO": 0.0, "CH4": 0.0, "H2": 0.0}
        gas_composition_molar = pd.DataFrame({"Molar % in Dry Gas": [0.0, 0.0, 0.0, 0.0]}, index=["CO2", "CO", "CH4", "H2"])
    else:
        # distribute gas mass (these are empirical fractions of the gas mass)
        gas_comp_mass = {
            "CO2": 0.45 * gas_total_mass,
            "CO":  0.35 * gas_total_mass,
            "CH4": 0.15 * gas_total_mass,
            "H2":  0.05 * gas_total_mass
        }
        total = sum(gas_comp_mass.values())
        # mass-based percentage of each species in the produced dry gas
        gas_composition_molar = pd.DataFrame({
            "Molar % in Dry Gas": [(m / total) * 100.0 for m in gas_comp_mass.values()]
        }, index=list(gas_comp_mass.keys()))

    # mass profile DataFrame (time series)
    mass_profile = pd.DataFrame({
        "Time (min)": t,
        "Moisture Fraction": moisture_profile,
        "Volatiles Fraction": sol[:, 1],
        "Biochar Fraction": biochar_profile
    }).set_index("Time (min)")

    # final parameters (consistent keys)
    parameters = {
        "initial_mass_kg": initial_mass_kg,
        "moisture_%": moisture,
        "temperature_C": temp_C,
        "duration_min": duration_min,
        "size": size,
        "k_devol_eff": k_devol_eff,
        "k_drying": k_drying
    }

    return {
        "yields_percent": yields_percent,
        "yields_mass": yields_mass,
        "temp_profile": pd.DataFrame({"Temperature (°C)": temp_C * np.ones_like(t)}, index=t),
        "gas_composition_molar": gas_composition_molar,
        "gas_comp_mass": gas_comp_mass,
        "mass_profile": mass_profile,
        "final_fractions": {
            "moisture_fraction": final_moisture,
            "volatiles_remaining_fraction": final_volatiles_remaining,
            "biochar_fraction": final_biochar_fraction,
            "ash_fraction": ash_content
        },
        "parameters": parameters
    }

# --- 3. Streamlit Main App (main) ---
def main():
    # Streamlit Config
    st.set_page_config(page_title="Chemisco Pro Torrefaction Simulator", layout="wide", initial_sidebar_state="expanded")

    # Inject small CSS (optional)
    st.markdown("""
        <style>
            .stApp { padding-top: 10px; }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar Inputs
    with st.sidebar:
        st.header("⚙️ Input Parameters")

        initial_mass_kg = st.number_input("Initial Biomass Mass (kg)", min_value=1.0, value=100.0, step=10.0)
        biomass_type = st.selectbox("Biomass Type", list(EMPIRICAL_DATA.keys()))
        moisture_content = st.slider("Initial Moisture Content (%)", 0.0, 50.0, 10.0, step=1.0)
        particle_size = st.selectbox("Particle Size", list(SIZE_FACTOR.keys()))
        temperature = st.slider("Torrefaction Temperature (°C)", 200, 350, 275, step=5)
        duration = st.slider("Process Duration (min)", 10, 120, 45, step=5)

        ash_percent = EMPIRICAL_DATA[biomass_type]["Ash"] * 100
        st.info(f"Assumed Initial Ash Content: {ash_percent:.1f}%")

    # Quick validation
    if moisture_content / 100.0 + EMPIRICAL_DATA[biomass_type]["Ash"] > 1.0:
        st.error("**Input Error:** Initial Moisture and Ash content exceed 100%. Please adjust the parameters.")
        return

    # Run simulation
    results = simulate_torrefaction(biomass_type, moisture_content, temperature, duration, particle_size, initial_mass_kg)

    # Display
    st.title("🔥 Advanced Torrefaction Simulator")
    st.header("📊 Simulation Results & Analysis")
    tab1, tab2, tab3, tab4 = st.tabs(["Yields & Mass Balance", "Mass Conversion Kinetics", "Gas Composition", "PDF Report"])

    with tab1:
        st.subheader(f"Product Yields (Based on {initial_mass_kg:.0f} kg Input)")
        col_m1, col_m2, col_m3 = st.columns(3)

        # Fixed: get k_devol_eff from parameters
        k_rate = results["parameters"]["k_devol_eff"]

        biochar_mass_metric = results["yields_mass"].loc["Biochar (Solid) & Ash", "Mass (kg)"]
        col_m1.metric("⚖️ Total Solid Product (kg)", f"{biochar_mass_metric:.2f} kg", delta=f"{k_rate:.3e} min⁻¹ (k_devol_eff)")

        gas_mass_metric = results["yields_mass"].loc["Non-Condensable Gases", "Mass (kg)"]
        col_m2.metric("💨 Non-Condensable Gas Mass (kg)", f"{gas_mass_metric:.2f} kg")

        moisture_mass_metric = results["yields_mass"].loc["Moisture Loss (Water Vapor)", "Mass (kg)"]
        col_m3.metric("💧 Water Vapor Loss (kg)", f"{moisture_mass_metric:.2f} kg")

        st.markdown("---")

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.subheader("Yield Distribution Tables")
            st.markdown("##### 1. Mass Yields (kg)")
            st.dataframe(results["yields_mass"].style.format("{:.2f}"), use_container_width=True)
            st.markdown("##### 2. Mass Fractions (%)")
            st.dataframe(results["yields_percent"].style.format("{:.2f}"), use_container_width=True)

        with col_t2:
            st.subheader("Mass Balance Pie Chart")
            fig1, ax1 = plt.subplots(figsize=(6, 6))
            filtered_yields = results["yields_percent"].iloc[[0, 1, 2]]
            ax1.pie(filtered_yields["Yield (%)"].values, labels=filtered_yields.index, autopct='%1.1f%%', startangle=90, colors=['#8B4513', '#A9A9A9', '#ADD8E6'])
            ax1.axis('equal')
            st.pyplot(fig1)
            plt.close(fig1)

    with tab2:
        st.subheader("Mass Component Conversion Over Time")
        st.line_chart(results["mass_profile"])
        st.caption("The curves show Moisture and Volatiles decrease while Biochar fraction forms over time.")

    with tab3:
        st.subheader("Non-Condensable Dry Gas Composition")
        st.bar_chart(results["gas_composition_molar"])
        st.caption("Molar percentages of gaseous products from devolatilization (dry basis).")

    with tab4:
        st.subheader("Generate Comprehensive PDF Report")
        st.markdown("Click the button below to generate and download a detailed report of the simulation.")
        if st.button("⬇️ Generate PDF Report"):
            pdf_buffer = generate_pdf_report(results)
            st.download_button(
                label="Download Report (PDF)",
                data=pdf_buffer,
                file_name=f"Torrefaction_Report_{biomass_type}_{temperature}C.pdf",
                mime="application/pdf"
            )

# --- 4. PDF Report Generation Function (generate_pdf_report) ---
def generate_pdf_report(results):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        title="Torrefaction Report",
        leftMargin=inch, rightMargin=inch, topMargin=inch, bottomMargin=inch
    )
    styles = getSampleStyleSheet()
    elements = []

    # Header & Banner
    elements.append(Paragraph("<font size=14 color='#4CAF50'>CHEMISCO PRO TORREFACTION REPORT</font>", styles["Title"]))
    elements.append(Paragraph(f"Report Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", styles.get("Normal", styles["Normal"])))
    elements.append(Spacer(1, 0.25*inch))

    # 1. Parameters Table (use keys from results["parameters"])
    elements.append(Paragraph("1. Simulation Parameters & Kinetics", styles["Heading2"]))
    p = results["parameters"]
    param_data = [
        ["Parameter", "Value"],
        ["Initial Biomass Mass", f"{p['initial_mass_kg']:.0f} kg"],
        ["Moisture Content", f"{p['moisture_%']}%"],
        ["Temperature", f"{p['temperature_C']} °C"],
        ["Duration", f"{p['duration_min']} min"],
        ["Particle Size", p["size"]],
        ["Effective Devol. Rate (k_devol_eff)", f"{p['k_devol_eff']:.3e} min⁻¹"],
    ]
    param_table = Table(param_data, colWidths=[2.7*inch, 3*inch], style=[('GRID', (0,0), (-1,-1), 1, colors.black)])
    elements.append(param_table)
    elements.append(Spacer(1, 0.25*inch))

    # 2. Yields Tables
    elements.append(Paragraph("2. Product Yields", styles["Heading2"]))

    # Mass Yields Table
    elements.append(Paragraph("2.1. Mass Yields (kg)", styles["Heading3"]))
    mass_rows = [["Component", "Mass (kg)"]]
    for idx, row in results["yields_mass"].iterrows():
        mass_rows.append([idx, f"{row['Mass (kg)']:.2f}"])
    mass_table = Table(mass_rows, colWidths=[3.5*inch, 2*inch], style=[('GRID', (0,0), (-1,-1), 1, colors.black)])
    elements.append(mass_table)
    elements.append(Spacer(1, 0.1*inch))

    # Percentage Yields Table
    elements.append(Paragraph("2.2. Percentage Yields (%)", styles["Heading3"]))
    percent_rows = [["Component", "Yield (%)"]]
    for idx, row in results["yields_percent"].iterrows():
        percent_rows.append([idx, f"{row['Yield (%)']:.2f}"])
    percent_table = Table(percent_rows, colWidths=[3.5*inch, 2*inch], style=[('GRID', (0,0), (-1,-1), 1, colors.black)])
    elements.append(percent_table)
    elements.append(Spacer(1, 0.25*inch))

    # 3. Charts
    elements.append(Paragraph("3. Results Visualization", styles["Heading2"]))

    # Chart 1: Mass Conversion Plot
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    results["mass_profile"].plot(ax=ax3)
    ax3.set_title("Mass Component Conversion Over Time")
    ax3.set_xlabel("Time (min)")
    ax3.set_ylabel("Mass Fraction")
    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    imgdata3 = BytesIO()
    fig3.savefig(imgdata3, format='png', dpi=300, bbox_inches='tight')
    imgdata3.seek(0)
    elements.append(ReportImage(imgdata3, width=5.5*inch, height=3.7*inch))
    elements.append(Spacer(1, 0.25*inch))

    # Chart 2: Mass balance pie chart
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    filtered_yields = results["yields_percent"].iloc[[0, 1, 2]]
    ax1.pie(filtered_yields["Yield (%)"].values, labels=filtered_yields.index, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    plt.title("Mass Balance Distribution (%)")
    imgdata1 = BytesIO()
    fig1.savefig(imgdata1, format='png', dpi=300, bbox_inches='tight')
    imgdata1.seek(0)
    elements.append(ReportImage(imgdata1, width=3*inch, height=3*inch))
    elements.append(Spacer(1, 0.25*inch))

    # Chart 3: Gas composition bar chart
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    results["gas_composition_molar"].plot(kind='bar', ax=ax2, legend=False)
    ax2.set_title("Dry Gas Composition (Molar %)")
    ax2.set_ylabel("Molar %")
    ax2.set_xticklabels(results["gas_composition_molar"].index, rotation=0)
    imgdata2 = BytesIO()
    fig2.savefig(imgdata2, format='png', dpi=300, bbox_inches='tight')
    imgdata2.seek(0)
    elements.append(ReportImage(imgdata2, width=4*inch, height=3.2*inch))

    plt.close('all')

    doc.build(elements)
    buffer.seek(0)
    return buffer

if __name__ == "__main__":
    main()
