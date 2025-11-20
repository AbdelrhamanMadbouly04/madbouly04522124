import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
)
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import tempfile, io, os, glob
import base64

# ----- CONFIG -----
st.set_page_config(page_title="Chemisco - Torrefaction", layout="wide", initial_sidebar_state="collapsed")

# ----- SESSION STATE INIT -----
if 'simulations' not in st.session_state:
    st.session_state.simulations = []

# --- Utility: find uploaded image ---
def find_first_file(containing):
    candidates = glob.glob("/mnt/data/*")
    for c in candidates:
        if containing.lower() in os.path.basename(c).lower():
            return c
    return ""

# Images
HERO_COVER = find_first_file("cover") or ""
BANNER_COVER = find_first_file("banner") or HERO_COVER
LOGO_PATH = find_first_file("logo") or ""

def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

HERO_B64 = img_to_base64(HERO_COVER) if HERO_COVER else ""
BANNER_B64 = img_to_base64(BANNER_COVER) if BANNER_COVER else ""

# Hero & Banner CSS
hero_css = f"""
<style>
.hero {{
  {"background-image: url('data:image/png;base64," + HERO_B64 + "');" if HERO_B64 else ""}
  background-size: cover;
  background-position: center;
  height: 40vh;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  text-shadow: 2px 2px #000;
}}
.hero h1 {{ font-size: 48px; margin:0; color:#FFD700; }}

.banner {{
  {"background-image: url('data:image/png;base64," + BANNER_B64 + "');" if BANNER_B64 else ""}
  background-size: cover;
  background-position: center;
  height: 12vh;
  display:flex;
  align-items:center;
  padding-left:2rem;
  color:#fff;
  text-shadow:1px 1px #000;
  border-radius:8px;
  margin-top:1rem;
  margin-bottom:1rem;
}}

.glass {{
    background: rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 18px;
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
    border: 1px solid rgba(255,255,255,0.12);
}}
</style>
"""
st.markdown(hero_css, unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>Chemisco Pro — Advanced Torrefaction</h1></div>', unsafe_allow_html=True)
st.markdown('<div class="banner"><h3>Torrefaction Simulator — Realistic process & analytics</h3></div>', unsafe_allow_html=True)

# ---------- Torrefaction simulation ----------
def simulate_torrefaction(waste_type, mass, moisture, temp, residence_time):
    water_loss = mass * (moisture / 100.0) * (1.0 - np.exp(-0.6 * residence_time))
    volatile_fraction = np.clip(0.30 + 0.12 * ((temp - 200.0) / 100.0), 0.0, 0.9)
    volatile_loss = max(0.0, (mass - water_loss) * volatile_fraction)
    ash_mass = mass * 0.05
    biochar_mass = max(0.0, mass - water_loss - volatile_loss - ash_mass)
    fixed_carbon = biochar_mass * 0.78

    return {
        'Biochar (kg)': biochar_mass,
        'Gas & Volatiles (kg)': volatile_loss,
        'Ash (kg)': ash_mass,
        'Fixed Carbon (kg)': fixed_carbon,
        'Water Loss (kg)': water_loss
    }

# ---------- ReportLab PDF ----------
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            super().showPage()
        super().save()

    def draw_page_number(self, page_count):
        self.setFont("Helvetica", 9)
        footer_text = f"Chemisco • Torrefaction Report • Page {self._pageNumber} of {page_count}"
        self.setFillColor(colors.HexColor("#666666"))
        self.drawRightString(19 * cm, 1 * cm, footer_text)

def _make_matplotlib_charts(sim):
    keys = ['Biochar (kg)', 'Gas & Volatiles (kg)', 'Ash (kg)', 'Fixed Carbon (kg)', 'Water Loss (kg)']
    values = [sim.get(k, 0.0) for k in keys]
    colors_list = ['#2E8B57', '#1E90FF', '#FFA500', '#808080', '#8B4513']

    pie_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig1, ax1 = plt.subplots(figsize=(3, 3))
    if sum(values) == 0: values = [1e-6] * len(values)
    ax1.pie(values, labels=keys, colors=colors_list, autopct=lambda pct: f"{pct:.1f}%", startangle=140, textprops={'fontsize': 6})
    ax1.axis('equal')
    fig1.savefig(pie_tmp.name, dpi=150, bbox_inches='tight', transparent=True)
    plt.close(fig1)

    bar_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig2, ax2 = plt.subplots(figsize=(4, 2))
    ax2.bar(keys, values, color=colors_list)
    ax2.set_xticklabels(keys, rotation=45, ha='right', fontsize=6)
    ax2.set_ylabel('kg', fontsize=8)
    fig2.savefig(bar_tmp.name, dpi=150, bbox_inches='tight', transparent=True)
    plt.close(fig2)
    return pie_tmp.name, bar_tmp.name

def create_pdf_report(sim, logo_path=LOGO_PATH):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2.5*cm)
    styles = getSampleStyleSheet()
    story = []
    
    title_style = ParagraphStyle('title', parent=styles['Title'], alignment=1, fontSize=28, textColor=colors.HexColor("#1E90FF"))
    subtitle_style = ParagraphStyle('subtitle', parent=styles['Heading2'], alignment=1, fontSize=14, textColor=colors.HexColor("#444444"))
    body_style = styles['BodyText']

    if logo_path and os.path.exists(logo_path):
        try:
            img = RLImage(logo_path, width=6*cm, height=6*cm)
            img.hAlign = 'CENTER'
            story.append(Spacer(1,0.5*cm))
            story.append(img)
        except Exception as e:
            print(f"Error loading logo: {e}")

    story.append(Spacer(1,0.3*cm))
    story.append(Paragraph("Chemisco Pro", title_style))
    story.append(Spacer(1,0.2*cm))
    story.append(Paragraph("Torrefaction Simulation Report", subtitle_style))
    story.append(Spacer(1,0.8*cm))
    story.append(Paragraph("Simulation results from Chemisco Torrefaction Simulator.", body_style))
    story.append(Spacer(1,1.8*cm))

    meta = [["Generated by","Chemisco Torrefaction Simulator"],["Report generated",pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")]]
    meta_table = Table(meta,colWidths=[5*cm,8*cm])
    meta_table.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'Helvetica'),('FONTSIZE',(0,0),(-1,-1),9),
                                    ('TEXTCOLOR',(0,0),(-1,-1),colors.HexColor("#333333")),
                                    ('BOTTOMPADDING',(0,0),(-1,-1),6)]))
    story.append(meta_table)
    story.append(PageBreak())

    story.append(Paragraph("Simulation Summary", styles['Heading2']))
    story.append(Spacer(1,0.2*cm))
    data=[["Parameter","Value"]]
    for k,v in sim.items():
        data.append([k,f"{v:.2f}" if isinstance(v,(int,float)) else str(v)])
    table = Table(data,colWidths=[9*cm,6*cm])
    table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor("#1E90FF")),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(-1,0),11),
        ('ALIGN',(0,0),(-1,-1),'LEFT'),
        ('BACKGROUND',(0,1),(-1,-1),colors.HexColor("#F7F7F7")),
        ('GRID',(0,0),(-1,-1),0.5,colors.HexColor("#DDDDDD")),
        ('BOTTOMPADDING',(0,0),(-1,0),8),
        ('LEFTPADDING',(0,0),(-1,-1),6),
    ]))
    story.append(table)
    story.append(Spacer(1,0.8*cm))

    try:
        pie_path, bar_path = _make_matplotlib_charts(sim)
        story.append(PageBreak())
        story.append(Paragraph("Visual Summary", styles['Heading2']))
        story.append(Spacer(1,0.3*cm))
        story.append(RLImage(pie_path,width=8*cm,height=8*cm))
        story.append(Spacer(1,0.5*cm))
        story.append(RLImage(bar_path,width=12*cm,height=5*cm))
    except Exception as e:
        story.append(Paragraph("Charts could not be generated.", body_style))
        print(f"Error generating charts: {e}")

    story.append(PageBreak())
    story.append(Paragraph("Technical details & notes", styles['Heading2']))
    story.append(Spacer(1,0.2*cm))
    story.append(Paragraph("Values are approximate simulation outputs for demonstration.", body_style))
    story.append(Spacer(1,0.6*cm))
    story.append(Paragraph("<i>Generated by Chemisco Torrefaction Simulator © 2025</i>", styles['Normal']))

    doc.build(story, canvasmaker=NumberedCanvas)

    try:
        for path in (pie_path,bar_path):
            if os.path.exists(path):
                os.unlink(path)
    except Exception as e:
        print(f"Error cleaning up temporary files: {e}")

    buffer.seek(0)
    return buffer

# ---------- App UI ----------
st.markdown('<div class="glass">', unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Input Parameters")

    # === Basic Inputs ===
    waste_type = st.selectbox("Waste Type", ['Municipal', 'Wood', 'Agricultural', 'Plastic'])
    if waste_type == 'Plastic':
        plastic_type = st.selectbox("Plastic Type", ['Mixed LDPE', 'PET', 'PP'])

    mass = st.number_input("Mass (kg)", min_value=1.0, max_value=10000.0, value=50.0, step=1.0, format="%.2f")
    moisture = st.slider("Moisture (%)", 0.0, 100.0, 15.0)
    temp = st.slider("Temperature (°C)", 200, 300, 250)
    residence_time = st.slider("Residence Time (hr)", 0.1, 5.0, 1.0)

    # === Advanced Settings ===
    if st.checkbox("Show advanced settings"):
        processing_cost_per_kg = st.number_input("Processing Cost per kg ($)", 0.01, 50.0, 1.0, format="%.2f")
        heating_rate = st.slider("Heating Rate (°C/min)", 1, 50, 10)
        reactor_type = st.selectbox("Reactor Type", ['Fixed Bed', 'Rotary', 'Fluidized'])
        atmosphere = st.selectbox("Atmosphere", ['Inert (N2)', 'Air', 'Steam'])
    else:
        processing_cost_per_kg = 1.0  # default

# === Run Simulation Button (outside sidebar) ===
if st.button("Run Simulation"):
    sim = {
        "Waste Type": waste_type,
        "Mass": mass,
        "Moisture": moisture,
        "Temperature": temp,
        "Residence Time": residence_time
    }

    sim_res = simulate_torrefaction(waste_type, mass, moisture, temp, residence_time)
    sim.update(sim_res)

    sim["Total Cost ($)"] = mass * processing_cost_per_kg

    st.session_state.simulations.append(sim)
    st.success("Simulation run added to dashboard.")

st.markdown('</div>', unsafe_allow_html=True)
# ---------- Dashboard ----------
if st.session_state.simulations:
    st.markdown("---")
    st.subheader("Dashboard — Simulations Overview")
    df=pd.DataFrame(st.session_state.simulations)
    st.dataframe(df.style.format("{:.2f}",subset=[c for c in df.columns if df[c].dtype==float]))

    latest=st.session_state.simulations[-1]
    kcols=st.columns(5)
    keys=['Biochar (kg)','Gas & Volatiles (kg)','Ash (kg)','Fixed Carbon (kg)','Total Cost ($)']
    kcolors=['#2E8B57','#1E90FF','#FFA500','#808080','#8B4513']
    for c,k,col_color in zip(kcols,keys,kcolors):
        c.metric(k,f"{latest.get(k,0):.2f}")

    # ---------- Compact Flow Diagram ----------
    st.subheader("Process Flow Diagram (Compact)")
    fig_block=go.Figure()
    blocks=[
        {"name":"Input Waste","x0":0,"x1":1.5,"y0":1.5,"y1":2.5,"color":"#8B4513"},
        {"name":"Drying","x0":2,"x1":3.5,"y0":1.5,"y1":2.5,"color":"#1E90FF"},
        {"name":"Torrefaction","x0":4,"x1":5.5,"y0":1.5,"y1":2.5,"color":"#FFA500"},
        {"name":"Products","x0":6,"x1":7.5,"y0":1.5,"y1":2.5,"color":"#2E8B57"}
    ]
    for block in blocks:
        fig_block.add_shape(type="rect",x0=block["x0"],x1=block["x1"],y0=block["y0"],y1=block["y1"],
                            line=dict(color="black",width=2),fillcolor=block["color"],layer="below")
        fig_block.add_annotation(x=(block["x0"]+block["x1"])/2,y=(block["y0"]+block["y1"])/2,
                                 text=f"<b>{block['name']}</b>",showarrow=False,font=dict(color="white",size=12))
    arrows=[(1.5,2,2,2),(3.5,2,4,2),(5.5,2,6,2)]
    for x0,y0,x1,y1 in arrows:
        fig_block.add_annotation(x=x1,y=y1,ax=x0,ay=y0,xref="x",yref="y",axref="x",ayref="y",
                                 showarrow=True,arrowhead=3,arrowsize=1.5,arrowwidth=2,arrowcolor="#333333")
    fig_block.update_xaxes(range=[-0.5,8],showticklabels=False,showgrid=False,zeroline=False)
    fig_block.update_yaxes(range=[1,3],showticklabels=False,showgrid=False,zeroline=False)
    fig_block.update_layout(height=200,margin=dict(l=10,r=10,t=10,b=10),paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_block,use_container_width=True)

    # ---------- PDF Report ----------
    if st.button("Print Report"):
        pdf_buffer=create_pdf_report(latest)
        st.download_button("Download Report (PDF)",data=pdf_buffer,file_name="torrefaction_report.pdf",mime="application/pdf")
