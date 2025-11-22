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
    # المعادلة: سرعة التبخير تزيد خطياً مع زيادة الحرارة فوق 100 درجة
    if temp_C < 100:
        k_drying = 0.05 # بطيء جدا تحت 100 (افتراضي)
    else:
        # كلما زادت الحرارة فوق 100، زاد المعدل بشكل كبير
        # هذا يضمن أن الماء يتبخر كلياً في وقت قصير عند 250 درجة مثلاً
        k_drying = 0.1 + (temp_C - 100) * 0.005 

    # 2. ثوابت التفاعل الكيميائي (Devolatilization)
    k_devol_arrhenius = data["A"] * np.exp(-data["Ea"] / (R_GAS * temp_K))
    k_devol_eff = k_devol_arrhenius * SIZE
