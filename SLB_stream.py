import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import os

# --- Configuración de la página ---
st.set_page_config(
    page_title="SLB • Energy Analytics Platform",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.slb.com/sustainability',
        'About': "### SLB Energy Analytics Platform\n*Análisis topológico avanzado para optimización energética*"
    }
)

# --- CSS Moderno y Responsivo ---
st.markdown("""
<style>
/* (incluye aquí todo tu bloque de estilos CSS tal cual lo tenías) */
</style>
""", unsafe_allow_html=True)

# --- Funciones de utilidad ---
@st.cache_data
def load_df_modelo():
    try:
        return pd.read_csv("df_modelo.csv")
    except:
        # Datos de ejemplo si no existe el archivo
        return pd.DataFrame({
            'conductor': ['Juan Pérez', 'Ana García', 'Carlos López'] * 10,
            'Unidad': ['VEH001', 'VEH002', 'VEH003'] * 10,
            'vehículo': ['VEH001', 'VEH002', 'VEH003'] * 10,
            'division': ['Norte', 'Sur', 'Centro'] * 10,
            'bl': ['BL001', 'BL002', 'BL003'] * 10,
            'mercancía': ['Petróleo', 'Gas', 'Químicos'] * 10,
            'no_estación_pemex': ['EST001', 'EST002', 'EST003'] * 10,
            'conductor_score': np.random.rand(30),
            'vehiculo_score': np.random.rand(30),
            'rend_cond_mean': np.random.rand(30),
            'rend_veh_mean': np.random.rand(30)
        })

@st.cache_resource
def load_model_rendimiento():
    # Depuración: carpeta de script y ficheros
    base = Path(__file__).parent
    st.write("🗂️ Files in script folder:", os.listdir(base))

    model_path = base / "modelo_rendimiento.pkl"
    if not model_path.exists():
        st.error(f"⚠️ Modelo No Disponible: no encontré {model_path.name}")
        return None

    try:
        return joblib.load(model_path)
    except Exception as e:
        # Capturamos el error concreto de pickle
        st.error(f"❌ Error cargando modelo: {e}")
        return None


def load_df_malos_contexto():
    try:
        return pd.read_csv("df_malos_contexto.csv")
    except:
        return pd.DataFrame({
            'conductor': ['Juan Pérez', 'Ana García', 'Carlos López'] * 5,
            'Unidad': ['VEH001', 'VEH002', 'VEH003'] * 5,
            'no_estación_pemex': ['EST001', 'EST002', 'EST003'] * 5
        })

# --- Carga inicial ---
df_modelo   = load_df_modelo()
model_rend  = load_model_rendimiento()
df_malos    = load_df_malos_contexto()

# --- Sidebar Navigation ---
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; margin-bottom:2rem;">
      <div style="font-size:2rem; font-weight:800; color:var(--primary);">SLB</div>
      <div style="font-size:0.9rem; color:var(--text-secondary);">Energy Analytics Platform</div>
    </div>""", unsafe_allow_html=True)

    pages = {
      "🏠 Introducción": "intro",
      "📊 Análisis de Datos": "analysis",
      "🗺️ Mapeo Topológico": "mapping",
      "🔮 Predicción": "prediction",
      "📈 Dashboard": "dashboard"
    }
    choice = st.selectbox("", list(pages.keys()), index=0)
    current_page = pages[choice]
    st.markdown("---")
    st.markdown("""
      <div style="padding:1rem; background:var(--bg-secondary); border-radius:12px;">
        <div style="font-weight:600; color:var(--text-primary); margin-bottom:0.5rem;">💡 Tip</div>
        <div style="font-size:0.85rem; color:var(--text-secondary);">
          Navega por las secciones para explorar el análisis completo.
        </div>
      </div>
    """, unsafe_allow_html=True)

# --- HEADER (todas excepto intro) ---
if current_page != "intro":
    st.markdown("""
    <div class="header-container">
      <div class="header-content">
        <div class="header-logo">SLB</div>
        <div class="header-title">Energy Analytics Platform</div>
        <div class="header-subtitle">Optimización energética mediante análisis topológico avanzado</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# --- PÁGINA: INTRODUCCIÓN ---
if current_page == "intro":
    st.markdown("""
    <div style="background:var(--gradient-primary); padding:4rem 2rem 5rem; margin:-1rem -1rem 3rem; border-radius:0 0 30px 30px; text-align:center; color:white;">
      <h1 style="font-size:3.5rem; font-weight:800; margin-bottom:1rem;">🔋 SLB Energy Analytics</h1>
      <p style="font-size:1.5rem; opacity:0.9;">Plataforma de análisis topológico para optimización energética</p>
    </div>
    """, unsafe_allow_html=True)
    # Aquí seguiría todo tu contenido de introducción…
    
# --- PÁGINA: ANÁLISIS DE DATOS ---
elif current_page == "analysis":
    # (Tu código de tabs con gráficos)
    st.write("Análisis de datos… (aquí irían tus tabs y gráficos)")

# --- PÁGINA: MAPEO TOPOLÓGICO ---
elif current_page == "mapping":
    # (Tu código para renderizar mapper_output_bueno.html)
    st.write("Mapeo Topológico…")

# --- PÁGINA: PREDICCIÓN ---
elif current_page == "prediction":
    st.markdown("""
    <div class="content-card">
      <div class="content-card-header">
        <div class="content-card-icon">🔮</div>
        <div>
          <div class="content-card-title">Sistema de Predicción de Eficiencia</div>
          <div class="content-card-subtitle">Modelo ML + TDA (89% de precisión)</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if model_rend is None:
        st.markdown("""
        <div style="background:linear-gradient(135deg,var(--error)0%,#DC2626100%); color:white; padding:2rem; border-radius:16px; text-align:center;">
          <h3>⚠️ Modelo No Disponible</h3>
          <p>El archivo modelo_rendimiento.pkl no se encontró en el directorio.</p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # Formulario de entrada…
    conductores = sorted(df_modelo["conductor"].astype(str).unique())
    vehiculos   = sorted(df_modelo["vehículo"].astype(str).unique())
    divisiones  = sorted(df_modelo["division"].astype(str).unique())
    bls         = sorted(df_modelo["bl"].astype(str).unique())
    mercas      = sorted(df_modelo["mercancía"].astype(str).unique())
    estaciones  = sorted(df_modelo["no_estación_pemex"].astype(str).unique())

    col1, col2, col3 = st.columns(3)
    with col1:
        conductor = st.selectbox("👨‍💼 Conductor", conductores)
        vehiculo  = st.selectbox("🚛 Vehículo", vehiculos)
    with col2:
        division  = st.selectbox("🏢 División", divisiones)
        bl        = st.selectbox("📋 BL", bls)
    with col3:
        mercancia = st.selectbox("📦 Mercancía", mercas)
        estacion  = st.selectbox("⛽ Estación", estaciones)

    if st.button("🚀 Predecir Eficiencia"):
        cs  = df_modelo[df_modelo["conductor"] == conductor]["conductor_score"].mean()
        vs  = df_modelo[df_modelo["vehículo"] == vehiculo]["vehiculo_score"].mean()
        rcm = df_modelo[df_modelo["conductor"] == conductor]["rend_cond_mean"].mean()
        rvm = df_modelo[df_modelo["vehículo"] == vehiculo]["rend_veh_mean"].mean()

        X = pd.DataFrame([{
            "conductor_score": cs or 0.5,
            "vehiculo_score":  vs or 0.5,
            "rend_cond_mean":  rcm or 10.0,
            "rend_veh_mean":   rvm or 10.0,
            "division":        division,
            "bl":              bl,
            "mercancía":       mercancia,
            "no_estación_pemex": estacion
        }])

        pred  = model_rend.predict(X)[0]
        proba = model_rend.predict_proba(X)[0][1] if hasattr(model_rend, "predict_proba") else None

        color = "#10B981" if pred==1 else "#EF4444"
        icon  = "✅" if pred==1 else "❌"
        text  = "EFICIENTE" if pred==1 else "INEFICIENTE"

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{color}0%,{color}100%); color:white; padding:2rem; border-radius:16px; text-align:center;">
          <div style="font-size:3rem;">{icon}</div>
          <h2>{text}</h2>
          {f"<p>Probabilidad de ineficiencia: {proba:.1%}</p>" if proba is not None else ""}
        </div>
        """, unsafe_allow_html=True)

# --- PÁGINA: DASHBOARD ---
elif current_page == "dashboard":
    st.write("Dashboard… (aquí tu código de KPI y gráficos)")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:var(--text-muted); padding:2rem 0;">
  © 2025 SLB Energy Analytics Platform
</div>
""", unsafe_allow_html=True)
