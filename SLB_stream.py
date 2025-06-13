import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import streamlit.components.v1 as components
from datetime import datetime, timedelta
from pathlib import Path

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

# --- (Aquí iría todo tu CSS en st.markdown, igual que ya lo tenías) ---
st.markdown("""
<style>
/* ... tu bloque completo de estilos ... */
</style>
""", unsafe_allow_html=True)


# --- Funciones de utilidad ---
@st.cache_data
def load_df_modelo():
    try:
        return pd.read_csv("df_modelo.csv")
    except FileNotFoundError:
        # Datos de ejemplo si no existe el archivo
        return pd.DataFrame({
            'conductor': ['Juan Pérez', 'Ana García', 'Carlos López'] * 10,
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
    try:
        return joblib.load("modelo_rendimiento.pkl")
    except Exception:
        return None

@st.cache_data
def load_df_malos_contexto():
    try:
        return pd.read_csv("df_malos_contexto.csv")
    except FileNotFoundError:
        return pd.DataFrame({
            'conductor': ['Juan Pérez', 'Ana García', 'Carlos López'] * 5,
            'Unidad': ['VEH001', 'VEH002', 'VEH003'] * 5,
            'no_estación_pemex': ['EST001', 'EST002', 'EST003'] * 5
        })

# --- Carga inicial ---
df_modelo  = load_df_modelo()
model_rend = load_model_rendimiento()
df_malos   = load_df_malos_contexto()

# --- Sidebar y navegación ---
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; margin-bottom:2rem;">
      <div style="font-size:2rem; font-weight:800; color:var(--primary);">SLB</div>
      <div style="font-size:0.9rem; color:var(--text-secondary);">Energy Analytics Platform</div>
    </div>
    """, unsafe_allow_html=True)

    pages = {
      "🏠 Introducción": "intro",
      "📊 Análisis de Datos": "analysis",
      "🗺️ Mapeo Topológico": "mapping",
      "🔮 Predicción": "prediction",
      "📈 Dashboard": "dashboard"
    }
    choice = st.selectbox("Selecciona sección", list(pages.keys()), index=0)
    current_page = pages[choice]
    st.markdown("---")
    st.markdown("""
    <div style="padding:1rem; background:var(--bg-secondary); border-radius:12px;">
      <div style="font-weight:600; color:var(--text-primary);">💡 Tip</div>
      <div style="font-size:0.85rem; color:var(--text-secondary);">
        Navega por las secciones para explorar el análisis completo.
      </div>
    </div>
    """, unsafe_allow_html=True)

# --- Header común (todas excepto Intro) ---
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

# --- Página: Introducción ---
if current_page == "intro":
    st.markdown("""
    <div style="background:var(--gradient-primary); padding:4rem; text-align:center; color:white; border-radius:0 0 30px 30px;">
      <h1>🔋 SLB Energy Analytics</h1>
      <p>Plataforma de análisis topológico para optimización energética</p>
    </div>
    """, unsafe_allow_html=True)

# --- Página: Análisis de Datos ---
elif current_page == "analysis":
    st.write("📈 **Análisis de datos**: aquí van tus tabs y gráficos…")

# --- Página: Mapeo Topológico ---
elif current_page == "mapping":
    st.markdown("""
    ### 🗺️ Análisis Topológico (TDA)
    Visualiza aquí tu **mapper_output_bueno.html** interactivo:
    """)
    html_path = Path("mapper_output_bueno.html")
    if html_path.exists():
        html_content = html_path.read_text(encoding="utf-8")
        # Inserta el HTML dentro del Streamlit
        components.html(html_content, height=700, scrolling=True)
    else:
        st.error(f"⚠️ No encontré `{html_path.name}` en tu directorio. Asegúrate de haberlo subido con `git add`.")

# --- Página: Predicción ---
elif current_page == "prediction":
    st.write("🔮 **Predicción**: formulario y resultados…")

# --- Página: Dashboard ---
elif current_page == "dashboard":
    st.write("📈 **Dashboard**: KPIs y gráficos…")

# --- Footer ---
st.markdown("---")
st.markdown("© 2025 SLB Energy Analytics Platform")
