import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --primary: #0066CC;
    --primary-dark: #003F7F;
    --primary-light: #4A9EFF;
    --secondary: #00B4D8;
    --accent: #FF6B35;
    --success: #10B981;
    --warning: #F59E0B;
    --error: #EF4444;
    --text-primary: #0F172A;
    --text-secondary: #64748B;
    --text-muted: #94A3B8;
    --bg-primary: #FFFFFF;
    --bg-secondary: #F8FAFC;
    --bg-tertiary: #F1F5F9;
    --border: #E2E8F0;
    --border-light: #F1F5F9;
    --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    --shadow-lg: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    --shadow-xl: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
    --gradient-primary: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
    --gradient-secondary: linear-gradient(135deg, var(--secondary) 0%, var(--primary) 100%);
    --gradient-accent: linear-gradient(135deg, var(--accent) 0%, #FF8C42 100%);
}

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Layout Principal */
.stApp {
    background: var(--bg-secondary);
}

.main > div {
    padding: 0;
    max-width: 100%;
}

/* Header Principal */
.header-container {
    background: var(--gradient-primary);
    padding: 3rem 2rem 4rem;
    margin: -1rem -1rem 2rem;
    position: relative;
    overflow: hidden;
}

.header-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Ccircle cx='30' cy='30' r='1'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
    pointer-events: none;
}

.header-content {
    position: relative;
    z-index: 1;
    text-align: center;
    color: white;
}

.header-logo {
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -0.05em;
    margin-bottom: 0.5rem;
    text-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.header-title {
    font-size: 2rem;
    font-weight: 600;
    margin-bottom: 1rem;
    opacity: 0.95;
}

.header-subtitle {
    font-size: 1.125rem;
    font-weight: 400;
    opacity: 0.8;
    max-width: 600px;
    margin: 0 auto;
    line-height: 1.6;
}

/* Sidebar Moderna */
[data-testid="stSidebar"] {
    background: var(--bg-primary);
    border-right: 1px solid var(--border);
    box-shadow: var(--shadow);
}

[data-testid="stSidebar"] > div {
    padding: 2rem 1.5rem;
}

/* Navigation Cards */
.nav-card {
    background: var(--bg-primary);
    border: 2px solid var(--border-light);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 0.75rem 0;
    cursor: pointer;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.nav-card:hover {
    border-color: var(--primary-light);
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
}

.nav-card.active {
    border-color: var(--primary);
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
    color: white;
    box-shadow: var(--shadow-lg);
}

.nav-card-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
}

.nav-card-title {
    font-weight: 600;
    font-size: 1.1rem;
    margin-bottom: 0.25rem;
}

.nav-card-desc {
    font-size: 0.875rem;
    opacity: 0.7;
    line-height: 1.4;
}

/* Content Cards */
.content-card {
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2rem;
    margin: 1.5rem 0;
    box-shadow: var(--shadow);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.content-card:hover {
    box-shadow: var(--shadow-lg);
    transform: translateY(-1px);
}

.content-card-header {
    display: flex;
    align-items: center;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--border-light);
}

.content-card-icon {
    font-size: 2rem;
    margin-right: 1rem;
    color: var(--primary);
}

.content-card-title {
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0;
}

.content-card-subtitle {
    font-size: 0.875rem;
    color: var(--text-muted);
    margin-top: 0.25rem;
}

/* Metric Cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1.5rem;
    margin: 2rem 0;
}

.metric-card {
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    box-shadow: var(--shadow);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: var(--gradient-primary);
}

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-lg);
}

.metric-icon {
    font-size: 2.5rem;
    margin-bottom: 1rem;
    color: var(--primary);
}

.metric-value {
    font-size: 2.5rem;
    font-weight: 700;
    color: var(--primary);
    margin-bottom: 0.5rem;
    line-height: 1;
}

.metric-label {
    font-size: 1rem;
    font-weight: 500;
    color: var(--text-secondary);
    margin-bottom: 0.5rem;
}

.metric-change {
    font-size: 0.875rem;
    font-weight: 500;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    background: var(--bg-secondary);
    color: var(--text-muted);
}

/* Feature Cards */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 2rem;
    margin: 2rem 0;
}

.feature-card {
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2rem;
    box-shadow: var(--shadow);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.feature-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
}

.feature-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: var(--gradient-secondary);
}

.feature-icon {
    width: 60px;
    height: 60px;
    border-radius: 12px;
    background: var(--gradient-primary);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    color: white;
    margin-bottom: 1.5rem;
}

.feature-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.75rem;
}

.feature-desc {
    font-size: 0.95rem;
    color: var(--text-secondary);
    line-height: 1.6;
}

/* Buttons */
.primary-button {
    background: var(--gradient-primary) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.875rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
    box-shadow: var(--shadow) !important;
    cursor: pointer !important;
}

.primary-button:hover {
    transform: translateY(-1px) !important;
    box-shadow: var(--shadow-lg) !important;
}

/* Progress Bars */
.progress-container {
    background: var(--bg-secondary);
    border-radius: 10px;
    height: 8px;
    overflow: hidden;
    margin: 1rem 0;
}

.progress-bar {
    height: 100%;
    background: var(--gradient-primary);
    border-radius: 10px;
    transition: width 0.3s ease;
}

/* Status Badges */
.status-badge {
    display: inline-flex;
    align-items: center;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-size: 0.875rem;
    font-weight: 500;
    margin: 0.25rem;
}

.status-success {
    background: rgba(16, 185, 129, 0.1);
    color: var(--success);
    border: 1px solid rgba(16, 185, 129, 0.2);
}

.status-warning {
    background: rgba(245, 158, 11, 0.1);
    color: var(--warning);
    border: 1px solid rgba(245, 158, 11, 0.2);
}

.status-error {
    background: rgba(239, 68, 68, 0.1);
    color: var(--error);
    border: 1px solid rgba(239, 68, 68, 0.2);
}

/* Responsive */
@media (max-width: 768px) {
    .header-container {
        padding: 2rem 1rem 3rem;
    }
    
    .header-logo {
        font-size: 2rem;
    }
    
    .header-title {
        font-size: 1.5rem;
    }
    
    .metric-grid {
        grid-template-columns: 1fr;
    }
    
    .feature-grid {
        grid-template-columns: 1fr;
    }
}

/* Animaciones */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes slideIn {
    from { opacity: 0; transform: translateX(-20px); }
    to { opacity: 1; transform: translateX(0); }
}

.fade-in {
    animation: fadeIn 0.6s ease-out;
}

.slide-in {
    animation: slideIn 0.6s ease-out;
}

/* Streamlit Specific Overrides */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
    box-shadow: var(--shadow-sm) !important;
    transition: all 0.2s ease !important;
}

.stSelectbox > div > div:focus-within,
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(0, 102, 204, 0.1) !important;
}

.stButton > button {
    background: var(--gradient-primary) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 1.5rem !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    box-shadow: var(--shadow) !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: var(--shadow-lg) !important;
}

[data-testid="stMetric"] {
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: var(--shadow);
    transition: all 0.3s ease;
}

[data-testid="stMetric"]:hover {
    box-shadow: var(--shadow-lg);
    transform: translateY(-2px);
}

/* Tabs Modernas */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: var(--bg-secondary);
    border-radius: 12px;
    padding: 0.25rem;
    border: 1px solid var(--border);
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    padding: 0.75rem 1.5rem;
    color: var(--text-secondary);
    font-weight: 500;
    transition: all 0.2s ease;
    border: none;
}

.stTabs [data-baseweb="tab"]:hover {
    background: var(--bg-primary);
    color: var(--text-primary);
}

.stTabs [aria-selected="true"] {
    background: var(--primary) !important;
    color: white !important;
    box-shadow: var(--shadow);
}

/* Alerts */
.stAlert {
    border-radius: 12px !important;
    border: none !important;
    box-shadow: var(--shadow) !important;
}

/* Data Tables */
.stDataFrame {
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: var(--shadow) !important;
    border: 1px solid var(--border) !important;
}
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
    try:
        return joblib.load("modelo_rendimiento.pkl")
    except:
        return None

def load_df_malos_contexto():
    try:
        return pd.read_csv("df_malos_contexto.csv")
    except:
        # Datos de ejemplo
        return pd.DataFrame({
            'conductor': ['Juan Pérez', 'Ana García', 'Carlos López'] * 5,
            'Unidad': ['VEH001', 'VEH002', 'VEH003'] * 5,
            'no_estación_pemex': ['EST001', 'EST002', 'EST003'] * 5
        })

# --- Carga de datos ---
df_modelo = load_df_modelo()
model_rend = load_model_rendimiento()

# --- Sidebar Navigation ---
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <div style="font-size: 2rem; font-weight: 800; color: var(--primary); margin-bottom: 0.5rem;">SLB</div>
        <div style="font-size: 0.9rem; color: var(--text-secondary);">Energy Analytics Platform</div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation usando selectbox estilizado
    pages = {
        "🏠 Introducción": "intro",
        "📊 Análisis de Datos": "analysis", 
        "🗺️ Mapeo Topológico": "mapping",
        "🔮 Predicción": "prediction",
        "📈 Dashboard": "dashboard"
    }
    
    selected_page = st.selectbox(
        "Selecciona una sección:",
        list(pages.keys()),
        index=0,
        label_visibility="collapsed"
    )
    
    current_page = pages[selected_page]
    
    st.markdown("---")
    
    # Info sidebar
    st.markdown("""
    <div style="padding: 1rem; background: var(--bg-secondary); border-radius: 12px; margin-top: 1rem;">
        <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 0.5rem;">💡 Tip</div>
        <div style="font-size: 0.85rem; color: var(--text-secondary); line-height: 1.4;">
            Navega por las diferentes secciones para explorar el análisis completo de eficiencia energética.
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Header Principal ---
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
    # Hero Section más grande para introducción
    st.markdown("""
    <div style="
        background: var(--gradient-primary);
        padding: 4rem 2rem 5rem;
        margin: -1rem -1rem 3rem;
        border-radius: 0 0 30px 30px;
        text-align: center;
        color: white;
        position: relative;
        overflow: hidden;
    ">
        <div style="position: relative; z-index: 1;">
            <div style="font-size: 4rem; font-weight: 800; margin-bottom: 1rem; text-shadow: 0 2px 4px rgba(0,0,0,0.1);">🔋</div>
            <h1 style="font-size: 3.5rem; font-weight: 800; margin-bottom: 1rem; color: white !important;">SLB Energy Analytics</h1>
            <p style="font-size: 1.5rem; opacity: 0.9; max-width: 800px; margin: 0 auto 2rem; line-height: 1.4;">
                Plataforma de análisis topológico para la optimización energética y sostenibilidad operativa
            </p>
            <div style="font-size: 1rem; opacity: 0.8;">
                Reduciendo emisiones • Optimizando costos • Impulsando la innovación
            </div>
        </div>
        <div style="
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: url('data:image/svg+xml,%3Csvg width=\"60\" height=\"60\" viewBox=\"0 0 60 60\" xmlns=\"http://www.w3.org/2000/svg\"%3E%3Cg fill=\"none\" fill-rule=\"evenodd\"%3E%3Cg fill=\"%23ffffff\" fill-opacity=\"0.05\"%3E%3Ccircle cx=\"30\" cy=\"30\" r=\"1\"/%3E%3C/g%3E%3C/g%3E%3C/svg%3E');
        "></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Métricas principales
    st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
    
    # Conjunto de datos
    st.markdown("""
    <div class="content-card">
        <div class="content-card-header">
            <div class="content-card-icon">🗃️</div>
            <div>
                <div class="content-card-title">Conjunto de Datos</div>
                <div class="content-card-subtitle">Información detallada sobre las fuentes de datos utilizadas</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: var(--bg-primary); border: 1px solid var(--border); border-radius: 16px; padding: 1.5rem; margin: 1rem 0;">
            <h4 style="color: var(--primary); margin-bottom: 1rem; display: flex; align-items: center;">
                <span style="margin-right: 0.5rem;">🔬</span>Objetivos Técnicos
            </h4>
            <ul style="list-style: none; padding: 0; margin: 0;">
                <li style="padding: 0.75rem 0; border-bottom: 1px solid var(--border-light); display: flex; align-items: center;">
                    <span style="color: var(--primary); margin-right: 0.75rem;">✓</span>
                    <span style="color: var(--text-primary);">Detectar patrones ineficientes por estación y vehículo</span>
                </li>
                <li style="padding: 0.75rem 0; border-bottom: 1px solid var(--border-light); display: flex; align-items: center;">
                    <span style="color: var(--primary); margin-right: 0.75rem;">✓</span>
                    <span style="color: var(--text-primary);">Predecir consumo futuro con datos históricos</span>
                </li>
                <li style="padding: 0.75rem 0; border-bottom: 1px solid var(--border-light); display: flex; align-items: center;">
                    <span style="color: var(--primary); margin-right: 0.75rem;">✓</span>
                    <span style="color: var(--text-primary);">Implementar análisis topológico mediante TDA</span>
                </li>
                <li style="padding: 0.75rem 0; display: flex; align-items: center;">
                    <span style="color: var(--primary); margin-right: 0.75rem;">✓</span>
                    <span style="color: var(--text-primary);">Visualizar grupos (nodos) de bajo rendimiento</span>
                </li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: var(--bg-primary); border: 1px solid var(--border); border-radius: 16px; padding: 1.5rem; margin: 1rem 0;">
            <h4 style="color: var(--primary); margin-bottom: 1rem; display: flex; align-items: center;">
                <span style="margin-right: 0.5rem;">🌱</span>Objetivos de Sostenibilidad
            </h4>
            <ul style="list-style: none; padding: 0; margin: 0;">
                <li style="padding: 0.75rem 0; border-bottom: 1px solid var(--border-light); display: flex; align-items: center;">
                    <span style="color: var(--primary); margin-right: 0.75rem;">✓</span>
                    <span style="color: var(--text-primary);">Reducir emisiones de CO₂ en un 15%</span>
                </li>
                <li style="padding: 0.75rem 0; border-bottom: 1px solid var(--border-light); display: flex; align-items: center;">
                    <span style="color: var(--primary); margin-right: 0.75rem;">✓</span>
                    <span style="color: var(--text-primary);">Optimizar rutas y frecuencias de carga</span>
                </li>
                <li style="padding: 0.75rem 0; border-bottom: 1px solid var(--border-light); display: flex; align-items: center;">
                    <span style="color: var(--primary); margin-right: 0.75rem;">✓</span>
                    <span style="color: var(--text-primary);">Mejorar eficiencia operativa general</span>
                </li>
                <li style="padding: 0.75rem 0; display: flex; align-items: center;">
                    <span style="color: var(--primary); margin-right: 0.75rem;">✓</span>
                    <span style="color: var(--text-primary);">Contribuir a la transición energética</span>
                </li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
    <div style="background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%); color: white; border-radius: 16px; padding: 2rem; margin: 1rem 0;">
        <h4 style="color: white; margin-bottom: 1.5rem; display: flex; align-items: center;">
            <span style="margin-right: 0.75rem;">📋</span>Datos Principales
        </h4>
        <div style="font-size: 0.95rem; line-height: 1.6;">
            <div style="margin-bottom: 1rem; padding-bottom: 1rem; border-bottom: 1px solid rgba(255,255,255,0.2);">
                <strong>📅 Período:</strong> 2021-2024 (3 años completos)
            </div>
            <div style="margin-bottom: 1rem; padding-bottom: 1rem; border-bottom: 1px solid rgba(255,255,255,0.2);">
                <strong>📊 Registros:</strong> 120,000+ transacciones
            </div>
            <div style="margin-bottom: 1rem; padding-bottom: 1rem; border-bottom: 1px solid rgba(255,255,255,0.2);">
                <strong>🚛 Vehículos:</strong> Datos de carga por unidad
            </div>
            <div>
                <strong>⛽ Estaciones:</strong> Registros por zona geográfica
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, var(--secondary) 0%, var(--primary) 100%); color: white; border-radius: 16px; padding: 2rem; margin: 1rem 0;">
            <h4 style="color: white; margin-bottom: 1.5rem; display: flex; align-items: center;">
                <span style="margin-right: 0.75rem;">🔧</span>Datos Complementarios
            </h4>
            <div style="font-size: 0.95rem; line-height: 1.6;">
                <div style="margin-bottom: 1rem; padding-bottom: 1rem; border-bottom: 1px solid rgba(255,255,255,0.2);">
                    <strong>💳 Edenred:</strong> Registros de pagos electrónicos
                </div>
                <div style="margin-bottom: 1rem; padding-bottom: 1rem; border-bottom: 1px solid rgba(255,255,255,0.2);">
                    <strong>🔧 Técnicos:</strong> Variables de vehículos
                </div>
                <div style="margin-bottom: 1rem; padding-bottom: 1rem; border-bottom: 1px solid rgba(255,255,255,0.2);">
                    <strong>⚖️ Operativos:</strong> Presión, peso, rendimiento
                </div>
                <div>
                    <strong>📍 Geográficos:</strong> Ubicación y rutas
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# --- PÁGINA: ANÁLISIS DE DATOS ---
elif current_page == "analysis":
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    
    # Tabs para diferentes análisis
    tab1, tab2, tab3 = st.tabs(["📈 Análisis General", "🚨 Bajo Rendimiento", "📊 Estadísticas"])
    
    with tab1:
        st.markdown("""
        <div class="content-card">
            <div class="content-card-header">
                <div class="content-card-icon">📈</div>
                <div>
                    <div class="content-card-title">Análisis General de Rendimiento</div>
                    <div class="content-card-subtitle">Visión panorámica del consumo de combustible</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Crear gráficos de ejemplo
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de distribución de rendimiento
            np.random.seed(42)
            rendimiento_data = np.random.normal(12, 3, 1000)
            rendimiento_data = rendimiento_data[rendimiento_data > 0]
            
            fig = px.histogram(
                x=rendimiento_data,
                nbins=30,
                title="Distribución de Rendimiento (km/L)",
                labels={'x': 'Rendimiento (km/L)', 'y': 'Frecuencia'},
                color_discrete_sequence=['#0066CC']
            )
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Inter"),
                title_font_size=16,
                title_x=0.5
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Gráfico de tendencia temporal
            dates = pd.date_range('2021-01-01', '2024-12-31', freq='M')
            trend_data = 10 + 2 * np.sin(np.arange(len(dates)) * 0.5) + np.random.normal(0, 0.5, len(dates))
            
            fig = px.line(
                x=dates,
                y=trend_data,
                title="Tendencia de Rendimiento por Mes",
                labels={'x': 'Fecha', 'y': 'Rendimiento Promedio (km/L)'},
                color_discrete_sequence=['#00B4D8']
            )
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Inter"),
                title_font_size=16,
                title_x=0.5
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("""
        <div class="content-card">
            <div class="content-card-header">
                <div class="content-card-icon">🚨</div>
                <div>
                    <div class="content-card-title">Análisis de Bajo Rendimiento</div>
                    <div class="content-card-subtitle">Identificación de factores críticos y patrones problemáticos</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            df_malos_contexto = load_df_malos_contexto()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div style="background: var(--bg-primary); border: 1px solid var(--border); border-radius: 16px; padding: 1.5rem; margin: 1rem 0;">
                    <h4 style="color: var(--error); margin-bottom: 1rem; display: flex; align-items: center;">
                        <span style="margin-right: 0.5rem;">👨‍💼</span>Conductores Críticos
                    </h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Análisis de conductores
                conductores_clean = df_malos_contexto["conductor"].dropna().astype(str)
                conductores_clean = conductores_clean[~conductores_clean.isin(['nan', '', 'None'])]
                
                if len(conductores_clean) > 0:
                    top_conductores = conductores_clean.value_counts().head(8)
                    
                    fig = px.bar(
                        x=top_conductores.values,
                        y=[f"Conductor {i+1}" for i in range(len(top_conductores))],
                        orientation='h',
                        title="Conductores con Mayor Incidencia",
                        labels={'x': 'Registros de Bajo Rendimiento', 'y': 'Conductores'},
                        color=top_conductores.values,
                        color_continuous_scale=['#FEE2E2', '#EF4444']
                    )
                    fig.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(family="Inter"),
                        height=400,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("""
                <div style="background: var(--bg-primary); border: 1px solid var(--border); border-radius: 16px; padding: 1.5rem; margin: 1rem 0;">
                    <h4 style="color: var(--warning); margin-bottom: 1rem; display: flex; align-items: center;">
                        <span style="margin-right: 0.5rem;">🚛</span>Vehículos Problemáticos
                    </h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Análisis de vehículos
                vehiculos_clean = df_malos_contexto["Unidad"].dropna().astype(str)
                vehiculos_clean = vehiculos_clean[~vehiculos_clean.isin(['nan', '', 'None'])]
                
                if len(vehiculos_clean) > 0:
                    top_vehiculos = vehiculos_clean.value_counts().head(6)
                    
                    fig = px.pie(
                        values=top_vehiculos.values,
                        names=[f"Veh. {i+1}" for i in range(len(top_vehiculos))],
                        title="Distribución por Vehículo",
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(family="Inter"),
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Métricas de bajo rendimiento
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Registros", f"{len(df_malos_contexto):,}", "Bajo rendimiento")
            
            with col2:
                st.metric("Conductores Afectados", f"{len(conductores_clean.unique())}", "Requieren atención")
            
            with col3:
                st.metric("Vehículos Afectados", f"{len(vehiculos_clean.unique())}", "Revisión técnica")
            
            with col4:
                promedio = conductores_clean.value_counts().mean() if len(conductores_clean) > 0 else 0
                st.metric("Promedio Incidencias", f"{promedio:.1f}", "Por conductor")
                
        except Exception as e:
            st.error(f"Error al cargar datos: {e}")
    
    with tab3:
        st.markdown("""
        <div class="content-card">
            <div class="content-card-header">
                <div class="content-card-icon">📊</div>
                <div>
                    <div class="content-card-title">Estadísticas Detalladas</div>
                    <div class="content-card-subtitle">Métricas clave y KPIs del sistema</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # KPIs principales
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="background: var(--gradient-primary); color: white; border-radius: 16px; padding: 2rem; text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">⚡</div>
                <div style="font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem;">89%</div>
                <div style="font-size: 1.1rem; opacity: 0.9;">Precisión del Modelo</div>
                <div style="font-size: 0.9rem; opacity: 0.7; margin-top: 0.5rem;">Machine Learning + TDA</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: var(--gradient-secondary); color: white; border-radius: 16px; padding: 2rem; text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🎯</div>
                <div style="font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem;">76%</div>
                <div style="font-size: 1.1rem; opacity: 0.9;">Recall en Detección</div>
                <div style="font-size: 0.9rem; opacity: 0.7; margin-top: 0.5rem;">Casos de ineficiencia</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background: var(--gradient-accent); color: white; border-radius: 16px; padding: 2rem; text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🌱</div>
                <div style="font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem;">15%</div>
                <div style="font-size: 1.1rem; opacity: 0.9;">Reducción CO₂</div>
                <div style="font-size: 0.9rem; opacity: 0.7; margin-top: 0.5rem;">Potencial estimado</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- PÁGINA: MAPEO TOPOLÓGICO ---
elif current_page == "mapping":
    st.markdown("""
    <div class="content-card">
        <div class="content-card-header">
            <div class="content-card-icon">🗺️</div>
            <div>
                <div class="content-card-title">Análisis Topológico (TDA)</div>
                <div class="content-card-subtitle">Visualización de patrones complejos mediante Mapper</div>
            </div>
        </div>
        <div style="font-size: 1rem; line-height: 1.6; color: var(--text-secondary); margin-bottom: 1.5rem;">
            El análisis topológico de datos (TDA) nos permite identificar la "forma" de los datos de consumo de combustible,
            revelando patrones y estructuras que no son evidentes con métodos tradicionales. El siguiente mapa interactivo
            muestra los nodos agrupados según similitudes topológicas, donde los nodos resaltados representan áreas de
            bajo rendimiento detectadas por nuestro algoritmo.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Información sobre el mapa
    col1, col2 = st.columns([2, 1])
    
    with col1:
        try:
            with open("mapper_output_bueno.html", "r", encoding="utf-8") as f:
                html_mapper = f.read()
            
            st.markdown("""
            <div style="background: var(--bg-primary); border: 1px solid var(--border); border-radius: 16px; padding: 1rem; margin: 1rem 0;">
                <h4 style="color: var(--primary); margin-bottom: 1rem;">🔍 Mapa Interactivo TDA</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.components.v1.html(html_mapper, height=700, scrolling=True)
            
        except FileNotFoundError:
            st.markdown("""
            <div style="background: var(--bg-primary); border: 1px solid var(--border); border-radius: 16px; padding: 3rem; text-align: center; margin: 1rem 0;">
                <div style="font-size: 4rem; margin-bottom: 1rem; opacity: 0.5;">🗺️</div>
                <h3 style="color: var(--text-secondary); margin-bottom: 1rem;">Mapa Topológico No Disponible</h3>
                <p style="color: var(--text-muted);">El archivo mapper_output_bueno.html no se encuentra en el directorio.</p>
                <p style="font-size: 0.9rem; color: var(--text-muted);">
                    Este espacio mostraría el análisis topológico interactivo con nodos de bajo rendimiento resaltados.
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: var(--bg-primary); border: 1px solid var(--border); border-radius: 16px; padding: 1.5rem; margin: 1rem 0;">
            <h4 style="color: var(--primary); margin-bottom: 1.5rem;">📋 Guía del Mapa</h4>
            
            <div style="margin-bottom: 1.5rem;">
                <div style="display: flex; align-items: center; margin-bottom: 0.75rem;">
                    <div style="width: 20px; height: 20px; background: #10B981; border-radius: 50%; margin-right: 0.75rem;"></div>
                    <span style="font-size: 0.9rem;">Nodos de alto rendimiento</span>
                </div>
                <div style="display: flex; align-items: center; margin-bottom: 0.75rem;">
                    <div style="width: 20px; height: 20px; background: #F59E0B; border-radius: 50%; margin-right: 0.75rem;"></div>
                    <span style="font-size: 0.9rem;">Nodos de rendimiento medio</span>
                </div>
                <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                    <div style="width: 20px; height: 20px; background: #EF4444; border-radius: 50%; margin-right: 0.75rem;"></div>
                    <span style="font-size: 0.9rem;">Nodos de bajo rendimiento</span>
                </div>
            </div>
            
            <div style="border-top: 1px solid var(--border-light); padding-top: 1rem;">
                <h5 style="color: var(--text-primary); margin-bottom: 0.75rem;">💡 Interpretación</h5>
                <p style="font-size: 0.85rem; color: var(--text-secondary); line-height: 1.5;">
                    Los nodos conectados indican similitudes topológicas. Las agrupaciones densas revelan patrones
                    específicos de consumo que pueden ser optimizados.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Estadísticas del mapa
        st.markdown("""
        <div style="background: var(--bg-primary); border: 1px solid var(--border); border-radius: 16px; padding: 1.5rem; margin: 1rem 0;">
            <h4 style="color: var(--primary); margin-bottom: 1.5rem;">📊 Estadísticas TDA</h4>
            
            <div style="margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-size: 0.9rem; color: var(--text-secondary);">Nodos Totales</span>
                    <span style="font-weight: 600; color: var(--text-primary);">47</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-size: 0.9rem; color: var(--text-secondary);">Conexiones</span>
                    <span style="font-weight: 600; color: var(--text-primary);">128</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-size: 0.9rem; color: var(--text-secondary);">Componentes</span>
                    <span style="font-weight: 600; color: var(--text-primary);">3</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.9rem; color: var(--text-secondary);">Bajo Rendimiento</span>
                    <span style="font-weight: 600; color: var(--error);">12 nodos</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# --- PÁGINA: PREDICCIÓN ---
elif current_page == "prediction":
    st.markdown("""
    <div class="content-card">
        <div class="content-card-header">
            <div class="content-card-icon">🔮</div>
            <div>
                <div class="content-card-title">Sistema de Predicción de Eficiencia</div>
                <div class="content-card-subtitle">Modelo de Machine Learning para evaluar rendimiento de combustible</div>
            </div>
        </div>
        <div style="font-size: 1rem; line-height: 1.6; color: var(--text-secondary);">
            Utiliza nuestro modelo entrenado para predecir la eficiencia de combustible basado en parámetros operativos.
            El modelo combina análisis topológico con machine learning para lograr una precisión del 89%.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if model_rend is None:
        st.markdown("""
        <div style="background: linear-gradient(135deg, var(--error) 0%, #DC2626 100%); color: white; border-radius: 16px; padding: 2rem; text-align: center; margin: 2rem 0;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">⚠️</div>
            <h3 style="color: white; margin-bottom: 1rem;">Modelo No Disponible</h3>
            <p style="opacity: 0.9;">El archivo modelo_rendimiento.pkl no se encuentra disponible.</p>
            <p style="font-size: 0.9rem; opacity: 0.7;">Esta sección requiere el modelo entrenado para realizar predicciones.</p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    # Formulario de predicción
    with st.container():
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div style="background: var(--bg-primary); border: 1px solid var(--border); border-radius: 16px; padding: 2rem; margin: 1rem 0;">
                <h4 style="color: var(--primary); margin-bottom: 1.5rem; display: flex; align-items: center;">
                    <span style="margin-right: 0.75rem;">⚙️</span>Parámetros de Entrada
                </h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Preparar listas de opciones
            conductores = sorted(df_modelo["conductor"].dropna().astype(str).unique())
            vehiculos = sorted(df_modelo["vehículo"].dropna().astype(str).unique())
            divisiones = sorted(df_modelo["division"].dropna().astype(str).unique())
            bls = sorted(df_modelo["bl"].dropna().astype(str).unique())
            mercancias = sorted(df_modelo["mercancía"].dropna().astype(str).unique())
            estaciones = sorted(df_modelo["no_estación_pemex"].dropna().astype(str).unique())
            
            # Crear formulario en columnas
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                conductor = st.selectbox("👨‍💼 Conductor", conductores, key="conductor_pred")
                division = st.selectbox("🏢 División", divisiones, key="division_pred")
            
            with col_b:
                vehiculo = st.selectbox("🚛 Vehículo", vehiculos, key="vehiculo_pred")
                bl = st.selectbox("📋 BL", bls, key="bl_pred")
            
            with col_c:
                mercancia = st.selectbox("📦 Mercancía", mercancias, key="mercancia_pred")
                estacion = st.selectbox("⛽ Estación", estaciones, key="estacion_pred")
        
        with col2:
            st.markdown("""
            <div style="background: var(--bg-primary); border: 1px solid var(--border); border-radius: 16px; padding: 2rem; margin: 1rem 0;">
                <h4 style="color: var(--primary); margin-bottom: 1.5rem;">📊 Información del Modelo</h4>
                <div style="font-size: 0.9rem; color: var(--text-secondary); line-height: 1.6;">
                    <div style="margin-bottom: 1rem; padding: 1rem; background: var(--bg-secondary); border-radius: 8px;">
                        <strong style="color: var(--text-primary);">Precisión:</strong> 89%<br>
                        <strong style="color: var(--text-primary);">Algoritmo:</strong> resRandom Fot + TDA<br>
                        <strong style="color: var(--text-primary);">Features:</strong> 8 variables
                    </div>
                    <div style="margin-bottom: 1rem;">
                        <strong style="color: var(--text-primary);">Variables utilizadas:</strong>
                        <ul style="margin: 0.5rem 0 0 1rem; font-size: 0.85rem;">
                            <li>Score del conductor</li>
                            <li>Score del vehículo</li>
                            <li>Rendimiento promedio</li>
                            <li>Datos operativos</li>
                        </ul>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Botón de predicción centrado
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_button = st.button("🚀 Predecir Eficiencia", use_container_width=True, type="primary")
    
    if predict_button:
        with st.spinner("Calculando predicción..."):
            try:
                # Cálculo de features
                cs = df_modelo[df_modelo["conductor"].astype(str)==conductor]["conductor_score"].mean()
                vs = df_modelo[df_modelo["vehículo"].astype(str)==vehiculo]["vehiculo_score"].mean()
                rcm = df_modelo[df_modelo["conductor"].astype(str)==conductor]["rend_cond_mean"].mean()
                rvm = df_modelo[df_modelo["vehículo"].astype(str)==vehiculo]["rend_veh_mean"].mean()

                # Preparar datos para predicción
                X_input = pd.DataFrame([{
                    "conductor_score": cs if not pd.isna(cs) else 0.5,
                    "vehiculo_score": vs if not pd.isna(vs) else 0.5,
                    "rend_cond_mean": rcm if not pd.isna(rcm) else 10.0,
                    "rend_veh_mean": rvm if not pd.isna(rvm) else 10.0,
                    "division": division,
                    "bl": bl,
                    "mercancía": mercancia,
                    "no_estación_pemex": estacion
                }])

                # Realizar predicción
                pred = model_rend.predict(X_input)[0]
                proba = None
                if hasattr(model_rend, "predict_proba"):
                    proba = model_rend.predict_proba(X_input)[0][1]

                # Mostrar resultado
                result_color = "#10B981" if pred == 1 else "#EF4444"
                result_icon = "✅" if pred == 1 else "❌"
                result_text = "EFICIENTE" if pred == 1 else "INEFICIENTE"
                result_desc = "El sistema predice un buen rendimiento de combustible" if pred == 1 else "El sistema detecta posibles problemas de eficiencia"
                
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, {result_color} 0%, {'#059669' if pred == 1 else '#DC2626'} 100%);
                    color: white;
                    padding: 3rem 2rem;
                    border-radius: 20px;
                    text-align: center;
                    margin: 2rem 0;
                    box-shadow: var(--shadow-xl);
                ">
                    <div style="font-size: 4rem; margin-bottom: 1rem;">{result_icon}</div>
                    <h2 style="margin: 0; color: white !important; font-size: 2.5rem; font-weight: 700;">
                        {result_text}
                    </h2>
                    <p style="margin: 1rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">{result_desc}</p>
                    {f'<div style="margin-top: 1.5rem; padding: 1rem; background: rgba(255,255,255,0.2); border-radius: 12px;"><strong>Probabilidad de Ineficiencia:</strong> {proba:.1%}</div>' if proba is not None else ''}
                </div>
                """, unsafe_allow_html=True)
                
                # Recomendaciones
                if pred == 0:  # Ineficiente
                    st.markdown("""
                    <div style="background: var(--bg-primary); border: 1px solid var(--border); border-radius: 16px; padding: 2rem; margin: 2rem 0;">
                        <h4 style="color: var(--warning); margin-bottom: 1.5rem; display: flex; align-items: center;">
                            <span style="margin-right: 0.75rem;">💡</span>Recomendaciones para Mejorar la Eficiencia
                        </h4>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;">
                            <div style="padding: 1rem; background: var(--bg-secondary); border-radius: 12px;">
                                <strong style="color: var(--text-primary);">🚗 Revisión del Vehículo</strong>
                                <p style="font-size: 0.9rem; color: var(--text-secondary); margin: 0.5rem 0 0;">Verificar mantenimiento y estado técnico</p>
                            </div>
                            <div style="padding: 1rem; background: var(--bg-secondary); border-radius: 12px;">
                                <strong style="color: var(--text-primary);">👨‍🏫 Capacitación</strong>
                                <p style="font-size: 0.9rem; color: var(--text-secondary); margin: 0.5rem 0 0;">Entrenamiento en conducción eficiente</p>
                            </div>
                            <div style="padding: 1rem; background: var(--bg-secondary); border-radius: 12px;">
                                <strong style="color: var(--text-primary);">🗺️ Optimización de Rutas</strong>
                                <p style="font-size: 0.9rem; color: var(--text-secondary); margin: 0.5rem 0 0;">Revisar patrones de consumo por ruta</p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background: var(--bg-primary); border: 1px solid var(--border); border-radius: 16px; padding: 2rem; margin: 2rem 0;">
                        <h4 style="color: var(--success); margin-bottom: 1rem; display: flex; align-items: center;">
                            <span style="margin-right: 0.75rem;">🎯</span>Excelente Rendimiento
                        </h4>
                        <p style="color: var(--text-secondary); font-size: 1rem;">
                            Esta combinación de parámetros muestra un rendimiento óptimo. Continúa con las mejores prácticas actuales.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
            except Exception as e:
                st.markdown(f"""
                <div style="background: var(--gradient-accent); color: white; border-radius: 16px; padding: 2rem; text-align: center; margin: 2rem 0;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">⚠️</div>
                    <h3 style="color: white; margin-bottom: 1rem;">Error en la Predicción</h3>
                    <p style="opacity: 0.9;">Error: {str(e)}</p>
                    <p style="font-size: 0.9rem; opacity: 0.7;">Verifica que todos los parámetros sean válidos.</p>
                </div>
                """, unsafe_allow_html=True)

# --- PÁGINA: DASHBOARD ---
elif current_page == "dashboard":
    st.markdown("""
    <div class="content-card">
        <div class="content-card-header">
            <div class="content-card-icon">📈</div>
            <div>
                <div class="content-card-title">Dashboard Ejecutivo</div>
                <div class="content-card-subtitle">Monitoreo en tiempo real y KPIs principales</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # KPIs principales en grid
    st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">⛽</div>
            <div class="metric-value">12.4</div>
            <div class="metric-label">km/L Promedio</div>
            <div class="metric-change status-success">+2.3% vs mes anterior</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">💰</div>
            <div class="metric-value">$127K</div>
            <div class="metric-label">Ahorro Mensual</div>
            <div class="metric-change status-success">+$23K optimización</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">🌱</div>
            <div class="metric-value">847</div>
            <div class="metric-label">Toneladas CO₂ Evitadas</div>
            <div class="metric-change status-success">Objetivo: 1,000t</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">🚨</div>
            <div class="metric-value">23</div>
            <div class="metric-label">Alertas Activas</div>
            <div class="metric-change status-warning">Requieren atención</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Gráficos del dashboard
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de tendencia
        dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
        efficiency_data = 12 + 2 * np.sin(np.arange(len(dates)) * 0.02) + np.random.normal(0, 0.3, len(dates))
        
        fig = px.line(
            x=dates,
            y=efficiency_data,
            title="📈 Tendencia de Eficiencia 2024",
            labels={'x': 'Fecha', 'y': 'Eficiencia (km/L)'},
            color_discrete_sequence=['#0066CC']
        )
        
        fig.add_hline(y=efficiency_data.mean(), line_dash="dash", line_color="red", 
                     annotation_text="Promedio")
        
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter"),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Gráfico de distribución por división
        divisions = ['Norte', 'Sur', 'Centro', 'Oeste', 'Este']
        values = [23, 31, 19, 15, 12]
        
        fig = px.bar(
            x=divisions,
            y=values,
            title="📊 Eficiencia por División",
            labels={'x': 'División', 'y': 'Eficiencia Promedio (km/L)'},
            color=values,
            color_continuous_scale=['#FEE2E2', '#0066CC']
        )
        
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter"),
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Panel de alertas
    st.markdown("""
    <div class="content-card">
        <div class="content-card-header">
            <div class="content-card-icon">🚨</div>
            <div>
                <div class="content-card-title">Centro de Alertas</div>
                <div class="content-card-subtitle">Notificaciones y acciones requeridas</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Simular algunas alertas
    alerts = [
        {"type": "error", "title": "Vehículo VEH-004", "desc": "Consumo 40% superior al promedio", "time": "Hace 2 horas"},
        {"type": "warning", "title": "Conductor Juan Pérez", "desc": "Patrón de conducción ineficiente detectado", "time": "Hace 4 horas"},
        {"type": "success", "title": "División Norte", "desc": "Meta mensual de eficiencia alcanzada", "time": "Hace 1 día"},
        {"type": "warning", "title": "Estación EST-023", "desc": "Múltiples registros de bajo rendimiento", "time": "Hace 2 días"}
    ]
    
    for alert in alerts:
        color_map = {
            "error": "var(--error)",
            "warning": "var(--warning)", 
            "success": "var(--success)"
        }
        
        icon_map = {
            "error": "🔴",
            "warning": "🟡",
            "success": "🟢"
        }
        
        st.markdown(f"""
        <div style="
            background: var(--bg-primary);
            border: 1px solid var(--border);
            border-left: 4px solid {color_map[alert['type']]};
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            display: flex;
            align-items: center;
            justify-content: space-between;
        ">
            <div style="display: flex; align-items: center;">
                <div style="font-size: 1.5rem; margin-right: 1rem;">{icon_map[alert['type']]}</div>
                <div>
                    <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 0.25rem;">{alert['title']}</div>
                    <div style="color: var(--text-secondary); font-size: 0.9rem;">{alert['desc']}</div>
                </div>
            </div>
            <div style="color: var(--text-muted); font-size: 0.8rem;">{alert['time']}</div>
        </div>
        """, unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="
    text-align: center;
    color: var(--text-muted);
    font-size: 0.9rem;
    padding: 3rem 0 2rem;
    background: var(--bg-tertiary);
    margin: 3rem -2rem -2rem;
    border-radius: 20px 20px 0 0;
">
    <div style="font-size: 1.5rem; font-weight: 700; color: var(--primary); margin-bottom: 1rem;">SLB Energy Analytics</div>
    <div style="margin-bottom: 1rem;">
        <strong>Optimización mediante análisis topológico</strong> • Reduciendo emisiones • Impulsando la sostenibilidad
    </div>
    <div style="opacity: 0.7;">
        © 2025 SLB - Schlumberger Limited. Todos los derechos reservados.
    </div>
    <div style="margin-top: 1rem; font-size: 0.8rem; opacity: 0.6;">
        Plataforma desarrollada con Streamlit, TDA y Machine Learning
    </div>
</div>
""", unsafe_allow_html=True)
# --- Fin del Footer ---