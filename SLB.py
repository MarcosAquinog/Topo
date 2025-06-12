import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# --- Configuración de la página con branding SLB ---
st.set_page_config(
    page_title="Optimización de Combustible SLB",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.slb.com/sustainability',
        'About': (
            "### Análisis de Consumo de Combustible para SLB\n"
            "*Visualización, predicción y optimización energética mediante análisis topológico*"
        )
    }
)

# --- Tus estilos CSS (pega aquí todo tu <style>…</style>) ---
st.markdown("""
<style>
  /* ...tu CSS existente... */
</style>
""", unsafe_allow_html=True)

# --- Carga del histórico para poblar selectboxes ---
@st.cache_data
def load_df_modelo():
    return pd.read_csv("df_modelo.csv")

df_modelo = load_df_modelo()

# --- Carga del pipeline de clasificación entrenado ---
@st.cache_resource
def load_model_rendimiento():
    try:
        return joblib.load("modelo_rendimiento.pkl")
    except Exception as e:
        st.error(f"Error al cargar el modelo de rendimiento: {e}")
        return None

model_rend = load_model_rendimiento()
if model_rend is None:
    st.stop()

# --- Sidebar y navegación ---
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0;">
      <h3 style="color:#003f5c;">SLB</h3>
      <p>Optimización del Consumo de Combustible</p>
    </div>
    """, unsafe_allow_html=True)

    page = st.selectbox(
        "Navegar",
        ["Inicio", "Análisis", "Modelo de Predicción", "Predicción Rendimiento"],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### Acerca de")
    st.markdown(
        "Esta herramienta permite a SLB identificar patrones de consumo ineficiente "
        "y mejorar la eficiencia energética de su flota mediante análisis topológico y modelos predictivos."
    )
    st.markdown("---")
    st.markdown(
        "<div style='font-size:.8rem;opacity:.6;'>SLB Energy Optimization © 2025</div>",
        unsafe_allow_html=True
    )

# --- Página: Inicio ---
if page == "Inicio":
    st.title("Optimización del Consumo de Combustible en SLB")
    st.write("Bienvenido a la aplicación de SLB para análisis y optimización de combustible.")

# --- Página: Análisis ---
elif page == "Análisis":
    st.title("📊 Análisis de Rendimiento de Combustible")
    st.write("Aquí van tus visualizaciones de Mapper y análisis de bajo rendimiento.")

# --- Página: Modelo de Predicción (ventas) ---
elif page == "Modelo de Predicción":
    st.title("📊 Sistema de Predicción de Ventas")
    # ... tu código de carga y predicción de ventas ...

# --- Página: Predicción Rendimiento ---
elif page == "Predicción Rendimiento":
    st.header("🔍 Predicción de Eficiencia de Combustible")

    # Prepara listas homogéneas de strings para cada selectbox
    conductores = sorted(df_modelo["conductor"].dropna().astype(str).unique())
    vehiculos   = sorted(df_modelo["vehículo"].dropna().astype(str).unique())
    divisiones  = sorted(df_modelo["division"].dropna().astype(str).unique())
    bls         = sorted(df_modelo["bl"].dropna().astype(str).unique())
    mercancias  = sorted(df_modelo["mercancía"].dropna().astype(str).unique())
    estaciones  = sorted(df_modelo["no_estación_pemex"].dropna().astype(str).unique())

    # Selectboxes
    conductor = st.selectbox("Conductor", conductores)
    vehiculo  = st.selectbox("Vehículo", vehiculos)
    division  = st.selectbox("División", divisiones)
    bl        = st.selectbox("BL", bls)
    mercancia = st.selectbox("Mercancía", mercancias)
    estacion  = st.selectbox("Estación Pemex", estaciones)

    if st.button("Calcular rendimiento"):
        # 1) Calcula features numéricos desde todo el histórico
        cs  = df_modelo[df_modelo["conductor"].astype(str)==conductor]["conductor_score"].mean()
        vs  = df_modelo[df_modelo["vehículo"].astype(str)==vehiculo]["vehiculo_score"].mean()
        rcm = df_modelo[df_modelo["conductor"].astype(str)==conductor]["rend_cond_mean"].mean()
        rvm = df_modelo[df_modelo["vehículo"].astype(str)==vehiculo]["rend_veh_mean"].mean()

        # 2) Construye el DataFrame de entrada
        X_input = pd.DataFrame([{
            "conductor_score":   cs,
            "vehiculo_score":    vs,
            "rend_cond_mean":    rcm,
            "rend_veh_mean":     rvm,
            "division":          division,
            "bl":                bl,
            "mercancía":         mercancia,
            "no_estación_pemex": estacion
        }])

        # 3) Predicción
        pred  = model_rend.predict(X_input)[0]
        proba = model_rend.predict_proba(X_input)[0][1] if hasattr(model_rend, "predict_proba") else None

        # 4) Mostrar resultado
        st.success("✅ EFICIENTE" if pred == 1 else "❌ INEFICIENTE")
        if proba is not None:
            st.write(f"Probabilidad de eficiencia: {proba:.1%}")
