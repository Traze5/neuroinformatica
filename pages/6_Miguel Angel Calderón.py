import streamlit as st
from modules.ui import require_auth, generar_menu

# --- guardia de login + menú propio ---
require_auth(login_page="pages/00_Auto_Auth.py")
generar_menu(
    usuario=st.session_state.get("usuario"),
    msisdn=st.session_state.get("user_msisdn"),
)

st.set_page_config(page_title="Sobre el autor", layout="wide")
st.header("👤 Miguel Angel Calderón")

st.subheader("Perfil")
st.write("""
Ingeniero de Sistemas y estudiante de Máster en Data Science (UAM). Intereses en
neuroinformática, modelos neuronales computacionales y plataformas cloud reproducibles.
""")

st.subheader("Proyecto TFM (resumen)")
st.write("""
**Título:** Herramienta Cloud con Open Gateway para la Simulación y Visualización de
Actividad Neuronal: Un Enfoque de Ciencia de Datos.

**Stack:** Python + Streamlit; despliegue previsto en Azure App Service; integración FAIR (repositorio
y DOI por release). Modelos incluidos: Hindmarsh–Rose (módulo principal), Rulkov (mapa), Hodgkin–Huxley
y Izhikevich (complementarios). Módulo de Invariantes Dinámicos (IDS) basado en trabajos recientes de la UAM.
**Open Gateway (NV):** demo funcional de autenticación reforzada previa al uso.
""")

st.subheader("Contacto")
c1, c2 = st.columns(2)
with c1:
    st.write("📧  miguelcalderon555@hotmail.com")
with c2:
    st.write("🌐  https://github.com/Traze5")  # ajusta si quieres

