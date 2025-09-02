import streamlit as st
from modules.ui import render_web_login

render_web_login(next_page="main.py")

# opcional: enlace al login móvil
st.page_link("pages/00_Login.py", label="Usar verificación móvil (Open Gateway)", icon="📱")

# --- DEBUG TEMPORAL: quítalo luego ---
from modules.ui import load_users, _make_hash
with st.expander("Debug temporal (quitar luego)", expanded=False):
    st.write(load_users().head())  # debe mostrar la fila 'admin'
    st.write("Hash(Admin123!) con salt 3f9a7c1b77a2c1d0:",
             _make_hash("Admin123!", "3f9a7c1b77a2c1d0"))

