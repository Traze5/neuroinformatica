import streamlit as st
from modules.ui import render_web_login

render_web_login(next_page="main.py")

# opcional: enlace al login móvil
st.page_link("pages/00_Login.py", label="Usar verificación móvil (Open Gateway)", icon="📱")



