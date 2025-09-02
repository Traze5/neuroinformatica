# pages/00_Auto_Auth.py
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Autenticación", page_icon="🔐", layout="centered")

def _get_qp():
    # Compatibilidad 1.49+ y previas
    try:
        return dict(st.query_params)
    except Exception:
        return st.experimental_get_query_params()

qp = _get_qp()
ua = None
if "ua" in qp:
    # en 1.49 st.query_params devuelve str; en previas es lista
    val = qp["ua"]
    ua = val if isinstance(val, str) else (val[0] if val else None)

if not ua:
    # Inyecta JS para detectar user-agent y recargar con ?ua=m|w
    components.html("""
    <script>
      (function() {
        const isMobile = /Mobi|Android|iPhone|iPad|iPod|Opera Mini|IEMobile|Mobile/i.test(navigator.userAgent);
        const url = new URL(window.parent.location);
        if (!url.searchParams.get('ua')) {
          url.searchParams.set('ua', isMobile ? 'm' : 'w');
          window.parent.history.replaceState({}, '', url);
          window.parent.location.reload();
        }
      })();
    </script>
    """, height=0)
    st.write("Detectando dispositivo…")
    st.stop()

# Con 'ua' ya definido, redirige
if ua == "m":
    # Móvil → flujo NV
    st.switch_page("pages/00_Login.py")
else:
    # Web → login usuario/contraseña
    st.switch_page("pages/00_Web_Login.py")
