import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Autenticación", page_icon="🔐", layout="centered")

# Si ya hay sesión => directo al main
if st.session_state.get("auth_ok"):
    st.switch_page("main.py")

# Query params (API nueva)
qp = st.query_params

# Atajos opcionales: ?force=nv | ?force=web
force = (qp.get("force") or [""])[0].lower()
if force in ("nv", "web"):
    if force == "nv":
        st.switch_page("pages/00_Login.py")
    else:
        st.switch_page("pages/00_Web_Login.py")
    st.stop()

# Detecta UA con JS y recarga con ?ua=m|w (mobile / web)
ua = (qp.get("ua") or [""])[0]
if not ua:
    components.html(
        """
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
        """,
        height=0,
    )
    st.write("Detectando dispositivo…")
    st.stop()

# Ruta según dispositivo
if ua == "m":
    st.switch_page("pages/00_Login.py")        # NV (móvil)
else:
    st.switch_page("pages/00_Web_Login.py")    # login web
