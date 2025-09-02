# modules/ui.py
import pathlib
import pandas as pd
import streamlit as st
import re, inspect
from pathlib import Path
import hmac, hashlib
PAGES_DIR = pathlib.Path("pages")

# -------------------- datos de usuario --------------------
@st.cache_data
def load_users():
    path = pathlib.Path("usuarios.csv")
    if not path.exists():
        # incluye las columnas de credenciales para evitar KeyError
        return pd.DataFrame(columns=["usuario", "nombre", "rol", "msisdn", "pass_hash", "salt"])
    # fuerza dtype=str y normaliza TODAS las columnas (incluye pass_hash/salt)
    df = pd.read_csv(path, dtype=str).fillna("")
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    return df


# -------------------- guardia de login --------------------
def require_auth(login_page: str = "pages/00_Login.py"):
    """Si no hay sesión, envía a la página de login (dentro de /pages)."""
    if not st.session_state.get("auth_ok", False):
        st.switch_page(login_page)

# -------------------- menú lateral "completo" --------------------
def _link_if_exists(file_name: str, label: str, icon: str):
    path = PAGES_DIR / file_name
    if path.exists():
        st.page_link(str(path).replace("\\", "/"), label=label, icon=icon)

def generar_menu(usuario: str | None = None, msisdn: str | None = None):
    """Menú lateral completo (para main.py u otras páginas 'hub')."""
    df = load_users()
    row = None
    if usuario:
        m = df[df["usuario"] == usuario]
        if not m.empty:
            row = m.iloc[0]
    if row is None and msisdn and "msisdn" in df.columns:
        m = df[df["msisdn"] == msisdn]
        if not m.empty:
            row = m.iloc[0]
            st.session_state["usuario"] = row.get("usuario", "")

    nombre = (row.get("nombre") if row is not None else "") or "Usuario"

    with st.sidebar:
        st.write(f"Hola **:blue-background[{nombre}]**")

        st.page_link("main.py", label="Inicio", icon="🏠")

        st.subheader("Modelos")
        _link_if_exists("1_Modelo Hindmarsh–Rose.py", "Modelo Hindmarsh–Rose", "🌊")
        _link_if_exists("3_Modelo Izhikevich.py",    "Modelo Izhikevich",     "🧠")
        _link_if_exists("2_Modelo Rulkov.py",        "Modelo Rulkov",         "🧩")
        _link_if_exists("4_Modelo Hodgkin–Huxley.py","Modelo Hodgkin–Huxley", "⚡")

        st.subheader("Análisis")
        _link_if_exists("5_Invariantes Dinámicos.py","Invariantes", "📊")

        st.subheader("Perfil")
        _link_if_exists("6_Miguel Angel Calderón.py","Miguel Angel Calderón", "👤")

        st.divider()
        if st.button("Salir", use_container_width=True):
            st.session_state.clear()
            st.switch_page("pages/00_Login.py")

# -------------------- modo minimal (páginas de modelos) --------------------
def hide_sidebar():
    """Oculta totalmente el contenedor del sidebar (para páginas 'minimal')."""
    st.markdown(
        "<style>[data-testid='stSidebar']{display:none;}</style>",
        unsafe_allow_html=True
    )

def navbar_minimal(title: str,
                   usuario: str | None = None,
                   msisdn: str | None = None,
                   show_logout: bool = True):
    """Barra superior compacta en una sola fila."""
    st.markdown("""
    <style>
      .ns-topbar-row { margin: .25rem 0 .35rem 0; }
      .ns-pill{
        font-size:.80rem; padding:.10rem .50rem; border-radius:999px;
        border:1px solid rgba(255,255,255,.16); opacity:.9; display:inline-block;
      }
      /* Botones y links de la fila */
      .ns-topbar a[kind="pageLink"], .ns-topbar .stButton>button{
        padding:.35rem .6rem; border-radius:10px;
        background: rgba(13,16,26,.45); border:1px solid rgba(255,255,255,.12);
        backdrop-filter: blur(6px);
      }
    </style>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns([1.1, 4, 2, 1], gap="small")
    with c1:
        st.page_link("main.py", label="← Inicio", icon="🏠")
    with c2:
        # Texto simple (sin caja grande)
        st.markdown(f"<div class='ns-topbar-row'><b>{title}</b></div>", unsafe_allow_html=True)
    with c3:
        label = (usuario or "Usuario") + (f" · {msisdn}" if msisdn else "")
        st.markdown(f"<div class='ns-topbar-row'><span class='ns-pill'>🔐 {label}</span></div>",
                    unsafe_allow_html=True)
    with c4:
        if show_logout and st.button("Salir", use_container_width=True):
            st.session_state.clear()
            st.switch_page("pages/00_Login.py")
# Mapa de títulos “bonitos” por archivo (sin .py)
_TITLE_MAP = {
    "main": "🏠 Inicio",
    "00_Login": "🔐 Login",
    "1_Modelo Hindmarsh–Rose": "🌊 Modelo Hindmarsh–Rose",
    "2_Modelo Rulkov": "🧩 Modelo Rulkov",
    "3_Modelo Izhikevich": "🧠 Modelo Izhikevich",
    "4_Modelo Hodgkin–Huxley": "⚡ Modelo Hodgkin–Huxley",
    "5_Invariantes Dinámicos": "📊 Invariantes Dinámicos",
    "6_Miguel Angel Calderón": "👤 Miguel Angel Calderón",
}

def _infer_page_title() -> str:
    """Intenta derivar un título bonito desde el archivo que llamó al helper."""
    caller_file = Path(inspect.stack()[1].filename).stem  # p.ej. '2_Modelo Rulkov'
    # 1) si está en el mapa, úsalo
    if caller_file in _TITLE_MAP:
        return _TITLE_MAP[caller_file]
    # 2) sino, limpiar nombre: quitar prefijo numérico y underscores
    name = re.sub(r"^\d+[_\-\.\s]*", "", caller_file)     # quita '00_', '1-', etc.
    name = name.replace("_", " ").strip()
    # capitalizar suave
    return name[:1].upper() + name[1:]
    
def sidebar_minimal(title: str | None = None,
                    usuario: str | None = None,
                    msisdn: str | None = None,
                    width_px: int = 300,
                    show_home: bool = True):
    """Sidebar compacto para páginas de modelos (sin botón Salir)."""
    if title is None:
        title = _infer_page_title()

    st.markdown(
        f"""
        <style>
          [data-testid="stSidebar"] {{ width: {width_px}px; min-width: {width_px}px; }}
          .ns-pill {{
            font-size:.80rem; padding:.10rem .50rem; border-radius:999px;
            border:1px solid rgba(255,255,255,.16); opacity:.9; display:inline-block;
          }}
        </style>
        """, unsafe_allow_html=True
    )
    with st.sidebar:
        label = (usuario or "Usuario") + (f" · {msisdn}" if msisdn else "")
        st.markdown(f"Hola **<span class='ns-pill'>{label}</span>**", unsafe_allow_html=True)
        if show_home:
            st.page_link("main.py", label="Inicio", icon="🏠")
        st.markdown(f"### {title}")
        st.divider()   # separador antes de tus controles

def _make_hash(password: str, salt: str) -> str:
    """SHA-256(salt + password) como hash simple (suficiente para demo)."""
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()

def check_credentials(usuario: str, password: str):
    """
    Devuelve (ok, row | None).  Lee usuarios.csv con columnas:
    usuario, nombre, rol, msisdn, pass_hash, salt
    """
    df = load_users()
    if df.empty:  # sin archivo o sin filas
        return False, None
    m = df[df["usuario"] == str(usuario).strip()]
    if m.empty:
        return False, None
    row = m.iloc[0]
    salt = str(row.get("salt", ""))
    expected = str(row.get("pass_hash", ""))
    calc = _make_hash(password, salt)
    ok = hmac.compare_digest(calc, expected)
    return ok, row

def render_web_login(next_page: str = "main.py"):
    """
    Formulario de login web.
    En éxito: st.session_state['auth_ok']=True y switch a next_page.
    """
    st.set_page_config(page_title="Login (web)", page_icon="🔐", layout="centered")

    st.title("Inicio de sesión")
    st.caption("Acceso web con usuario y contraseña.")

    usuario = st.text_input("Usuario", autocomplete="username")
    password = st.text_input("Contraseña", type="password", autocomplete="current-password")
    entrar = st.button("Entrar", type="primary", use_container_width=True)

    if entrar:
        ok, row = check_credentials(usuario, password)
        if ok:
            st.session_state["auth_ok"] = True
            st.session_state["usuario"] = row.get("usuario", "")
            # si tienes msisdn guardado, lo reutilizamos para que se vea en la UI
            st.session_state["user_msisdn"] = row.get("msisdn", "")
            st.success("Autenticado.")
            st.switch_page(next_page)
        else:
            st.error("Usuario o contraseña incorrectos.")