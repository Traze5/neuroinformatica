# main.py — Menú principal unificado (tema oscuro)
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import plotly.express as px

# ============================================================
# Helper para ejecutar módulos externos (sin romper la app)
# ============================================================
def run_module(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()
        exec(compile(code, path, "exec"), {})
    except FileNotFoundError:
        st.error(f"No se encontró el archivo: {path}")
    except Exception as e:
        st.error(f"Error al ejecutar {path}: {e}")
        st.exception(e)

# ============================================================
# Previews científicas (rápidas) para el Home
#   - Parámetros/presets estándar y horizontes cortos
#   - Computo ligero para que la app siga fluida
# ============================================================

# Hindmarsh–Rose (preset “normal” típico)
def preview_hr(t_max=2000, dt=0.02):
    a, b, c, d, e = 1.0, 3.0, 1.0, 5.0, 3.0
    r, s, x0 = 0.0021, 4.0, -1.6
    y0, z0 = -10.0, 3.0
    n = int(t_max / dt)
    x, y, z = x0, y0, z0
    xs = np.empty(n, float); ts = np.empty(n, float)
    for i in range(n):
        dx = y - a*x**3 + b*x**2 - z + e
        dy = c - d*x**2 - y
        dz = r*(s*(x - x0) - z)
        x += dt*dx; y += dt*dy; z += dt*dz
        xs[i] = x; ts[i] = i*dt
    return ts, xs

# Izhikevich (Regular Spiking)
def preview_izh(t_max=300, dt=1.0):
    a, b, c, d, I = 0.02, 0.2, -65.0, 8.0, 10.0
    v, u = -65.0, b*(-65.0)
    n = int(t_max / dt)
    vs = np.empty(n, float); ts = np.empty(n, float)
    for i in range(n):
        if v >= 30.0:
            v = c; u += d
        dv = 0.04*v*v + 5*v + 140 - u + I
        du = a*(b*v - u)
        v += dt*dv; u += dt*du
        vs[i] = min(v, 30.0); ts[i] = i*dt
    return ts, vs

# Rulkov (mapa 2D versión de tu rk.py / neu4.py)
def preview_rk(n_steps=1400, a=6.0, mu=0.001, sigma=-0.10, x=-1.958753, y=-3.983966):
    xs = np.empty(n_steps, float); ts = np.arange(n_steps, dtype=float)
    for i in range(n_steps):
        # rápido por tramos (a/(1-x)+y, a+y, -1)
        if x <= 0.0:
            denom = 1.0 - x
            denom = denom if abs(denom) > 1e-12 else (1e-12 if denom >= 0 else -1e-12)
            x1 = a/denom + y
        elif x < a + y:
            x1 = a + y
        else:
            x1 = -1.0
        y = y - mu*(x + 1.0 - sigma)  # tu forma lenta original
        x = x1
        xs[i] = x
    return ts, xs

# Hodgkin–Huxley (axón de calamar; paso I=10 μA/cm², 6.3 °C)
def _rates(V):
    an = 0.01*(V+55)/(1 - np.exp(-(V+55)/10)) if abs(V+55) > 1e-6 else 0.1
    bn = 0.125*np.exp(-(V+65)/80)
    am = 0.1*(V+40)/(1 - np.exp(-(V+40)/10)) if abs(V+40) > 1e-6 else 1.0
    bm = 4*np.exp(-(V+65)/18)
    ah = 0.07*np.exp(-(V+65)/20)
    bh = 1/(1 + np.exp(-(V+35)/10))
    return an, bn, am, bm, ah, bh

def preview_hh(t_max=50.0, dt=0.01, I=10.0):
    Cm, gNa, gK, gL = 1.0, 120.0, 36.0, 0.3
    ENa, EK, EL = 50.0, -77.0, -54.4
    V = -65.0
    # compuertas en reposo coherentes con V
    an, bn, am, bm, ah, bh = _rates(V)
    n = an/(an+bn); m = am/(am+bm); h = ah/(ah+bh)
    nsteps = int(t_max/dt)
    Vs = np.empty(nsteps, float); ts = np.empty(nsteps, float)
    for i in range(nsteps):
        INa = gNa*(m**3)*h*(ENa - V)
        IK  = gK*(n**4)*(EK - V)
        IL  = gL*(EL - V)
        dV = (INa + IK + IL + I)/Cm
        an, bn, am, bm, ah, bh = _rates(V)
        dn = an*(1-n) - bn*n
        dm = am*(1-m) - bm*m
        dh = ah*(1-h) - bh*h
        V += dt*dV; n += dt*dn; m += dt*dm; h += dt*dh
        Vs[i] = V; ts[i] = i*dt
    return ts, Vs

# Pequeño helper de plotting con estética consistente
def small_line(x, y, title):
    fig = px.line(x=x, y=y, labels={'x': 'Tiempo', 'y': 'x'})
    fig.update_layout(title=title, template="plotly_dark",
                      margin=dict(l=10, r=10, t=35, b=10), height=300)
    fig.update_traces(line=dict(width=1.5))
    return fig

# ============================================================
# Sidebar / Menú (formato actual, sin páginas nuevas)
# ============================================================
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=[
            "Home",
            "Modelo Hindmarsh–Rose",
            "Modelo Izhikevich",
            "Modelo Rulkov",
            "Modelo Hodgkin–Huxley",
            "Invariantes",
            "Miguel Angel Calderón",
        ],
        icons=[
            "house",
            "activity",
            "cpu",
            "triangle",
            "bezier",
            "bar-chart-line",
            "person",
        ],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important"},
            "icon": {"color": "#6c757d", "font-size": "18px"},
            "nav-link": {"font-size": "15px", "text-align": "left", "margin": "0"},
            "nav-link-selected": {"background-color": "#1f2937"},
        },
    )

# ============================================================
# Ruteo de páginas
# ============================================================

# Home — cuadrícula de previews
if selected == "Home":
    st.header("Aplicación de Simulación de Neuronas")
    st.write("Bienvenido. Usa el menú de la izquierda para navegar por los modelos y análisis.")

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    with col1:
        t, x = preview_hr()
        st.plotly_chart(small_line(t, x, "Modelo Hindmarsh–Rose"), use_container_width=True)

    with col2:
        t, x = preview_rk()
        st.plotly_chart(small_line(t, x, "Modelo Rulkov"), use_container_width=True)

    with col3:
        t, x = preview_hh()
        st.plotly_chart(small_line(t, x, "Modelo Hodgkin–Huxley"), use_container_width=True)

    with col4:
        t, x = preview_izh()
        st.plotly_chart(small_line(t, x, "Modelo Izhikevich"), use_container_width=True)

# Modelos
elif selected == "Modelo Hindmarsh–Rose":
    st.header("Simulación del Modelo Hindmarsh–Rose")
    run_module("hr.py")

elif selected == "Modelo Izhikevich":
    st.header("Simulación del Modelo Izhikevich")
    run_module("iz.py")   # <- tu módulo basado en neu3.py con ventana temporal

elif selected == "Modelo Rulkov":
    st.header("Simulación del Modelo Rulkov")
    run_module("rk.py")    # <- tu módulo de Rulkov

elif selected == "Modelo Hodgkin–Huxley":
    st.header("Simulación del Modelo Hodgkin–Huxley")
    run_module("hh.py")

# Invariantes (módulo único que consume session_state["timeseries"])
elif selected == "Invariantes":
    st.header("Invariantes Dinámicos")
    run_module("hri.py")   # <- módulo unificado de invariantes

# Ficha personal
elif selected == "Miguel Angel Calderón":
    st.header("Miguel Angel Calderón")
    st.write("Data Scientist.")
