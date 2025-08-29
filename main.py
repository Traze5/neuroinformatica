# main.py — Menú principal mejorado (tema oscuro)

import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import plotly.express as px

# ============================================================
# Navegación (sin pages): usamos session_state["menu_idx"]
# ============================================================
MENU_ITEMS = [
    "Home",
    "Modelo Hindmarsh–Rose",
    "Modelo Izhikevich",
    "Modelo Rulkov",
    "Modelo Hodgkin–Huxley",
    "Invariantes",
    "Miguel Angel Calderón",
]
ICONOS = ["house", "activity", "cpu", "triangle", "bezier", "bar-chart-line", "person"]
MENU_INDEX = {name: i for i, name in enumerate(MENU_ITEMS)}
if "menu_idx" not in st.session_state:
    st.session_state["menu_idx"] = 0

def goto(name: str):
    st.session_state["menu_idx"] = MENU_INDEX[name]
    st.experimental_rerun()

# ============================================================
# Helper para ejecutar módulos externos
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
# Previews rápidos (parámetros estándar; horizontes cortos)
# ============================================================
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

def preview_rk(n_steps=1400, a=6.0, mu=0.001, sigma=-0.10, x=-1.958753, y=-3.983966):
    xs = np.empty(n_steps, float); ts = np.arange(n_steps, dtype=float)
    for i in range(n_steps):
        if x <= 0.0:
            denom = 1.0 - x
            denom = denom if abs(denom) > 1e-12 else (1e-12 if denom >= 0 else -1e-12)
            x1 = a/denom + y
        elif x < a + y:
            x1 = a + y
        else:
            x1 = -1.0
        y = y - mu*(x + 1.0 - sigma)
        x = x1
        xs[i] = x
    return ts, xs

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

# ============================================================
# Plot helper + métricas rápidas
# ============================================================
def small_line(x, y, title, ylab):
    fig = px.line(x=x, y=y, labels={'x': 'Tiempo', 'y': ylab})
    fig.update_layout(title=title, template="plotly_dark",
                      margin=dict(l=10, r=10, t=35, b=10), height=300)
    fig.update_traces(line=dict(width=1.6))
    return fig

def quick_metrics(y, t):
    amp = float(np.max(y) - np.min(y)) if len(y) else 0.0
    mean = float(np.mean(y)) if len(y) else 0.0
    # picos por percentil 90 (aprox)
    spk = 0
    if len(y) >= 3:
        thr = np.percentile(y, 90.0)
        xm1, xc, xp1 = y[:-2], y[1:-1], y[2:]
        spk = int(np.sum((xc > xm1) & (xc >= xp1) & (xc > thr)))
    dur = (t[-1] - t[0]) if len(t) else 0.0
    rate = (spk / dur) if dur > 0 else 0.0
    return amp, mean, spk, rate

def card_preview(title, preview_fun, ylab, goto_label):
    t, y = preview_fun()
    fig = small_line(t, y, title, ylab)
    st.plotly_chart(fig, use_container_width=True)
    amp, mean, spk, rate = quick_metrics(y, t)
    c1, c2, c3 = st.columns(3)
    c1.metric("Amplitud", f"{amp:.2f}")
    c2.metric("Picos", f"{spk}")
    c3.metric("Tasa aprox.", f"{rate:.2f} /t")
    st.button(f"➡️ Ir a {goto_label}", use_container_width=True, on_click=goto, args=(goto_label,))

# ============================================================
# Sidebar / Menú
# ============================================================
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=MENU_ITEMS,
        icons=ICONOS,
        menu_icon="cast",
        default_index=st.session_state["menu_idx"],
        styles={
            "container": {"padding": "0!important"},
            "icon": {"color": "#9aa0a6", "font-size": "18px"},
            "nav-link": {
                "font-size": "15px",
                "text-align": "left",
                "margin": "0",
                "padding": "8px 14px",
                "border-radius": "8px",
            },
            "nav-link-selected": {"background-color": "#1f2937"},
        },
    )
    # Sincroniza el índice al cambiar manualmente
    st.session_state["menu_idx"] = MENU_INDEX[selected]

# ============================================================
# Ruteo de páginas
# ============================================================
if selected == "Home":
    st.header("Herramienta de Simulación de Neuronas")
    st.caption("Vista previa rápida de los modelos. Usa los botones para saltar a cada simulador.")
    # Cuadrícula 2x2 de tarjetas
    col1, col2 = st.columns(2)
    with col1:
        card_preview("Modelo Hindmarsh–Rose", preview_hr, "x", "Modelo Hindmarsh–Rose")
    with col2:
        card_preview("Modelo Rulkov", preview_rk, "x", "Modelo Rulkov")
    col3, col4 = st.columns(2)
    with col3:
        card_preview("Modelo Hodgkin–Huxley", preview_hh, "V (mV)", "Modelo Hodgkin–Huxley")
    with col4:
        card_preview("Modelo Izhikevich", preview_izh, "v (mV)", "Modelo Izhikevich")

elif selected == "Modelo Hindmarsh–Rose":
    st.header("Simulación del Modelo Hindmarsh–Rose")
    run_module("hr.py")

elif selected == "Modelo Izhikevich":
    st.header("Simulación del Modelo Izhikevich")
    run_module("izh.py")   # cambia a "iz.py" si tu archivo se llama así

elif selected == "Modelo Rulkov":
    st.header("Simulación del Modelo Rulkov")
    run_module("rk.py")

elif selected == "Modelo Hodgkin–Huxley":
    st.header("Simulación del Modelo Hodgkin–Huxley")
    run_module("hh.py")

elif selected == "Invariantes":
    st.header("Invariantes Dinámicos")
    run_module("hri.py")

elif selected == "Miguel Angel Calderón":
    st.header("Miguel Angel Calderón")
    st.write("Data Scientist.")
