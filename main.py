# main.py
import streamlit as st
import numpy as np
import plotly.express as px
from modules.ui import require_auth, generar_menu



st.set_page_config(page_title="Herramienta de Simulación de Neuronas", layout="wide")

# --- guardia de login + menú propio ---
require_auth(login_page="pages/00_Auto_Auth.py")
generar_menu(
    usuario=st.session_state.get("usuario"),
    msisdn=st.session_state.get("user_msisdn"),
)

# ----------------------------- ESTILO (ligero, “futurista”) -----------------------------
st.markdown("""
<style>
/* botones y links */
.stButton button, a[kind="pageLink"] {
  border-radius: 12px; padding: .6rem 1rem;
  backdrop-filter: blur(6px);
  background: rgba(13, 16, 26, 0.55);
  border: 1px solid rgba(255,255,255,0.12);
  box-shadow: 0 6px 24px rgba(0,0,0,0.25), 0 0 0 1px rgba(255,255,255,0.03) inset;
}
/* métricas */
[data-testid="stMetric"] {
  background: rgba(31, 41, 55, .45);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 14px; padding: .6rem .8rem; margin-top: .2rem;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------- PREVIEWS (tus funciones) ---------------------------------
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

# --------------------------------- helpers de visualización ---------------------------------
def small_line(x, y, title, ylab):
    fig = px.line(x=x, y=y, labels={'x':'Tiempo','y':ylab})
    fig.update_layout(title=title, template="plotly_dark",
                      margin=dict(l=10, r=10, t=35, b=10), height=300)
    fig.update_traces(line=dict(width=1.6))
    return fig

def quick_metrics_no_rate(y):
    amp = float(np.max(y) - np.min(y)) if len(y) else 0.0
    spk = 0
    if len(y) >= 3:
        thr = np.percentile(y, 90.0)
        xm1, xc, xp1 = y[:-2], y[1:-1], y[2:]
        spk = int(np.sum((xc > xm1) & (xc >= xp1) & (xc > thr)))
    return amp, spk

def card_preview(title, preview_fun, ylab, page_path):
    t, y = preview_fun()
    st.plotly_chart(small_line(t, y, title, ylab), use_container_width=True)
    amp, spk = quick_metrics_no_rate(y)
    c1, c2 = st.columns(2)
    c1.metric("Amplitud", f"{amp:.2f}")
    c2.metric("Picos", f"{spk}")
    st.page_link(page_path, label=f"➡️ Ir a {title}", use_container_width=True)
    if st.button(f"Abrir {title}", use_container_width=True):
        st.switch_page(page_path)

# --------------------------------- HOME ---------------------------------
st.header("Herramienta de Simulación de Neuronas")
st.caption("Vista previa rápida de los modelos. Usa los enlaces o botones para ir a cada simulador.")

col1, col2 = st.columns(2)
with col1:
    card_preview("Modelo Hindmarsh–Rose", preview_hr, "x", "pages/1_Modelo_Hindmarsh_Rose.py")
with col2:
    card_preview("Modelo Rulkov",         preview_rk, "x", "pages/2_Modelo Rulkov.py")

col3, col4 = st.columns(2)
with col3:
    card_preview("Modelo Hodgkin–Huxley", preview_hh, "V", "pages/4_Modelo_Hodgkin_Huxley.py")
with col4:
    card_preview("Modelo Izhikevich",     preview_izh,"v","pages/3_Modelo Izhikevich.py")






