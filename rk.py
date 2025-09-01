# -*- coding: utf-8 -*-
# Rulkov 2002 — Simulador con 3 presets canónicos y UI limpia

import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ===== Numba opcional =====
try:
    from numba import njit
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False

# ===== Mapa de Rulkov (forma canónica del paper) =====
def _f_fast(x, y, alpha):
    if x <= 0.0:
        denom = 1.0 - x
        if abs(denom) < 1e-12:
            denom = 1e-12 if denom >= 0 else -1e-12
        return alpha / denom + y
    elif x < alpha + y:
        return alpha + y
    else:
        return -1.0

def iterate_rulkov(alpha, mu, sigma, x0, y0, n_steps):
    x = np.empty(n_steps, dtype=np.float64)
    y = np.empty(n_steps, dtype=np.float64)
    xp, yp = float(x0), float(y0)
    for n in range(n_steps):
        xn1 = _f_fast(xp, yp, alpha)
        yn1 = yp - mu * (xp + 1.0) + mu * sigma   # <- usa x_n (paper 2002)
        x[n] = xn1
        y[n] = yn1
        xp, yp = xn1, yn1
    return x, y

if NUMBA_OK:
    @njit(cache=True, fastmath=True)
    def _f_fast_jit(x, y, alpha):
        if x <= 0.0:
            denom = 1.0 - x
            if abs(denom) < 1e-12:
                denom = 1e-12 if denom >= 0 else -1e-12
            return alpha / denom + y
        elif x < alpha + y:
            return alpha + y
        else:
            return -1.0

    @njit(cache=True, fastmath=True)
    def iterate_rulkov_jit(alpha, mu, sigma, x0, y0, n_steps):
        x = np.empty(n_steps, dtype=np.float64)
        y = np.empty(n_steps, dtype=np.float64)
        xp, yp = float(x0), float(y0)
        for n in range(n_steps):
            xn1 = _f_fast_jit(xp, yp, alpha)
            yn1 = yp - mu * (xp + 1.0) + mu * sigma
            x[n] = xn1
            y[n] = yn1
            xp, yp = xn1, yn1
        return x, y

def integrate(fun, t_max, dt, x0_state, args, method, variant):
    if dt <= 0:
        dt = 1.0
    n_steps = int(max(1, round(t_max / dt)))
    n_steps = int(min(n_steps, 2_000_000))
    alpha, mu, sigma = args
    if NUMBA_OK:
        x, y = iterate_rulkov_jit(alpha, mu, sigma, x0_state[0], x0_state[1], n_steps)
    else:
        x, y = iterate_rulkov(alpha, mu, sigma, x0_state[0], x0_state[1], n_steps)
    t = np.arange(n_steps, dtype=np.float64) * dt
    return t, x, y, None

def throttle(ms=150, key="rulkov_throttle_ms"):
    now = time.time()
    last = st.session_state.get(key, 0.0)
    if (now - last) * 1000.0 < ms:
        return True
    st.session_state[key] = now
    return False

STATE_KEYS = {
    "variant": "rulkov_variant",
    "params": "rulkov_params",
    "method": "rulkov_method",
    "last_sig": "rulkov_last_sig",
    "last_sim_t": "rulkov_last_sim_t",
    "data": "rulkov_data",
}

# ===== UI =====
st.set_page_config(page_title="Rulkov 2002 — Simulador", layout="wide")
st.title("Mapa de Rulkov — tres regímenes canónicos")

PRESETS = {
    # Del paper (Fig. 2b): “continuous tonic spiking”
    "Espigueo tónico continuo": dict(alpha=4.0, sigma=+0.01, mu=0.001, t_max=2000),
    # Del paper (Fig. 3a): spiking–bursting (intermedio)
    "Spiking–bursting (régimen intermedio)": dict(alpha=4.5, sigma=+0.14, mu=0.001, t_max=2000),
    # Del paper (Fig. 3c): spiking–bursting (alto α)
    "Spiking–bursting (alto α)": dict(alpha=6.0, sigma=+0.386, mu=0.001, t_max=2000),
}

with st.sidebar:
    st.subheader("Configuración")
    st.session_state[STATE_KEYS["variant"]] = "Rulkov 2D (canónico 2002)"
    st.session_state[STATE_KEYS["method"]] = "Iterated map (Rulkov)"

    preset_name = st.selectbox("Preset", list(PRESETS.keys()), index=0)
    p = PRESETS[preset_name]

    col1, col2 = st.columns(2)
    with col1:
        alpha = st.slider("α (rápido)", 3.5, 7.0, float(p["alpha"]), step=0.01)
        sigma = st.slider("σ (bias)", -0.5, 0.5, float(p["sigma"]), step=0.001)
    with col2:
        mu = st.number_input("μ (lento)", value=float(p["mu"]),
                             min_value=1e-6, max_value=0.01, step=0.0005, format="%.6f")
        t_max = st.number_input("N iteraciones (t_max)", value=int(p["t_max"]), step=100)

    st.markdown("---")
    x0 = st.number_input("x0", value=-1.50, step=0.01, format="%.2f")
    y0 = st.number_input("y0", value=-3.50, step=0.01, format="%.2f")
    dt = 1.0
    downsample = st.checkbox("Downsampling", value=True)
    max_points = st.number_input("Puntos máx. a dibujar", value=50_000, step=1000)

    st.session_state[STATE_KEYS["params"]] = dict(
        alpha=round(float(alpha), 4),
        mu=round(float(mu), 6),
        sigma=round(float(sigma), 4),
        x0=round(float(x0), 2),
        y0=round(float(y0), 2),
        dt=float(dt),
        t_max=int(t_max),
    )

if throttle(150):
    st.stop()

pars = st.session_state[STATE_KEYS["params"]]
t, x, y, _ = integrate(None, t_max=float(pars["t_max"]), dt=float(pars["dt"]),
                       x0_state=(pars["x0"], pars["y0"]),
                       args=(pars["alpha"], pars["mu"], pars["sigma"]),
                       method=st.session_state[STATE_KEYS["method"]],
                       variant=st.session_state[STATE_KEYS["variant"]])

base_key = f"{len(t)}_{int(t[-1])}"
t0, t1 = st.slider("Ventana temporal", min_value=float(t[0]), max_value=float(t[-1]),
                   value=(float(t[0]), float(t[-1])), step=float(pars["dt"]), key=f"win_{base_key}")
mask = (t >= t0) & (t <= t1)
t_v, x_v, y_v = t[mask], x[mask], y[mask]

if downsample and len(t_v) > max_points:
    stride = max(1, len(t_v) // int(max_points))
    t_v, x_v, y_v = t_v[::stride], x_v[::stride], y_v[::stride]
else:
    stride = 1

fig = go.Figure()
fig.add_trace(go.Scattergl(x=t_v, y=x_v, mode="lines", name="x", line=dict(width=1)))
fig.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10),
                  height=420, title=f"Rulkov — {preset_name}")
fig.update_xaxes(title_text="n (iteraciones)")
fig.update_yaxes(title_text="x")
st.plotly_chart(fig, use_container_width=True)

# Exportes para módulos (invariantes, etc.)
st.session_state["rulkov_timeseries"] = dict(
    t=t, x=x, y=y, z=None, variant=st.session_state[STATE_KEYS["variant"]],
    params=st.session_state[STATE_KEYS["params"]], method=st.session_state[STATE_KEYS["method"]]
)
st.session_state["timeseries"] = dict(
    t=t, x=x, y=y, z=None, model="rulkov", params=st.session_state[STATE_KEYS["params"]]
)
st.session_state[STATE_KEYS["data"]] = dict(t_v=t_v, x_v=x_v, y_v=y_v, stride=stride)
st.session_state[STATE_KEYS["last_sig"]] = base_key
st.session_state[STATE_KEYS["last_sim_t"]] = time.time()
