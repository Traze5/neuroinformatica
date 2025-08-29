# izh.py — Modelo de Izhikevich (2003)
# Patrón idéntico a hr.py: throttle, Euler por defecto (RK4 opcional),
# Numba opcional, submuestreo, detección de picos, retratos OFF por defecto,
# exportes: st.session_state["izh_timeseries"] y ["timeseries"].

import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ------------------------------------------------------------
# Aceleración opcional con Numba
# ------------------------------------------------------------
try:
    from numba import njit
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False

# ------------------------------------------------------------
# Dinámica Izhikevich
# v' = 0.04 v^2 + 5 v + 140 - u + I
# u' = a (b v - u)
# Disparo: si v >= 30 mV -> registrar pico, luego v=c, u=u+d
# ------------------------------------------------------------
def izh_euler(a, b, c, d, I, v0, u0, dt, n_steps):
    v = np.empty(n_steps, dtype=np.float64)
    u = np.empty(n_steps, dtype=np.float64)
    spikes = np.zeros(n_steps, dtype=np.bool_)
    v_t, u_t = float(v0), float(u0)

    for i in range(n_steps):
        # Paso Euler
        dv = 0.04 * v_t * v_t + 5.0 * v_t + 140.0 - u_t + I
        du = a * (b * v_t - u_t)
        v_t = v_t + dt * dv
        u_t = u_t + dt * du

        # Evento de spike
        if v_t >= 30.0:
            v[i] = 30.0
            spikes[i] = True
            v_t = c
            u_t = u_t + d
        else:
            v[i] = v_t

        u[i] = u_t

    return v, u, spikes

def izh_rk4(a, b, c, d, I, v0, u0, dt, n_steps):
    v = np.empty(n_steps, dtype=np.float64)
    u = np.empty(n_steps, dtype=np.float64)
    spikes = np.zeros(n_steps, dtype=np.bool_)
    v_t, u_t = float(v0), float(u0)

    for i in range(n_steps):
        # k1
        dv1 = 0.04 * v_t * v_t + 5.0 * v_t + 140.0 - u_t + I
        du1 = a * (b * v_t - u_t)

        # k2
        v2 = v_t + 0.5 * dt * dv1
        u2 = u_t + 0.5 * dt * du1
        dv2 = 0.04 * v2 * v2 + 5.0 * v2 + 140.0 - u2 + I
        du2 = a * (b * v2 - u2)

        # k3
        v3 = v_t + 0.5 * dt * dv2
        u3 = u_t + 0.5 * dt * du2
        dv3 = 0.04 * v3 * v3 + 5.0 * v3 + 140.0 - u3 + I
        du3 = a * (b * v3 - u3)

        # k4
        v4 = v_t + dt * dv3
        u4 = u_t + dt * du3
        dv4 = 0.04 * v4 * v4 + 5.0 * v4 + 140.0 - u4 + I
        du4 = a * (b * v4 - u4)

        v_t = v_t + (dt / 6.0) * (dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4)
        u_t = u_t + (dt / 6.0) * (du1 + 2.0 * du2 + 2.0 * du3 + du4)

        # Evento de spike (aplicado al final del paso)
        if v_t >= 30.0:
            v[i] = 30.0
            spikes[i] = True
            v_t = c
            u_t = u_t + d
        else:
            v[i] = v_t

        u[i] = u_t

    return v, u, spikes

if NUMBA_OK:
    @njit(cache=True, fastmath=True)
    def izh_euler_jit(a, b, c, d, I, v0, u0, dt, n_steps):
        v = np.empty(n_steps, dtype=np.float64)
        u = np.empty(n_steps, dtype=np.float64)
        spikes = np.zeros(n_steps, dtype=np.bool_)
        v_t, u_t = float(v0), float(u0)
        for i in range(n_steps):
            dv = 0.04 * v_t * v_t + 5.0 * v_t + 140.0 - u_t + I
            du = a * (b * v_t - u_t)
            v_t = v_t + dt * dv
            u_t = u_t + dt * du
            if v_t >= 30.0:
                v[i] = 30.0
                spikes[i] = True
                v_t = c
                u_t = u_t + d
            else:
                v[i] = v_t
            u[i] = u_t
        return v, u, spikes

# ------------------------------------------------------------
# Firma integrate(...) igual a hr.py
# ------------------------------------------------------------
def integrate(fun, t_max, dt, x0_state, args, method, variant):
    # x0_state = (v0, u0), args = (a, b, c, d, I)
    n_steps = int(max(1, round(t_max / max(1e-9, dt))))
    a, b, c, d, I = args
    v0, u0 = x0_state

    if method == "RK4":
        v, u, spikes = izh_rk4(a, b, c, d, I, v0, u0, dt, n_steps)
    else:
        if NUMBA_OK:
            v, u, spikes = izh_euler_jit(a, b, c, d, I, v0, u0, dt, n_steps)
        else:
            v, u, spikes = izh_euler(a, b, c, d, I, v0, u0, dt, n_steps)

    t = np.arange(n_steps, dtype=np.float64) * dt
    # Guardamos los spikes detectados en session_state (para marcadores)
    st.session_state["izh_spike_idx"] = np.where(spikes)[0]
    return t, v, u, None

# ------------------------------------------------------------
# Utilidades comunes (picos, ráfagas, throttle)
# ------------------------------------------------------------
def detect_spikes_percentile(x, perc=90.0):
    if len(x) < 3:
        return np.array([], dtype=int)
    thr = np.percentile(x, perc)
    xm1 = x[:-2]; xc = x[1:-1]; xp1 = x[2:]
    return np.where((xc > xm1) & (xc >= xp1) & (xc > thr))[0] + 1

def detect_bursts(x, hi_pct=0.75, lo_pct=0.55):
    if len(x) == 0:
        return []
    xmin, xmax = float(np.min(x)), float(np.max(x))
    rng = xmax - xmin if xmax > xmin else 1.0
    hi = xmin + hi_pct * rng
    lo = xmin + lo_pct * rng
    inside = False; start = 0; bands = []
    for i, v in enumerate(x):
        if not inside and v >= hi:
            inside = True; start = i
        elif inside and v <= lo:
            inside = False; bands.append((start, i))
    if inside:
        bands.append((start, len(x) - 1))
    return bands

def throttle(ms=150, key="izh_throttle_ms"):
    now = time.time()
    last = st.session_state.get(key, 0.0)
    if (now - last) * 1000.0 < ms:
        return True
    st.session_state[key] = now
    return False

STATE_KEYS = {
    "variant": "izh_variant",
    "params": "izh_params",
    "method": "izh_method",
    "last_sig": "izh_last_sig",
    "last_sim_t": "izh_last_sim_t",
    "data": "izh_data",
}

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.set_page_config(page_title="Izhikevich — Simulación interactiva", layout="wide")
st.title("Izhikevich — Simulación interactiva")

# Presets clásicos (Izhikevich 2003/2004)
PRESETS = {
    "CH (Chattering)":        dict(a=0.02, b=0.20, c=-50.0, d=2.0,  I=10.0, v0=-50.0, u0=-10.0, dt=1.0, t_max=1500),
    "RS (Regular Spiking)":   dict(a=0.02, b=0.20, c=-65.0, d=8.0,  I=10.0, v0=-65.0, u0=-13.0, dt=1.0,  t_max=1500),
    "FS (Fast Spiking)":      dict(a=0.10, b=0.20, c=-65.0, d=2.0,  I=10.0, v0=-65.0, u0=-13.0, dt=0.5, t_max=1500),
    "IB (Intrinsically Burst)": dict(a=0.02, b=0.20, c=-55.0, d=4.0, I=10.0, v0=-55.0, u0=-11.0, dt=1.0, t_max=1500),
    "LTS (Low-Threshold)":    dict(a=0.02, b=0.25, c=-65.0, d=2.0,  I=10.0, v0=-65.0, u0=-16.25, dt=1.0, t_max=1500),
}

with st.sidebar:
    st.subheader("Configuración")

    # Variante y método: únicos (guardados para compatibilidad)
    variant = "Izhikevich (2003)"
    method = st.selectbox("Método numérico", ["Euler", "RK4"], index=0)
    st.session_state[STATE_KEYS["variant"]] = variant
    st.session_state[STATE_KEYS["method"]] = method

    preset_name = st.selectbox("Preset", list(PRESETS.keys()), index=0)
    p = PRESETS[preset_name]

    # Parámetros (3×3)
    col1, col2, col3 = st.columns(3)
    with col1:
        a = st.number_input("a", value=float(p["a"]), step=0.001, format="%.3f")
        b = st.number_input("b", value=float(p["b"]), step=0.01, format="%.2f")
        c = st.number_input("c (reset v)", value=float(p["c"]), step=1.0, format="%.1f")
    with col2:
        d = st.number_input("d (reset u)", value=float(p["d"]), step=0.5, format="%.1f")
        I = st.number_input("I (corriente DC)", value=float(p["I"]), step=0.5, format="%.2f")
        v0 = st.number_input("v0 (init)", value=float(p["v0"]), step=1.0, format="%.1f")
    with col3:
        u0 = st.number_input("u0 (init)", value=float(p["u0"]), step=1.0, format="%.2f")
        dt = st.number_input("dt (ms)", value=float(p["dt"]), step=0.5, format="%.1f")
        t_max = st.number_input("t_max (ms)", value=int(p["t_max"]), step=100)

    st.markdown("---")
    show_phase_xy = st.checkbox("Retrato de fase v–u (OFF por defecto)", value=False)
    mark_spikes = st.checkbox("Marcar picos (ON por defecto)", value=True)
    show_bursts = st.checkbox("Mostrar ráfagas (OFF por defecto)", value=False)
    downsample = st.checkbox("Activar downsampling", value=True)
    max_points = st.number_input("Puntos máx. a dibujar", value=50_000, step=1000)

    st.session_state[STATE_KEYS["params"]] = dict(
        a=a, b=b, c=c, d=d, I=I, v0=v0, u0=u0, dt=dt, t_max=t_max
    )

# Throttle ~150 ms
if throttle(150):
    st.stop()

# Simulación
args = (a, b, c, d, I)
t, v, u, _ = integrate(None, t_max=float(t_max), dt=float(dt),
                       x0_state=(v0, u0), args=args, method=method, variant=variant)

# Ventana temporal
base_key = f"{len(t)}_{int(t[-1])}"
t0, t1 = st.slider(
    "Ventana temporal",
    min_value=float(t[0]), max_value=float(t[-1]),
    value=(float(t[0]), float(t[-1])),
    step=float(dt), key=f"win_{base_key}"
)
mask = (t >= t0) & (t <= t1)
t_v, v_v, u_v = t[mask], v[mask], u[mask]

# Downsampling
if downsample and len(t_v) > max_points:
    stride = max(1, len(t_v) // int(max_points))
    t_v, v_v, u_v = t_v[::stride], v_v[::stride], u_v[::stride]
else:
    stride = 1

# Picos: usamos los del propio modelo si existen; si no, fallback por percentil
spk_idx_model = st.session_state.get("izh_spike_idx", np.array([], dtype=int))
spk_idx = spk_idx_model[(spk_idx_model >= np.searchsorted(t, t0))
                        & (spk_idx_model <= np.searchsorted(t, t1))]
if mark_spikes and spk_idx.size == 0:
    spk_idx = detect_spikes_percentile(v_v)

# Ráfagas
bursts = detect_bursts(v_v) if show_bursts else []

# Gráfico principal (v vs t)
fig = go.Figure()
fig.add_trace(go.Scattergl(x=t_v, y=v_v, mode="lines", name="v", line=dict(width=1)))
if mark_spikes and spk_idx.size > 0:
    fig.add_trace(go.Scattergl(x=t_v[spk_idx], y=v_v[spk_idx], mode="markers",
                               name="picos", marker=dict(size=5)))
if show_bursts and bursts:
    for b0, b1 in bursts:
        fig.add_vrect(x0=t_v[b0], x1=t_v[b1], fillcolor="white", opacity=0.08, line_width=0)

fig.update_traces(fill=None, selector=dict(type='scattergl'))
fig.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10),
                  height=420, title=f"Izhikevich — {preset_name}")
fig.update_xaxes(title_text="t (ms)")
fig.update_yaxes(title_text="v (mV)")
st.plotly_chart(fig, use_container_width=True)

# Retrato de fase v–u (opcional)
if show_phase_xy:
    fig2 = go.Figure()
    fig2.add_trace(go.Scattergl(x=v_v, y=u_v, mode="lines", name="fase v–u", line=dict(width=1)))
    fig2.update_traces(fill=None, selector=dict(type='scattergl'))
    fig2.update_layout(template="plotly_dark", height=360,
                       margin=dict(l=10, r=10, t=30, b=10), title="Retrato de fase v–u")
    fig2.update_xaxes(title_text="v"); fig2.update_yaxes(title_text="u")
    st.plotly_chart(fig2, use_container_width=True)

# Exportes de estado (compatibles con invariantes)
st.session_state["izh_timeseries"] = dict(
    t=t, x=v, y=u, z=None, variant=variant,
    params=st.session_state[STATE_KEYS["params"]], method=method
)
st.session_state["timeseries"] = dict(
    t=t, x=v, y=u, z=None, model="izhikevich",
    params=st.session_state[STATE_KEYS["params"]]
)

st.session_state[STATE_KEYS["data"]] = dict(t_v=t_v, x_v=v_v, y_v=u_v, stride=stride)
st.session_state[STATE_KEYS["last_sig"]] = base_key
st.session_state[STATE_KEYS["last_sim_t"]] = time.time()
