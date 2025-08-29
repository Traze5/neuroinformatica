import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ------------------------------------------------------------
# Numba opcional
# ------------------------------------------------------------
try:
    from numba import njit
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False

# ------------------------------------------------------------
# Rulkov 2D (forma de tu neu4.py)
#   x_{n+1} = f(x_n,y_n;a)
#   y_{n+1} = y_n - mu * (x_{n+1} - sigma)
#   f(x,y;a) = a/(1-x) + y  si x <= 0
#              a + y        si 0 < x < a + y
#             -1            si x >= a + y
# ------------------------------------------------------------
def _f_fast(x, y, a):
    if x <= 0.0:
        denom = 1.0 - x
        if abs(denom) < 1e-12:
            denom = 1e-12 if denom >= 0 else -1e-12
        return a / denom + y
    elif x < a + y:
        return a + y
    else:
        return -1.0

def iterate_rulkov(a, mu, sigma, x0, y0, n_steps):
    x = np.empty(n_steps, dtype=np.float64)
    y = np.empty(n_steps, dtype=np.float64)
    xp, yp = float(x0), float(y0)
    for n in range(n_steps):
        xn1 = _f_fast(xp, yp, a)
        yn1 = yp - mu * (xn1 - sigma)
        x[n] = xn1
        y[n] = yn1
        xp, yp = xn1, yn1
    return x, y

if NUMBA_OK:
    @njit(cache=True, fastmath=True)
    def _f_fast_jit(x, y, a):
        if x <= 0.0:
            denom = 1.0 - x
            if abs(denom) < 1e-12:
                denom = 1e-12 if denom >= 0 else -1e-12
            return a / denom + y
        elif x < a + y:
            return a + y
        else:
            return -1.0

    @njit(cache=True, fastmath=True)
    def iterate_rulkov_jit(a, mu, sigma, x0, y0, n_steps):
        x = np.empty(n_steps, dtype=np.float64)
        y = np.empty(n_steps, dtype=np.float64)
        xp, yp = float(x0), float(y0)
        for n in range(n_steps):
            xn1 = _f_fast_jit(xp, yp, a)
            yn1 = yp - mu * (xn1 - sigma)
            x[n] = xn1
            y[n] = yn1
            xp, yp = xn1, yn1
        return x, y

# ------------------------------------------------------------
# integrate(...) con la firma de hr.py
# ------------------------------------------------------------
def integrate(fun, t_max, dt, x0_state, args, method, variant):
    if dt <= 0:
        dt = 1.0
    n_steps = int(max(1, round(t_max / dt)))
    # protección suave para no colgar la UI
    n_steps = int(min(n_steps, 2_000_000))
    a, mu, sigma = args
    if NUMBA_OK:
        x, y = iterate_rulkov_jit(a, mu, sigma, x0_state[0], x0_state[1], n_steps)
    else:
        x, y = iterate_rulkov(a, mu, sigma, x0_state[0], x0_state[1], n_steps)
    t = np.arange(n_steps, dtype=np.float64) * dt
    return t, x, y, None

# ------------------------------------------------------------
# utilidades picos / ráfagas / throttle
# ------------------------------------------------------------
def detect_spikes(x, perc=90.0):
    if len(x) < 3:
        return np.array([], dtype=int)
    thr = np.percentile(x, perc)
    xm1, xc, xp1 = x[:-2], x[1:-1], x[2:]
    return np.where((xc > xm1) & (xc >= xp1) & (xc > thr))[0] + 1

def detect_bursts(x, hi_pct=0.75, lo_pct=0.55):
    if len(x) == 0:
        return []
    xmin, xmax = float(np.min(x)), float(np.max(x))
    rng = xmax - xmin if xmax > xmin else 1.0
    hi, lo = xmin + hi_pct * rng, xmin + lo_pct * rng
    inside = False; start = 0; bands = []
    for i, v in enumerate(x):
        if not inside and v >= hi:
            inside = True; start = i
        elif inside and v <= lo:
            inside = False; bands.append((start, i))
    if inside:
        bands.append((start, len(x) - 1))
    return bands

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

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.set_page_config(page_title="Mapa de Rulkov — Simulación interactiva", layout="wide")
st.title("Mapa de Rulkov — Simulación interactiva")

# Presets base (sin limitar rangos de entrada)
PRESETS = {
    "Spiking–Bursting (a=6.0, σ=-0.10, μ=0.001)": dict(a=6.0, mu=0.001, sigma=-1.10, x0=-1.958753, y0=-3.983966, t_max=1400),
    "Tónico (a=4.0, σ=0.10, μ=0.001)":            dict(a=4.0, mu=0.001, sigma= 1.10, x0=-1.95,     y0=-3.98,     t_max=15000),
    "Silencio (a=4.0, σ=-0.01, μ=0.001)":         dict(a=4.0, mu=0.001, sigma=-1.01, x0=-1.95,     y0=-3.98,     t_max=15000),
}

with st.sidebar:
    st.subheader("Configuración")
    st.session_state[STATE_KEYS["variant"]] = "Rulkov 2D (neu4.py)"
    st.session_state[STATE_KEYS["method"]] = "Iterated map (Rulkov)"

    preset_name = st.selectbox("Preset", list(PRESETS.keys()), index=0)
    p = PRESETS[preset_name]

    col1, col2, col3 = st.columns(3)
    with col1:
        a = st.number_input("a (rápido)", value=float(p["a"]), step=0.01, format="%.4f")
        mu = st.number_input("μ (lento)", value=float(p["mu"]), step=1e-4, format="%.6f")
        sigma = st.number_input("σ (desplazamiento)", value=float(p["sigma"]), step=0.01, format="%.4f")
    with col2:
        x0 = st.number_input("x0 (init)", value=float(p["x0"]), step=0.01, format="%.6f")
        y0 = st.number_input("y0 (init)", value=float(p["y0"]), step=0.01, format="%.6f")
        dt = st.number_input("dt (iteraciones)", value=1.0, step=1.0, format="%.0f")
    with col3:
        t_max = st.number_input("N iteraciones (t_max)", value=int(p["t_max"]), step=100)
        downsample = st.checkbox("Activar downsampling", value=True)
        max_points = st.number_input("Puntos máx. a dibujar", value=50_000, step=1000)

    st.markdown("---")
    show_phase_xy = st.checkbox("Retrato de fase x–y (OFF por defecto)", value=False)
    mark_spikes = st.checkbox("Marcar picos (ON por defecto)", value=True)
    show_bursts = st.checkbox("Mostrar ráfagas (OFF por defecto)", value=False)

    st.session_state[STATE_KEYS["params"]] = dict(
        a=a, mu=mu, sigma=sigma, x0=x0, y0=y0, dt=dt, t_max=t_max
    )

# ------------------------------------------------------------
# Throttle
# ------------------------------------------------------------
if throttle(150):
    st.stop()

# ------------------------------------------------------------
# Simulación
# ------------------------------------------------------------
args = (a, mu, sigma)
t, x, y, z = integrate(None, t_max=float(t_max), dt=float(dt),
                       x0_state=(x0, y0), args=args,
                       method=st.session_state[STATE_KEYS["method"]],
                       variant=st.session_state[STATE_KEYS["variant"]])

# Ventana temporal
base_key = f"{len(t)}_{int(t[-1])}"
t0, t1 = st.slider(
    "Ventana temporal",
    min_value=float(t[0]), max_value=float(t[-1]),
    value=(float(t[0]), float(t[-1])),
    step=float(dt), key=f"win_{base_key}"
)
mask = (t >= t0) & (t <= t1)
t_v, x_v, y_v = t[mask], x[mask], y[mask]

# Downsampling
if downsample and len(t_v) > max_points:
    stride = max(1, len(t_v) // int(max_points))
    t_v, x_v, y_v = t_v[::stride], x_v[::stride], y_v[::stride]
else:
    stride = 1

# Eventos
spk_idx = detect_spikes(x_v) if mark_spikes else np.array([], dtype=int)
bursts = detect_bursts(x_v) if show_bursts else []

# ------------------------------------------------------------
# Gráfico principal (sin relleno)
# ------------------------------------------------------------
fig = go.Figure()
fig.add_trace(go.Scattergl(x=t_v, y=x_v, mode="lines", name="x", line=dict(width=1)))
if mark_spikes and spk_idx.size > 0:
    fig.add_trace(go.Scattergl(x=t_v[spk_idx], y=x_v[spk_idx], mode="markers",
                               name="picos", marker=dict(size=5)))
if show_bursts and bursts:
    for b0, b1 in bursts:
        fig.add_vrect(x0=t_v[b0], x1=t_v[b1], fillcolor="white", opacity=0.08, line_width=0)

fig.update_traces(fill=None, selector=dict(type='scattergl'))
fig.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10),
                  height=420, title=f"Rulkov — {preset_name}")
fig.update_xaxes(title_text="t (iteraciones)")
fig.update_yaxes(title_text="x")
try:
    st.plotly_chart(fig, use_container_width=True, theme=None)
except TypeError:
    st.plotly_chart(fig, use_container_width=True)

# Retrato de fase x–y (opcional)
if show_phase_xy:
    fig2 = go.Figure()
    fig2.add_trace(go.Scattergl(x=x_v, y=y_v, mode="lines", name="fase x–y", line=dict(width=1)))
    fig2.update_traces(fill=None, selector=dict(type='scattergl'))
    fig2.update_layout(template="plotly_dark", height=360,
                       margin=dict(l=10, r=10, t=30, b=10), title="Retrato de fase x–y")
    fig2.update_xaxes(title_text="x"); fig2.update_yaxes(title_text="y")
    try:
        st.plotly_chart(fig2, use_container_width=True, theme=None)
    except TypeError:
        st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------------------------
# Exportes de estado (para invariantes y otros módulos)
# ------------------------------------------------------------
st.session_state["rulkov_timeseries"] = dict(
    t=t, x=x, y=y, z=None,
    variant=st.session_state[STATE_KEYS["variant"]],
    params=st.session_state[STATE_KEYS["params"]],
    method=st.session_state[STATE_KEYS["method"]]
)
st.session_state["timeseries"] = dict(
    t=t, x=x, y=y, z=None, model="rulkov",
    params=st.session_state[STATE_KEYS["params"]]
)
st.session_state[STATE_KEYS["data"]] = dict(t_v=t_v, x_v=x_v, y_v=y_v, stride=stride)
st.session_state[STATE_KEYS["last_sig"]] = base_key
st.session_state[STATE_KEYS["last_sim_t"]] = time.time()
