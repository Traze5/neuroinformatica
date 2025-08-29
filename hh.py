# hh.py — Hodgkin–Huxley (squid giant axon, 6.3°C)
# Patrón idéntico a hr.py: Euler por defecto (RK4 opcional), Numba opcional,
# throttle ~150 ms, downsampling, picos ON, ráfagas OFF, retratos OFF por defecto,
# exportes "hh_timeseries" y genérico "timeseries" (x=V, y=n, z=m).

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
# Utilidades HH
# ------------------------------------------------------------
def vtrap(x, y):
    # Evita indeterminaciones para x/y ~ 0
    s = x / y
    return y * (1.0 - np.exp(-s)) if abs(s) > 1e-6 else y * (1.0 - (1.0 - s + 0.5*s*s))

def rates(V):
    # V en mV (referido a 0 mV), formas clásicas HH (1952; 6.3°C)
    an = 0.01 * vtrap(-(V + 55.0), 10.0) / (np.exp(-(V + 55.0)/10.0) - 1.0) if abs(V + 55.0) > 1e-6 else 0.01/ (1.0/10.0)
    # reescrito de forma estable (equivalente a 0.01*(V+55)/(1 - exp(-(V+55)/10)))
    an = 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0)/10.0)) if abs(V + 55.0) > 1e-6 else 0.01*10.0
    bn = 0.125 * np.exp(-(V + 65.0)/80.0)

    am = 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0)/10.0)) if abs(V + 40.0) > 1e-6 else 0.1*10.0
    bm = 4.0 * np.exp(-(V + 65.0)/18.0)

    ah = 0.07 * np.exp(-(V + 65.0)/20.0)
    bh = 1.0 / (1.0 + np.exp(-(V + 35.0)/10.0))
    return an, bn, am, bm, ah, bh

def hh_rhs(V, m, h, n, params, Iext):
    Cm, gNa, gK, gL, ENa, EK, EL = params
    # Corrientes: g * (Erev - V) (convención entrada positiva si E > V)
    INa = gNa * (m**3) * h * (ENa - V)
    IK  = gK  * (n**4)     * (EK  - V)
    IL  = gL               * (EL  - V)
    dVdt = (INa + IK + IL + Iext) / Cm

    an, bn, am, bm, ah, bh = rates(V)
    dndt = an * (1.0 - n) - bn * n
    dmdt = am * (1.0 - m) - bm * m
    dhdt = ah * (1.0 - h) - bh * h
    return dVdt, dmdt, dhdt, dndt

# ------------------------------------------------------------
# Integradores
# ------------------------------------------------------------
def step_euler(V, m, h, n, params, Iext, dt):
    dV, dm, dh, dn = hh_rhs(V, m, h, n, params, Iext)
    return (V + dt*dV, m + dt*dm, h + dt*dh, n + dt*dn)

def step_rk4(V, m, h, n, params, Iext, dt):
    # k1
    dV1, dm1, dh1, dn1 = hh_rhs(V, m, h, n, params, Iext)
    # k2
    dV2, dm2, dh2, dn2 = hh_rhs(V + 0.5*dt*dV1, m + 0.5*dt*dm1, h + 0.5*dt*dh1, n + 0.5*dt*dn1, params, Iext)
    # k3
    dV3, dm3, dh3, dn3 = hh_rhs(V + 0.5*dt*dV2, m + 0.5*dt*dm2, h + 0.5*dt*dh2, n + 0.5*dt*dn2, params, Iext)
    # k4
    dV4, dm4, dh4, dn4 = hh_rhs(V + dt*dV3, m + dt*dm3, h + dt*dh3, n + dt*dn3, params, Iext)
    Vn = V + (dt/6.0)*(dV1 + 2*dV2 + 2*dV3 + dV4)
    mn = m + (dt/6.0)*(dm1 + 2*dm2 + 2*dm3 + dm4)
    hn = h + (dt/6.0)*(dh1 + 2*dh2 + 2*dh3 + dh4)
    nn = n + (dt/6.0)*(dn1 + 2*dn2 + 2*dn3 + dn4)
    return Vn, mn, hn, nn

if NUMBA_OK:
    @njit(cache=True, fastmath=True)
    def step_euler_jit(V, m, h, n, params, Iext, dt):
        Cm, gNa, gK, gL, ENa, EK, EL = params
        # rates
        an = 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0)/10.0)) if abs(V + 55.0) > 1e-6 else 0.01*10.0
        bn = 0.125 * np.exp(-(V + 65.0)/80.0)
        am = 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0)/10.0)) if abs(V + 40.0) > 1e-6 else 0.1*10.0
        bm = 4.0 * np.exp(-(V + 65.0)/18.0)
        ah = 0.07 * np.exp(-(V + 65.0)/20.0)
        bh = 1.0 / (1.0 + np.exp(-(V + 35.0)/10.0))
        # currents
        INa = gNa * (m**3) * h * (ENa - V)
        IK  = gK  * (n**4)     * (EK  - V)
        IL  = gL               * (EL  - V)
        dVdt = (INa + IK + IL + Iext) / Cm
        dndt = an * (1.0 - n) - bn * n
        dmdt = am * (1.0 - m) - bm * m
        dhdt = ah * (1.0 - h) - bh * h
        return (V + dt*dVdt, m + dt*dmdt, h + dt*dhdt, n + dt*dndt)

# Mantener firma integrate(...) del patrón hr.py
def integrate(fun, t_max, dt, x0_state, args, method, variant):
    # x0_state = (V0, m0, h0, n0)
    # args = (Cm, gNa, gK, gL, ENa, EK, EL, Iext)
    n_steps = int(max(1, round(t_max / max(1e-12, dt))))
    V0, m0, h0, n0 = [float(v) for v in x0_state]
    Cm, gNa, gK, gL, ENa, EK, EL, Iext = args
    params = (Cm, gNa, gK, gL, ENa, EK, EL)

    V = np.empty(n_steps, dtype=np.float64)
    m = np.empty(n_steps, dtype=np.float64)
    h = np.empty(n_steps, dtype=np.float64)
    n = np.empty(n_steps, dtype=np.float64)

    Vt, mt, ht, nt = V0, m0, h0, n0
    if method == "RK4":
        for i in range(n_steps):
            Vt, mt, ht, nt = step_rk4(Vt, mt, ht, nt, params, Iext, dt)
            V[i], m[i], h[i], n[i] = Vt, mt, ht, nt
    else:
        if NUMBA_OK:
            for i in range(n_steps):
                Vt, mt, ht, nt = step_euler_jit(Vt, mt, ht, nt, params, Iext, dt)
                V[i], m[i], h[i], n[i] = Vt, mt, ht, nt
        else:
            for i in range(n_steps):
                Vt, mt, ht, nt = step_euler(Vt, mt, ht, nt, params, Iext, dt)
                V[i], m[i], h[i], n[i] = Vt, mt, ht, nt

    t = np.arange(n_steps, dtype=np.float64) * dt
    return t, V, m, h, n

# ------------------------------------------------------------
# Detección de picos y ráfagas (igual patrón)
# ------------------------------------------------------------
def detect_spikes(x, perc=90.0):
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

def throttle(ms=150, key="hh_throttle_ms"):
    now = time.time()
    last = st.session_state.get(key, 0.0)
    if (now - last) * 1000.0 < ms:
        return True
    st.session_state[key] = now
    return False

STATE_KEYS = {
    "variant": "hh_variant",
    "params": "hh_params",
    "method": "hh_method",
    "last_sig": "hh_last_sig",
    "last_sim_t": "hh_last_sim_t",
    "data": "hh_data",
}

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.set_page_config(page_title="Hodgkin–Huxley — Simulación interactiva", layout="wide")
st.title("Hodgkin–Huxley — Simulación interactiva")

# Presets (parámetros clásicos HH 1952; 6.3°C)
PRESETS = {
    "HH clásico (reposo, I=0)": dict(
        Cm=1.0, gNa=120.0, gK=36.0, gL=0.3, ENa=50.0, EK=-77.0, EL=-54.4,
        Iext=0.0, V0=-65.0, dt=0.01, t_max=100.0
    ),
    "HH paso (I=10 µA/cm²)": dict(
        Cm=1.0, gNa=120.0, gK=36.0, gL=0.3, ENa=50.0, EK=-77.0, EL=-54.4,
        Iext=10.0, V0=-65.0, dt=0.01, t_max=100.0
    ),
    "HH fuerte (I=20 µA/cm²)": dict(
        Cm=1.0, gNa=120.0, gK=36.0, gL=0.3, ENa=50.0, EK=-77.0, EL=-54.4,
        Iext=20.0, V0=-65.0, dt=0.01, t_max=100.0
    ),
}

with st.sidebar:
    st.subheader("Configuración")

    # Variante y método (guardado para compatibilidad con hr.py)
    variant = "Hodgkin–Huxley (squid axon)"
    method = st.selectbox("Método numérico", ["Euler", "RK4"], index=0)
    st.session_state[STATE_KEYS["variant"]] = variant
    st.session_state[STATE_KEYS["method"]] = method

    preset_name = st.selectbox("Preset", list(PRESETS.keys()), index=1)
    p = PRESETS[preset_name]

    # Parámetros 3×3
    col1, col2, col3 = st.columns(3)
    with col1:
        Cm = st.number_input("Cm (µF/cm²)", value=float(p["Cm"]), step=0.1, format="%.2f")
        gNa = st.number_input("ḡNa (mS/cm²)", value=float(p["gNa"]), step=1.0, format="%.1f")
        gK  = st.number_input("ḡK  (mS/cm²)", value=float(p["gK"]),  step=1.0, format="%.1f")
    with col2:
        gL  = st.number_input("gL (mS/cm²)", value=float(p["gL"]), step=0.05, format="%.3f")
        ENa = st.number_input("E_Na (mV)", value=float(p["ENa"]), step=1.0, format="%.1f")
        EK  = st.number_input("E_K  (mV)", value=float(p["EK"]),  step=1.0, format="%.1f")
    with col3:
        EL   = st.number_input("E_L (mV)", value=float(p["EL"]), step=0.1, format="%.1f")
        Iext = st.number_input("I (µA/cm²)", value=float(p["Iext"]), step=1.0, format="%.2f")
        V0   = st.number_input("V0 (mV)", value=float(p["V0"]), step=1.0, format="%.1f")

    st.markdown("---")
    dt = st.number_input("dt (ms)", value=float(p["dt"]), step=0.005, format="%.3f")
    t_max = st.number_input("t_max (ms)", value=float(p["t_max"]), step=10.0, format="%.1f")

    show_phase_vn = st.checkbox("Retrato de fase V–n (OFF por defecto)", value=False)
    show_phase_vm = st.checkbox("Retrato de fase V–m (OFF por defecto)", value=False)
    mark_spikes = st.checkbox("Marcar picos (ON por defecto)", value=True)
    show_bursts = st.checkbox("Mostrar ráfagas (OFF por defecto)", value=False)
    downsample = st.checkbox("Activar downsampling", value=True)
    max_points = st.number_input("Puntos máx. a dibujar", value=50_000, step=1000)

    st.session_state[STATE_KEYS["params"]] = dict(
        Cm=Cm, gNa=gNa, gK=gK, gL=gL, ENa=ENa, EK=EK, EL=EL, Iext=Iext,
        V0=V0, dt=dt, t_max=t_max
    )

# ------------------------------------------------------------
# Throttle ~150 ms
# ------------------------------------------------------------
if throttle(150):
    st.stop()

# ------------------------------------------------------------
# Inicialización de compuertas en reposo (siempre coherentes con V0)
# ------------------------------------------------------------
def steady_gate(V):
    an, bn, am, bm, ah, bh = rates(V)
    n_inf = an / (an + bn)
    m_inf = am / (am + bm)
    h_inf = ah / (ah + bh)
    return m_inf, h_inf, n_inf

m0, h0, n0 = steady_gate(V0)

# ------------------------------------------------------------
# Simulación
# ------------------------------------------------------------
args = (Cm, gNa, gK, gL, ENa, EK, EL, Iext)
t, V, m, h, n = integrate(None, t_max=float(t_max), dt=float(dt),
                          x0_state=(V0, m0, h0, n0),
                          args=args, method=method, variant=variant)

# ------------------------------------------------------------
# Ventana temporal + downsampling
# ------------------------------------------------------------
base_key = f"{len(t)}_{int(t[-1])}"
t0, t1 = st.slider(
    "Ventana temporal",
    min_value=float(t[0]), max_value=float(t[-1]),
    value=(float(t[0]), float(t[-1])),
    step=float(dt), key=f"win_{base_key}"
)
sel = (t >= t0) & (t <= t1)
t_v, V_v, m_v, h_v, n_v = t[sel], V[sel], m[sel], h[sel], n[sel]

if downsample and len(t_v) > max_points:
    stride = max(1, len(t_v) // int(max_points))
    t_v, V_v, m_v, h_v, n_v = t_v[::stride], V_v[::stride], m_v[::stride], h_v[::stride], n_v[::stride]
else:
    stride = 1

# ------------------------------------------------------------
# Eventos (picos y ráfagas)
# ------------------------------------------------------------
spk_idx = detect_spikes(V_v) if mark_spikes else np.array([], dtype=int)
bursts = detect_bursts(V_v) if show_bursts else []

# ------------------------------------------------------------
# Gráfico principal V(t)
# ------------------------------------------------------------
fig = go.Figure()
fig.add_trace(go.Scattergl(x=t_v, y=V_v, mode="lines", name="V (mV)", line=dict(width=1)))
if mark_spikes and spk_idx.size > 0:
    fig.add_trace(go.Scattergl(x=t_v[spk_idx], y=V_v[spk_idx], mode="markers",
                               name="picos", marker=dict(size=5)))
if show_bursts and bursts:
    for b0, b1 in bursts:
        fig.add_vrect(x0=t_v[b0], x1=t_v[b1], fillcolor="white", opacity=0.08, line_width=0)

fig.update_traces(fill=None, selector=dict(type='scattergl'))
fig.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10),
                  height=420, title=f"Hodgkin–Huxley — {preset_name}")
fig.update_xaxes(title_text="t (ms)")
fig.update_yaxes(title_text="V (mV)")
st.plotly_chart(fig, use_container_width=True)

# Retratos de fase (opcionales)
if show_phase_vn:
    fig2 = go.Figure()
    fig2.add_trace(go.Scattergl(x=V_v, y=n_v, mode="lines", name="V–n", line=dict(width=1)))
    fig2.update_traces(fill=None, selector=dict(type='scattergl'))
    fig2.update_layout(template="plotly_dark", height=360,
                       margin=dict(l=10, r=10, t=30, b=10), title="Retrato de fase V–n")
    fig2.update_xaxes(title_text="V (mV)"); fig2.update_yaxes(title_text="n")
    st.plotly_chart(fig2, use_container_width=True)

if show_phase_vm:
    fig3 = go.Figure()
    fig3.add_trace(go.Scattergl(x=V_v, y=m_v, mode="lines", name="V–m", line=dict(width=1)))
    fig3.update_traces(fill=None, selector=dict(type='scattergl'))
    fig3.update_layout(template="plotly_dark", height=360,
                       margin=dict(l=10, r=10, t=30, b=10), title="Retrato de fase V–m")
    fig3.update_xaxes(title_text="V (mV)"); fig3.update_yaxes(title_text="m")
    st.plotly_chart(fig3, use_container_width=True)

# ------------------------------------------------------------
# Exportes de estado (compatibles con invariantes)
# ------------------------------------------------------------
st.session_state["hh_timeseries"] = dict(
    t=t, x=V, y=n, z=m, extra=dict(h=h),
    variant=variant, params=st.session_state[STATE_KEYS["params"]], method=method
)
st.session_state["timeseries"] = dict(
    t=t, x=V, y=n, z=m, model="hodgkin-huxley",
    params=st.session_state[STATE_KEYS["params"]]
)

st.session_state[STATE_KEYS["data"]] = dict(t_v=t_v, x_v=V_v, y_v=n_v, z_v=m_v, stride=stride)
st.session_state[STATE_KEYS["last_sig"]] = base_key
st.session_state[STATE_KEYS["last_sim_t"]] = time.time()
