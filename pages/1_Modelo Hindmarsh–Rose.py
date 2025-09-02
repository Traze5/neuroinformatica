# Hindmarsh每Rose: presets (normal, ca車tico, modificado) con simulaci車n en tiempo real
import time
import streamlit as st
import numpy as np
from dataclasses import dataclass
import plotly.graph_objects as go
from modules.ui import require_auth, sidebar_minimal

st.set_page_config(page_title="?? Modelo Hindmarsh每Rose", layout="wide")

# Autenticaci車n + sidebar minimal
require_auth(login_page="pages/00_Auto_Auth.py")
sidebar_minimal(
    "?? Modelo Hindmarsh每Rose",
    usuario=st.session_state.get("usuario"),
    msisdn=st.session_state.get("user_msisdn"),
    width_px=300
)

# Numba opcional
try:
    from numba import njit
    NUMBA = True
except Exception:
    NUMBA = False

# -------------------- utilidades --------------------
def _sig(d):
    return hash(tuple(sorted(d.items())))

@dataclass
class Bursts:
    onset_t: np.ndarray
    offset_t: np.ndarray

def detect_bursts(t, x, thr_hi=None, thr_lo=None, min_sep=0.05):
    t = np.asarray(t); x = np.asarray(x)
    if x.size < 4: return Bursts(np.array([]), np.array([]))
    if thr_hi is None: thr_hi = np.percentile(x, 75)
    if thr_lo is None: thr_lo = np.percentile(x, 55)
    above = x > thr_hi
    below = x < thr_lo
    onset_idx  = np.where((~above[:-1]) & (above[1:]))[0] + 1
    offset_idx = np.where((~below[:-1]) & (below[1:]))[0] + 1
    on, off = [], []; j = 0
    for i in onset_idx:
        while j < len(offset_idx) and offset_idx[j] <= i: j += 1
        if j < len(offset_idx) and (t[offset_idx[j]] - t[i] >= min_sep):
            on.append(t[i]); off.append(t[offset_idx[j]]); j += 1
    return Bursts(np.array(on), np.array(off))

def detect_spikes(t, x, thr=None):
    t = np.asarray(t); x = np.asarray(x)
    if x.size < 3: return np.array([]), np.array([])
    if thr is None: thr = np.percentile(x, 90.0)
    mid = (x[1:-1] > thr) & (x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])
    idx = np.where(mid)[0] + 1
    return t[idx], x[idx]

# -------------------- din芍mica HR --------------------
def f_hr(x, y, z, a, b, c, d, e, r, s, x0):
    dx = y - a*x**3 + b*x**2 - z + e
    dy = c - d*x**2 - y
    dz = r*(s*(x - x0) - z)
    return dx, dy, dz

def f_hr_mod(x, y, z, a, b, c, d, e, r, s, x0, nu):
    dx = y - a*x**3 + b*x**2 - z + e
    dy = c - d*x**2 - y
    dz = r*(-nu*z + s*(x - x0))
    return dx, dy, dz

# -------------------- integradores --------------------
if NUMBA:
    @njit
    def euler_hr_numba(a,b,c,d,e,r,s,x0, dt, n, x0x, x0y, x0z):
        t = np.empty(n+1); x = np.empty(n+1); y = np.empty(n+1); z = np.empty(n+1)
        t[0]=0.0; x[0]=x0x; y[0]=x0y; z[0]=x0z
        for i in range(n):
            dx = y[i] - a*x[i]**3 + b*x[i]**2 - z[i] + e
            dy = c - d*x[i]**2 - y[i]
            dz = r*(s*(x[i] - x0) - z[i])
            x[i+1] = x[i] + dt*dx
            y[i+1] = y[i] + dt*dy
            z[i+1] = z[i] + dt*dz
            t[i+1] = t[i] + dt
        return t,x,y,z

    @njit
    def euler_hrmod_numba(a,b,c,d,e,r,s,x0,nu, dt, n, x0x, x0y, x0z):
        t = np.empty(n+1); x = np.empty(n+1); y = np.empty(n+1); z = np.empty(n+1)
        t[0]=0.0; x[0]=x0x; y[0]=x0y; z[0]=x0z
        for i in range(n):
            dx = y[i] - a*x[i]**3 + b*x[i]**2 - z[i] + e
            dy = c - d*x[i]**2 - y[i]
            dz = r*(-nu*z[i] + s*(x[i] - x0))
            x[i+1] = x[i] + dt*dx
            y[i+1] = y[i] + dt*dy
            z[i+1] = z[i] + dt*dz
            t[i+1] = t[i] + dt
        return t,x,y,z

def integrate(fun, t_max, dt, x0_state, args=(), method="euler", variant="HR"):
    n = int(np.floor(t_max/dt))
    if method == "euler" and NUMBA:
        if variant == "HR modificado":
            return euler_hrmod_numba(args[0],args[1],args[2],args[3],args[4],
                                     args[5],args[6],args[7],args[8],
                                     dt, n, x0_state[0], x0_state[1], x0_state[2])
        else:
            return euler_hr_numba(args[0],args[1],args[2],args[3],args[4],
                                  args[5],args[6],args[7],
                                  dt, n, x0_state[0], x0_state[1], x0_state[2])

    t = np.linspace(0.0, n*dt, n+1)
    x = np.empty(n+1); y = np.empty(n+1); z = np.empty(n+1)
    x[0], y[0], z[0] = x0_state
    if method == "euler":
        for i in range(n):
            dx, dy, dz = fun(x[i], y[i], z[i], *args)
            x[i+1] = x[i] + dt*dx
            y[i+1] = y[i] + dt*dy
            z[i+1] = z[i] + dt*dz
    else:
        for i in range(n):
            k1x, k1y, k1z = fun(x[i], y[i], z[i], *args)
            k2x, k2y, k2z = fun(x[i]+0.5*dt*k1x, y[i]+0.5*dt*k1y, z[i]+0.5*dt*k1z, *args)
            k3x, k3y, k3z = fun(x[i]+0.5*dt*k2x, y[i]+0.5*dt*k2y, z[i]+0.5*dt*k2z, *args)
            k4x, k4y, k4z = fun(x[i]+dt*k3x,   y[i]+dt*k3y,   z[i]+dt*k3z,   *args)
            x[i+1] = x[i] + (dt/6.0)*(k1x + 2*k2x + 2*k3x + k4x)
            y[i+1] = y[i] + (dt/6.0)*(k1y + 2*k2y + 2*k3y + k4y)
            z[i+1] = z[i] + (dt/6.0)*(k1z + 2*k2z + 2*k3z + k4z)
    return t, x, y, z

# -------------------- presets --------------------
PRESETS = {
    "HR normal":     dict(a=1.0, b=3.0, c=1.0, d=5.0, e=3.0,   r=0.0021, s=4.0,  x0=-1.6, dt=0.01, t_max=15000.0),
    "HR ca車tico":    dict(a=1.0, b=3.0, c=1.0, d=5.0, e=2.5,   r=0.0021, s=4.1,  x0=-1.6, dt=0.01, t_max=15000.0),
    "HR modificado": dict(a=1.0, b=3.0, c=1.0, d=5.0, e=3.281, r=0.0021, s=1.0,  x0=-1.6, nu=0.10, dt=0.01, t_max=15000.0),
}

# -------------------- estado --------------------
if "hr_variant" not in st.session_state:
    st.session_state.hr_variant = "HR modificado"
if "hr_params" not in st.session_state:
    st.session_state.hr_params = PRESETS[st.session_state.hr_variant].copy()
if "hr_method" not in st.session_state:
    st.session_state.hr_method = "euler"
if "hr_last_sig" not in st.session_state:
    st.session_state.hr_last_sig = None
if "hr_last_sim_t" not in st.session_state:
    st.session_state.hr_last_sim_t = 0.0
if "hr_data" not in st.session_state:
    # simulaci車n inicial breve
    pm = PRESETS['HR modificado']
    args0 = (pm['a'], pm['b'], pm['c'], pm['d'], pm['e'], pm['r'], pm['s'], pm['x0'], pm['nu'])
    t0, x0, y0, z0 = integrate(
        f_hr_mod, t_max=3000.0, dt=pm['dt'],
        x0_state=(0.1,0.0,0.0), args=args0, method="euler", variant="HR modificado"
    )
    st.session_state.hr_data = dict(t=t0,x=x0,y=y0,z=z0)

# -------------------- controles --------------------
st.title("Hindmarsh每Rose ﹞ Presets")

variant = st.sidebar.selectbox(
    "Variante", list(PRESETS.keys()),
    index=list(PRESETS.keys()).index(st.session_state.hr_variant)
)
if variant != st.session_state.hr_variant:
    st.session_state.hr_variant = variant
    st.session_state.hr_params = PRESETS[variant].copy()

p = st.session_state.hr_params

with st.sidebar.expander("Par芍metros", expanded=True):
    c1,c2,c3 = st.columns(3)
    p["a"] = c1.number_input("a", 0.0, 10.0, p["a"], 0.1)
    p["b"] = c2.number_input("b", 0.0, 10.0, p["b"], 0.1)
    p["c"] = c3.number_input("c", 0.0, 10.0, p["c"], 0.1)
    c4,c5,c6 = st.columns(3)
    p["d"]  = c4.number_input("d", 0.0, 20.0, p["d"], 0.1)
    p["e"]  = c5.number_input("e", 0.0, 6.0, p["e"], 0.001, format="%.3f")
    p["r"]  = c6.number_input("米", 0.0, 0.02, p["r"], 0.0001, format="%.4f")
    c7,c8,c9 = st.columns(3)
    p["s"]  = c7.number_input("S", 0.0, 8.0, p["s"], 0.1)
    p["x0"] = c8.number_input("x?", -3.0, 3.0, p["x0"], 0.1)
    if variant == "HR modificado":
        p["nu"] = c9.number_input("糸", 0.0, 2.0, p.get("nu",0.10), 0.01)

with st.sidebar.expander("Tiempo e integraci車n", expanded=True):
    p["dt"]    = st.number_input("dt", 0.0005, 0.1, p["dt"], 0.0005, format="%.4f")
    p["t_max"] = st.number_input("t_max", 1000.0, 60000.0, p["t_max"], 500.0)
    method = st.selectbox("Integrador", ["euler", "rk4"],
                          index=0 if st.session_state.hr_method=="euler" else 1)
    st.session_state.hr_method = "euler" if method=="euler" else "rk4"
    max_points = st.slider("Puntos m芍x. a dibujar", 1000, 200000, 50000, 1000)
    downsample = st.checkbox("Submuestrear si excede puntos", True)

with st.sidebar.expander("Presentaci車n", expanded=True):
    show_y = st.checkbox("y(t)", False); show_z = st.checkbox("z(t)", False)
    show_phase_xy = st.checkbox("Retrato x每y", False)
    show_phase_xz = st.checkbox("Retrato x每z", False)
    mark_spikes = st.checkbox("Marcar picos", True)
    shade_bursts = st.checkbox("Resaltar r芍fagas", False)
    burst_alpha = st.slider("Opacidad r芍fagas", 0.0, 0.25, 0.06, 0.01)

# -------------------- simulaci車n: tiempo real con throttle --------------------
THROTTLE = 0.15  # s
now = time.time()
sig = _sig({**p, "variant": st.session_state.hr_variant, "method": st.session_state.hr_method})
need_sim = (sig != st.session_state.hr_last_sig) and ((now - st.session_state.hr_last_sim_t) > THROTTLE)

if need_sim:
    if st.session_state.hr_variant == "HR modificado":
        fun = f_hr_mod
        args = (p["a"],p["b"],p["c"],p["d"],p["e"],p["r"],p["s"],p["x0"],p["nu"])
        variant_key = "HR modificado"
    else:
        fun = f_hr
        args = (p["a"],p["b"],p["c"],p["d"],p["e"],p["r"],p["s"],p["x0"])
        variant_key = "HR normal" if st.session_state.hr_variant == "HR normal" else "HR ca車tico"

    t, x, y, z = integrate(
        fun, t_max=float(p["t_max"]), dt=float(p["dt"]),
        x0_state=(0.1,0.0,0.0), args=args,
        method=st.session_state.hr_method, variant=variant_key
    )
    st.session_state.hr_data = dict(t=t,x=x,y=y,z=z)
    st.session_state.hr_last_sig = sig
    st.session_state.hr_last_sim_t = now

# -------------------- visualizaci車n --------------------
t = st.session_state.hr_data["t"]; x = st.session_state.hr_data["x"]
y = st.session_state.hr_data["y"]; z = st.session_state.hr_data["z"]

stride = max(1, int(np.ceil(len(t)/max_points))) if (downsample and len(t)>max_points) else 1
t_d, x_d, y_d, z_d = t[::stride], x[::stride], y[::stride], z[::stride]

st.subheader("Se?ales")
step_f = float(p["dt"])*stride

# Ventana por defecto que NO empieza en 0 (p. ej., 迆ltimos 6000)
VENTANA_DEF = 6000.0
t1_def = float(t_d[-1])
t0_def = max(float(t_d[0]), t1_def - VENTANA_DEF)

t0, t1 = st.slider(
    "Ventana temporal",
    min_value=float(t_d[0]),
    max_value=float(t_d[-1]),
    value=(t0_def, t1_def),   # no arranca en 0
    step=step_f,
    key=f"win_{_sig(p)}"
)

sel = (t_d>=t0) & (t_d<=t1)
tt, xx, yy, zz = t_d[sel], x_d[sel], y_d[sel], z_d[sel]

fig = go.Figure()
fig.update_layout(template="plotly_dark", height=380, margin=dict(l=40,r=20,t=40,b=40),
                  title=f"{st.session_state.hr_variant} 〞 x(t)")
fig.add_trace(go.Scattergl(x=tt, y=xx, name="x(t)", mode="lines", line=dict(width=1.2)))
if show_y: fig.add_trace(go.Scattergl(x=tt, y=yy, name="y(t)", mode="lines", line=dict(width=1)))
if show_z: fig.add_trace(go.Scattergl(x=tt, y=zz, name="z(t)", mode="lines", line=dict(width=1)))

if mark_spikes:
    tspk, xspk = detect_spikes(tt, xx)
    if len(tspk) > 1200:
        sel_spk = np.linspace(0, len(tspk)-1, 1200).astype(int)
        tspk = tspk[sel_spk]; xspk = xspk[sel_spk]
    fig.add_trace(go.Scattergl(x=tspk, y=xspk, mode="markers",
                               marker=dict(size=5), name="spikes"))

if shade_bursts and (t1 - t0) <= 4000:
    bursts = detect_bursts(tt, xx)
    if len(bursts.onset_t) < 500:
        for t_on, t_off in zip(bursts.onset_t, bursts.offset_t):
            fig.add_vrect(x0=float(t_on), x1=float(t_off),
                          fillcolor=f"rgba(255,255,255,{burst_alpha})", line_width=0)

fig.update_xaxes(title="tiempo"); fig.update_yaxes(title="x")
st.plotly_chart(fig, use_container_width=True, theme=None)

if show_phase_xy or show_phase_xz:
    st.subheader("Retratos de fase")
    cols = st.columns(2)
    if show_phase_xy:
        fig_xy = go.Figure()
        fig_xy.update_layout(template="plotly_dark", height=360, margin=dict(l=40,r=20,t=30,b=40))
        fig_xy.add_trace(go.Scattergl(x=x_d, y=y_d, mode="lines", line=dict(width=1), name="x每y"))
        fig_xy.update_xaxes(title="x"); fig_xy.update_yaxes(title="y")
        cols[0].plotly_chart(fig_xy, use_container_width=True, theme=None)
    if show_phase_xz:
        fig_xz = go.Figure()
        fig_xz.update_layout(template="plotly_dark", height=360, margin=dict(l=40,r=20,t=30,b=40))
        fig_xz.add_trace(go.Scattergl(x=x_d, y=z_d, mode="lines", line=dict(width=1), name="x每z"))
        fig_xz.update_xaxes(title="x"); fig_xz.update_yaxes(title="z")
        (cols[1] if show_phase_xy else cols[0]).plotly_chart(fig_xz, use_container_width=True, theme=None)

# exporta a otros m車dulos (IDS)
st.session_state["hr_timeseries"] = dict(
    t=t, x=x, y=y, z=z,
    variant=st.session_state.hr_variant,
    params=st.session_state.hr_params,
    method=st.session_state.hr_method
)
