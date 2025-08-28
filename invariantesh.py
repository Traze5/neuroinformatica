# -*- coding: utf-8 -*-
"""
Hindmarsh–Rose (3D) — 2 neuronas: sinapsis química y eléctrica, análisis de picos, ráfagas e invariantes (IDS).
- Ecuaciones y parámetros exactamente como las fuentes originales.
- Panorámica (envolventes o líneas decimadas), Detalle (alta resolución).
- Raster, retratos de fase, histogramas, IDS y tablas.
- Integradores: LSODA (por defecto), RK4, RK6.
- Tema Plotly Claro/Oscuro corregido (Streamlit: theme='streamlit' o None).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
import streamlit as st
from plotly import graph_objects as go
from plotly.subplots import make_subplots

# --------- SciPy opcional ----------
try:
    from scipy.integrate import odeint as _odeint
    from scipy.signal import find_peaks
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False
    def find_peaks(x, height=None, distance=None):
        x = np.asarray(x)
        idx = np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))[0] + 1
        if height is not None:
            thr = height if np.isscalar(height) else height[0]
            idx = idx[x[idx] >= thr]
        if distance is not None and len(idx) > 1:
            sel = [idx[0]]
            for i in idx[1:]:
                if i - sel[-1] >= distance:
                    sel.append(i)
            idx = np.array(sel)
        return idx, {"peak_heights": x[idx]}

st.set_page_config(page_title="Hindmarsh–Rose · Invariantes", layout="wide")

# =============== Parámetros del modelo (idénticos a las fuentes) ===============
@dataclass
class HRParams:
    e: float = 3.282     # corriente
    u: float = 0.0021    # escala lenta
    s1: float = 1.0
    s2: float = 1.0
    v1: float = 0.1
    v2: float = 0.1
    Esyn: float = -1.8   # potencial reversa inhibidor (química)
    Vfast: float = -1.1  # umbral sigmoide (química)
    sfast: float = 0.2   # pendiente sigmoide (química)

# =================== RHS: sinapsis química y eléctrica (exactos) ===================
def rhs_quimica(state, t, p: HRParams):
    x1, y1, z1, x2, y2, z2 = state
    s2 = 1.0 / (1.0 + np.exp(p.sfast * (p.Vfast - x2)))  # gating por x_pre
    s1 = 1.0 / (1.0 + np.exp(p.sfast * (p.Vfast - x1)))
    dx1 = y1 + 3.0*x1*x1 - x1*x1*x1 - z1 + p.e - 0.1*(x1 - p.Esyn)*s2
    dy1 = 1.0 - 5.0*x1*x1 - y1
    dz1 = p.u * (-p.v1*z1 + p.s1*(x1 + 1.6))
    dx2 = y2 + 3.0*x2*x2 - x2*x2*x2 - z2 + p.e - 0.1*(x2 - p.Esyn)*s1
    dy2 = 1.0 - 5.0*x2*x2 - y2
    dz2 = p.u * (-p.v2*z2 + p.s2*(x2 + 1.6))
    return np.array([dx1, dy1, dz1, dx2, dy2, dz2], dtype=float)

def rhs_electrica(state, t, p: HRParams):
    x1, y1, z1, x2, y2, z2 = state
    dx1 = y1 + 3.0*x1*x1 - x1*x1*x1 - z1 + p.e + 0.05*(x1 - x2)
    dy1 = 1.0 - 5.0*x1*x1 - y1
    dz1 = p.u * (-p.v1*z1 + p.s1*(x1 + 1.6))
    dx2 = y2 + 3.0*x2*x2 - x2*x2*x2 - z2 + p.e + 0.05*(x2 - x1)
    dy2 = 1.0 - 5.0*x2*x2 - y2
    dz2 = p.u * (-p.v2*z2 + p.s2*(x2 + 1.6))
    return np.array([dx1, dy1, dz1, dx2, dy2, dz2], dtype=float)

# ========================= Integradores =========================
def rk4_step(y, t, dt, f, *args):
    k1 = f(y, t, *args)
    k2 = f(y + 0.5*dt*k1, t + 0.5*dt, *args)
    k3 = f(y + 0.5*dt*k2, t + 0.5*dt, *args)
    k4 = f(y + dt*k3, t + dt, *args)
    return y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def rk4(f, y0, t, *args):
    y = np.zeros((len(t), len(y0)), dtype=float)
    y[0] = y0
    for i in range(len(t)-1):
        dt = t[i+1] - t[i]
        y[i+1] = rk4_step(y[i], t[i], dt, f, *args)
    return y

def rk6(f, y0, t, *args):
    """RK6 con los coeficientes del integrador original."""
    y = np.zeros((len(t), len(y0)), dtype=float)
    y[0] = y0
    for i in range(len(t)-1):
        dt = t[i+1] - t[i]
        x = y[i]
        k0 = dt * f(x, t[i], *args)
        a  = x + 0.2*k0
        k1 = dt * f(a, t[i] + dt/5.0, *args)
        a  = x + 0.075*k0 + 0.225*k1
        k2 = dt * f(a, t[i] + 0.3*dt, *args)
        a  = x + 0.3*k0 - 0.9*k1 + 1.2*k2
        k3 = dt * f(a, t[i] + 0.6*dt, *args)
        a  = x + 0.075*k0 + 0.675*k1 - 0.6*k2 + 0.75*k3
        k4 = dt * f(a, t[i] + 0.9*dt, *args)
        a  = x + 0.660493827160493*k0 + 2.5*k1 - 5.185185185185185*k2 \
               + 3.888888888888889*k3 - 0.864197530864197*k4
        k5 = dt * f(a, t[i] + dt, *args)
        y[i+1] = x + (0.098765432098765)*k0 + (0.396825396825396)*k2 \
                   + (0.231481481481481)*k3 + (0.308641975308641)*k4 \
                   - (0.035714285714285)*k5
    return y

# ==================== Utilidades de análisis/visual ====================
def lttb_downsample(t, y, n_out=8000):
    """Decimación aproximada LTTB para dibujar líneas largas sin congelar la UI."""
    t = np.asarray(t); y = np.asarray(y)
    n = len(t)
    if n_out >= n or n_out < 3:
        return t, y
    bucket_size = (n - 2) / (n_out - 2)
    out_t = np.empty(n_out); out_y = np.empty(n_out)
    out_t[0] = t[0]; out_y[0] = y[0]
    a = 0
    for i in range(1, n_out - 1):
        start = int(np.floor((i - 1) * bucket_size)) + 1
        end   = int(np.floor(i * bucket_size)) + 1
        end   = min(end, n-1)
        t_bucket = t[start:end]; y_bucket = y[start:end]
        t_next_start = int(np.floor(i * bucket_size)) + 1
        t_next_end   = int(np.floor((i + 1) * bucket_size)) + 1
        t_next_end   = min(t_next_end, n)
        t_next = t[t_next_start:t_next_end]
        y_next = y[t_next_start:t_next_end]
        ya = y[a]; ta = t[a]
        area = np.abs((ta - t_next) * (y_bucket.mean() - ya) - (ya - y_next) * (t_bucket.mean() - ta))
        if area.size == 0:
            a = start
        else:
            a = start + int(np.argmax(area))
        out_t[i] = t[a]; out_y[i] = y[a]
    out_t[-1] = t[-1]; out_y[-1] = y[-1]
    return out_t, out_y

def detect_spikes(t, x, thr=0.20, min_gap=0.010):
    dt = float(np.median(np.diff(t)))
    min_samples = max(1, int(np.ceil(min_gap/dt)))
    pk, _ = find_peaks(x, height=thr, distance=min_samples)
    return pk.astype(int)

def detect_bursts(t, x, v_th=-0.60, min_on=0.10, min_off=0.05):
    above = x > v_th
    bursts = []
    i = 1
    while i < len(t):
        while i < len(t) and not (above[i] and not above[i-1]): i += 1
        if i >= len(t): break
        on = t[i]
        while i < len(t) and not ((not above[i]) and above[i-1]): i += 1
        if i >= len(t): break
        off = t[i]
        if (off - on) >= min_on and (not bursts or (on - bursts[-1][1]) >= min_off):
            bursts.append((on, off))
    return bursts

def pair_intervals(BA, BB):
    out = {"P": [], "BA": [], "BB": [], "D_AB": [], "D_BA": []}
    for i in range(len(BA)-1):
        a_on, a_off = BA[i]
        a_next_on   = BA[i+1][0]
        cand = [b for b in BB if b[0] >= a_on and b[0] < a_next_on]
        if not cand: continue
        b_on, b_off = cand[0]
        out["P"].append(a_next_on - a_on)
        out["BA"].append(a_off - a_on)
        out["BB"].append(b_off - b_on)
        out["D_AB"].append(b_on - a_off)
        out["D_BA"].append(a_next_on - b_off)
    return {k: np.array(v, float) for k, v in out.items() if len(v) > 0}

def linfit(x, y):
    if len(x) < 2: return np.nan, np.nan, np.nan
    X = np.vstack([x, np.ones_like(x)]).T
    m, q = np.linalg.lstsq(X, y, rcond=None)[0]
    yhat = m*x + q
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2) + 1e-12
    return float(m), float(q), float(1 - ss_res/ss_tot)

def minmax_envelope(t, x, n_bins=2500):
    n_bins = int(max(50, n_bins))
    idx = np.linspace(0, len(x), n_bins+1, dtype=int)
    tc  = 0.5*(t[idx[:-1]] + t[np.clip(idx[1:]-1, 0, len(t)-1)])
    xmin = np.array([x[a:b].min() for a, b in zip(idx[:-1], idx[1:])])
    xmax = np.array([x[a:b].max() for a, b in zip(idx[:-1], idx[1:])])
    return tc, xmin, xmax

# ===================== Simulación segmentada (cache) =====================
@st.cache_data(show_spinner=False)
def simulate_segment(mode: str, integrator: str, params: dict,
                     T0: float, T1: float, N: int, y0: np.ndarray):
    p = HRParams(**params)
    t = np.linspace(float(T0), float(T1), int(N))
    f = rhs_quimica if mode == "quimica" else rhs_electrica
    if integrator == "LSODA" and HAVE_SCIPY:
        sol = _odeint(lambda y,tt: f(y,tt,p), y0, t, atol=1e-6, rtol=1e-6)
    elif integrator == "RK6":
        sol = rk6(f, y0, t, p)
    else:
        sol = rk4(f, y0, t, p)
    mask = t >= 0.0
    return t[mask], sol[mask,:]

# =============================== UI ======================================
st.title("Hindmarsh–Rose (2 neuronas) — invariantes y gráficas")

colA, colB, colC = st.columns([1.2,1.2,1.0])
with colA:
    modo_lbl = st.selectbox("Sinapsis", ["Química (sigmoide)", "Eléctrica (difusiva)"], index=0)
    mode = "quimica" if modo_lbl.startswith("Química") else "electrica"
with colB:
    integ_lbl = st.radio("Integrador", ["LSODA (SciPy)","RK4 (NumPy)","RK6"], index=0 if HAVE_SCIPY else 1, horizontal=True)
    integrator = "LSODA" if (integ_lbl.startswith("LSODA") and HAVE_SCIPY) else ("RK6" if "RK6" in integ_lbl else "RK4")
with colC:
    theme_lbl = st.radio("Tema Plotly", ["Claro","Oscuro"], index=1, horizontal=True)
    THEME_ARG = "streamlit" if theme_lbl=="Claro" else None
    TEMPLATE  = "plotly_white" if theme_lbl=="Claro" else None

st.sidebar.header("Parámetros del modelo")
e     = st.sidebar.slider("e (corriente)", 2.8, 3.5, 3.282, 0.001)
u     = st.sidebar.slider("u (lento)",     0.0008, 0.0040, 0.0021, 0.0001, format="%.4f")
Esyn  = st.sidebar.slider("Esyn (química)", -3.0, 0.0, -1.8, 0.1)
Vfast = st.sidebar.slider("Vfast (química)", -2.0, 1.0, -1.1, 0.05)
sfast = st.sidebar.slider("sfast (química)", 0.05, 1.0, 0.20, 0.05)

st.sidebar.header("Tiempo y resolución")
T_over = st.sidebar.slider("T_over (s, panorámica)", 100.0, 15000.0, 3000.0, 50.0)
N_over = st.sidebar.slider("N_over (puntos)", 2001, 80001, 30001, 2000)
T_det  = st.sidebar.slider("T_detalle (s)", 20.0, 300.0, 80.0, 5.0)
N_det  = st.sidebar.slider("N_det (puntos, detalle)", 20001, 1_000_000, 1_000_000, 10_000)

st.sidebar.header("Detección")
auto_vth = st.sidebar.checkbox("Umbral de ráfaga automático", value=False)
v_th     = st.sidebar.slider("v_th (manual)", -3.0, 2.0, -0.60, 0.02)
pk_thr   = st.sidebar.slider("Umbral de picos (x)", -2.0, 2.0, 0.20, 0.01)
min_on   = st.sidebar.slider("Duración mínima ON (s)", 0.02, 0.40, 0.10, 0.01)
min_off  = st.sidebar.slider("Duración mínima OFF (s)",0.01, 0.30, 0.05, 0.01)
min_gap  = st.sidebar.slider("Separación mínima entre picos (s)", 0.004, 0.050, 0.010, 0.001, format="%.3f")

# Condiciones iniciales originales
y0 = np.array([-0.915325, -3.208968, 3.350784, -1.307949, -7.580493, 3.068898], float)
params = dict(e=e, u=u, Esyn=Esyn, Vfast=Vfast, sfast=sfast)

# ===================== PANORÁMICA =====================
T0_over = -min(1000.0, 0.2*T_over)
tt_over, sol_over = simulate_segment(mode, integrator, params, T0_over, T_over, N_over, y0)
x1_over, x2_over = sol_over[:,0], sol_over[:,3]
tc1, mn1, mx1 = minmax_envelope(tt_over, x1_over, n_bins=int(min(3000, N_over//10)))
tc2, mn2, mx2 = minmax_envelope(tt_over, x2_over, n_bins=int(min(3000, N_over//10)))

st.subheader("Vista panorámica")
vista_over = st.radio("Modo de dibujo", ["Envolvente (rápida)", "Líneas (decimadas)"], index=0, horizontal=True)

if vista_over.startswith("Envolvente"):
    fig_over = go.Figure()
    fig_over.add_trace(go.Scatter(x=tc1, y=mx1, mode='lines', line=dict(width=0.5), showlegend=False))
    fig_over.add_trace(go.Scatter(x=tc1, y=mn1, mode='lines', fill='tonexty', name='A envolvente', opacity=0.60))
    fig_over.add_trace(go.Scatter(x=tc1, y=0.5*(mn1+mx1), mode='lines', name='A contorno', line=dict(width=1)))
    fig_over.add_trace(go.Scatter(x=tc2, y=mx2, mode='lines', line=dict(width=0.5), showlegend=False))
    fig_over.add_trace(go.Scatter(x=tc2, y=mn2, mode='lines', fill='tonexty', name='B envolvente', opacity=0.40))
    fig_over.add_trace(go.Scatter(x=tc2, y=0.5*(mn2+mx2), mode='lines', name='B contorno', line=dict(width=1)))
else:
    tA, yA = lttb_downsample(tt_over, x1_over, n_out=8000)
    tB, yB = lttb_downsample(tt_over, x2_over, n_out=8000)
    fig_over = go.Figure()
    fig_over.add_trace(go.Scatter(x=tA, y=yA, mode='lines', name='x1'))
    fig_over.add_trace(go.Scatter(x=tB, y=yB, mode='lines', name='x2'))

fig_over.update_layout(title=f"Panorámica — {modo_lbl}", xaxis_title="tiempo (s)", yaxis_title="x",
                       height=320, margin=dict(l=10, r=10, b=10, t=50), template=TEMPLATE)
st.plotly_chart(fig_over, use_container_width=True, theme=THEME_ARG)

# ===================== DETALLE =====================
use_over_final_as_ic = st.checkbox("Usar estado final de la panorámica como IC del detalle", value=True)
y0_det = sol_over[-1, :].copy() if use_over_final_as_ic else y0

T0_det = 0.0  # al heredar el estado final ya no necesitamos warm-up negativo
tt, sol = simulate_segment(mode, integrator, params, T0_det, T_det, N_det, y0_det)
x1, y1, z1, x2, y2, z2 = sol.T

# Umbral de ráfaga efectivo
v_th_eff = float(np.quantile(x1, 0.60)) if auto_vth else v_th

# Detección picos/ráfagas
sp1 = detect_spikes(tt, x1, thr=pk_thr, min_gap=min_gap)
sp2 = detect_spikes(tt, x2, thr=pk_thr, min_gap=min_gap)
BA  = detect_bursts(tt, x1, v_th=v_th_eff, min_on=min_on, min_off=min_off)
BB  = detect_bursts(tt, x2, v_th=v_th_eff, min_on=min_on, min_off=min_off)

# Downsample visual
max_pts = 8000
step_vis = max(1, int(len(tt)/max_pts))
Tvis = tt[::step_vis]; X1vis = x1[::step_vis]; X2vis = x2[::step_vis]

# ===================== TABS =====================
tab_ts, tab_raster, tab_phase, tab_hist, tab_ids, tab_tables = st.tabs(
    ["Series temporales", "Raster de picos", "Fase (x–y)", "Histogramas", "IDS (invariantes)", "Tablas"]
)

with tab_ts:
    fig_det = go.Figure()
    fig_det.add_trace(go.Scatter(x=Tvis, y=X1vis, name='x1 (A)', mode='lines'))
    fig_det.add_trace(go.Scatter(x=Tvis, y=X2vis, name='x2 (B)', mode='lines'))
    for on, off in BA:
        fig_det.add_vline(x=on, line_dash='dash', opacity=0.35)
        fig_det.add_vline(x=off, line_dash='dash', opacity=0.35)
    for on, off in BB:
        fig_det.add_vline(x=on, line_dash='dot', opacity=0.35)
        fig_det.add_vline(x=off, line_dash='dot', opacity=0.35)
    if len(sp1): fig_det.add_trace(go.Scatter(x=tt[sp1], y=x1[sp1], mode='markers', marker=dict(size=5), name='picos A'))
    if len(sp2): fig_det.add_trace(go.Scatter(x=tt[sp2], y=x2[sp2], mode='markers', marker=dict(size=5), name='picos B'))
    fig_det.add_hline(y=v_th_eff, line_dash='dash', opacity=0.45)
    fig_det.update_layout(title=f"Detalle — {modo_lbl} | integ={integ_lbl}",
                          xaxis_title="tiempo (s)", yaxis_title="x",
                          height=420, margin=dict(l=10, r=10, b=10, t=60), template=TEMPLATE)
    st.plotly_chart(fig_det, use_container_width=True, theme=THEME_ARG)

with tab_raster:
    fig_r = go.Figure()
    fig_r.add_trace(go.Scatter(x=(tt[sp1] if len(sp1) else []), y=np.ones(len(sp1)), mode='markers', name='A'))
    fig_r.add_trace(go.Scatter(x=(tt[sp2] if len(sp2) else []), y=np.zeros(len(sp2)), mode='markers', name='B'))
    fig_r.update_layout(title="Raster de picos", xaxis_title="tiempo (s)",
                        yaxis=dict(tickmode='array', tickvals=[0,1], ticktext=["B","A"]),
                        height=260, margin=dict(l=10, r=10, b=10, t=60), template=TEMPLATE)
    st.plotly_chart(fig_r, use_container_width=True, theme=THEME_ARG)

with tab_phase:
    figp = make_subplots(rows=1, cols=2, subplot_titles=("N1 (x–y)", "N2 (x–y)"))
    figp.add_trace(go.Scatter(x=x1, y=y1, mode='lines', name='N1'), row=1, col=1)
    figp.add_trace(go.Scatter(x=x2, y=y2, mode='lines', name='N2'), row=1, col=2)
    figp.update_xaxes(title_text="x1", row=1, col=1); figp.update_yaxes(title_text="y1", row=1, col=1)
    figp.update_xaxes(title_text="x2", row=1, col=2); figp.update_yaxes(title_text="y2", row=1, col=2)
    figp.update_layout(height=360, title="Retratos de fase", showlegend=False,
                       margin=dict(l=10, r=10, b=10, t=60), template=TEMPLATE)
    st.plotly_chart(figp, use_container_width=True, theme=THEME_ARG)

with tab_hist:
    fig_h = make_subplots(rows=1, cols=2, subplot_titles=("ISI — N1", "ISI — N2"))
    if len(sp1) > 1:
        isi1 = np.diff(tt[sp1])
        fig_h.add_trace(go.Histogram(x=isi1, nbinsx=min(40, max(10, len(isi1)//2))), row=1, col=1)
    if len(sp2) > 1:
        isi2 = np.diff(tt[sp2])
        fig_h.add_trace(go.Histogram(x=isi2, nbinsx=min(40, max(10, len(isi2)//2))), row=1, col=2)
    fig_h.update_xaxes(title_text="ISI (s)", row=1, col=1); fig_h.update_yaxes(title_text="Frecuencia", row=1, col=1)
    fig_h.update_xaxes(title_text="ISI (s)", row=1, col=2); fig_h.update_yaxes(title_text="Frecuencia", row=1, col=2)
    fig_h.update_layout(height=360, title="Histogramas ISI", showlegend=False, bargap=0.05,
                        margin=dict(l=10, r=10, b=10, t=60), template=TEMPLATE)
    st.plotly_chart(fig_h, use_container_width=True, theme=THEME_ARG)

with tab_ids:
    Iv = pair_intervals(BA, BB)
    if not Iv:
        st.warning("No se emparejaron ciclos A→B→A. Ajusta e, u, la sinapsis o extiende T_over y usa su estado final como IC.")
    else:
        P, BAi, BBi, DAB, DBA = Iv["P"], Iv["BA"], Iv["BB"], Iv["D_AB"], Iv["D_BA"]
        def ttl(name, x, y):
            m,q,r2 = linfit(x,y); return f"{name} | m={m:.3f}, q={q:.3f}, R²={r2:.3f}", (m,q,r2)
        t1,s1 = ttl("BA vs P", P, BAi);  t2,s2 = ttl("BB vs P", P, BBi)
        t3,s3 = ttl("D_AB vs P", P, DAB); t4,s4 = ttl("D_BA vs P", P, DBA)
        t5,s5 = ttl("BA + D_AB vs P", P, BAi + DAB); t6,s6 = ttl("BB + D_BA vs P", P, BBi + DBA)

        fig_cmp = make_subplots(rows=2, cols=3, subplot_titles=(t1,t2,t3,t4,t5,t6))
        def scatter_fit(row,col,x,y,stats):
            m,q,_ = stats; xx = np.linspace(np.min(x), np.max(x), 100)
            fig_cmp.add_trace(go.Scatter(x=x,y=y,mode='markers'), row=row, col=col)
            fig_cmp.add_trace(go.Scatter(x=xx,y=m*xx+q,mode='lines',showlegend=False), row=row, col=col)
            fig_cmp.update_xaxes(title_text="P", row=row, col=col)
            fig_cmp.update_yaxes(title_text="valor", row=row, col=col)
        scatter_fit(1,1,P,BAi,s1); scatter_fit(1,2,P,BBi,s2); scatter_fit(1,3,P,DAB,s3)
        scatter_fit(2,1,P,DBA,s4); scatter_fit(2,2,P,BAi+DAB,s5); scatter_fit(2,3,P,BBi+DBA,s6)
        fig_cmp.update_layout(height=560, title="Comparativas IDS (dispersión + ajuste)",
                              showlegend=False, margin=dict(l=10, r=10, b=10, t=60), template=TEMPLATE)
        st.plotly_chart(fig_cmp, use_container_width=True, theme=THEME_ARG)

with tab_tables:
    c1,c2,c3 = st.columns([1,1,1])
    def bursts_df(B):
        if not B: return pd.DataFrame(columns=["burst_start","burst_end","dur"])
        arr = np.array(B)
        return pd.DataFrame({"burst_start":arr[:,0], "burst_end":arr[:,1], "dur":arr[:,1]-arr[:,0]})
    dfA = bursts_df(BA); dfB = bursts_df(BB)
    with c1:
        st.subheader("Intervalos N1")
        st.dataframe(dfA, use_container_width=True, height=280)
    with c2:
        st.subheader("Intervalos N2")
        st.dataframe(dfB, use_container_width=True, height=280)
    with c3:
        Iv = pair_intervals(BA, BB)
        st.subheader("Secuencias (A ancla)")
        if Iv:
            dfIv = pd.DataFrame({"P":Iv["P"], "BA":Iv["BA"], "BB":Iv["BB"], "D_AB":Iv["D_AB"], "D_BA":Iv["D_BA"]})
            st.dataframe(dfIv, use_container_width=True, height=280)
        else:
            st.info("Sin secuencias A→B→A detectadas.")

# Diagnóstico breve
st.info(f"Detección — picos A={len(sp1)}, B={len(sp2)} | ráfagas A={len(BA)}, B={len(BB)} | v_th usado={v_th_eff:.3f}")
st.success(f"Muestras detalle: {len(tt):,} | Integrador: {integ_lbl} | Downsample visual: {step_vis}×.")
