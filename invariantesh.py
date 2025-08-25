# -*- coding: utf-8 -*-
"""
HINDMARSH–ROSE (3D) — 2 neuronas (HCO) + INVARIANTES (IDS)
Panorámica (envolventes) + Detalle + Raster + Comparativas IDS
Integrador por defecto: SciPy odeint (LSODA) con alternativa RK4 (NumPy puro).

• Presets aplican y simulan automáticamente.
• Actualización en tiempo real al mover sliders.
• Gráficos con Plotly (interactivo) u opción Matplotlib (estático) restaurada.

Ejecutar:
    streamlit run streamlit_hr_hco_invariantes.py

Dependencias: streamlit, numpy, scipy (opcional), plotly, pandas, (matplotlib opcional)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass

import streamlit as st
from plotly import graph_objects as go
from plotly.subplots import make_subplots

# --- SciPy opcional (LSODA y find_peaks) ---
try:
    from scipy.integrate import odeint as _odeint
    from scipy.signal import find_peaks
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False
    def find_peaks(x, height=None, distance=None):
        x = np.asarray(x)
        # Fallback simple de picos (máximos locales)
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

st.set_page_config(page_title="HR (HCO) + IDS — Streamlit", layout="wide")

# -------------------- Modelo HR --------------------
@dataclass
class HRParams:
    e: float        # corriente (I)
    mu: float       # escala lenta (r)
    S: float = 4.0  # ganancia lenta (s)

@dataclass
class SynParams:
    g_syn: float = 0.35   # inhibición química
    g_el:  float = 0.00   # acoplo eléctrico
    theta: float = -0.25  # umbral sigmoide
    k:     float = 10.0   # pendiente sigmoide
    E_syn: float = -2.0   # potencial reversa inhibidor

def sigm(x, th, k):
    return 1.0/(1.0 + np.exp(-k*(x - th)))

def hr_pair_rhs(state12, t, p1: HRParams, p2: HRParams, syn: SynParams):
    x1,y1,z1, x2,y2,z2 = state12
    s1, s2 = sigm(x1, syn.theta, syn.k), sigm(x2, syn.theta, syn.k)
    I_syn1 = syn.g_syn * s2 * (syn.E_syn - x1);  I_syn2 = syn.g_syn * s1 * (syn.E_syn - x2)
    I_el1  = syn.g_el  * (x2 - x1);              I_el2  = syn.g_el  * (x1 - x2)
    dx1 = y1 + 3*x1**2 - x1**3 - z1 + p1.e + I_syn1 + I_el1
    dy1 = 1 - 5*x1**2 - y1
    dz1 = p1.mu * (-z1 + p1.S*(x1 + 1.6))
    dx2 = y2 + 3*x2**2 - x2**3 - z2 + p2.e + I_syn2 + I_el2
    dy2 = 1 - 5*x2**2 - y2
    dz2 = p2.mu * (-z2 + p2.S*(x2 + 1.6))
    return np.array([dx1,dy1,dz1, dx2,dy2,dz2], dtype=float)

# -------------------- Integración --------------------
def rk4(f, y0, t, *args):
    y = np.zeros((len(t), len(y0)), dtype=float)
    y[0] = y0
    for i in range(len(t)-1):
        dt = t[i+1]-t[i]
        k1 = f(y[i], t[i], *args)
        k2 = f(y[i] + 0.5*dt*k1, t[i] + 0.5*dt, *args)
        k3 = f(y[i] + 0.5*dt*k2, t[i] + 0.5*dt, *args)
        k4 = f(y[i] + dt*k3, t[i] + dt, *args)
        y[i+1] = y[i] + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return y

@st.cache_data(show_spinner=False)
def simulate_segment(e1, e2, mu, S, g_syn, g_el, theta, k, E_syn, T0, T1, N, y0, integrator="LSODA", have_scipy=True):
    """Devuelve (tt, sol) con tt≥0. Argumentos primitivos para cache estable."""
    p1 = HRParams(e=float(e1), mu=float(mu), S=float(S))
    p2 = HRParams(e=float(e2), mu=float(mu), S=float(S))
    syn = SynParams(g_syn=float(g_syn), g_el=float(g_el), theta=float(theta), k=float(k), E_syn=float(E_syn))
    t = np.linspace(float(T0), float(T1), int(N))
    if integrator == "LSODA" and have_scipy:
        sol = _odeint(lambda y,tt: hr_pair_rhs(y,tt,p1,p2,syn), y0, t, atol=1e-6, rtol=1e-6)
    else:
        sol = rk4(lambda y,tt,pp1,pp2,ss: hr_pair_rhs(y,tt,pp1,pp2,ss), y0, t, p1, p2, syn)
    mask = t >= 0.0
    tt   = t[mask] - 0.0  # mantener semántica original (re-base a 0)
    return tt, sol[mask,:]

# -------------------- Utilidades análisis --------------------
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
        while i < len(t) and not (above[i] and not above[i-1]):
            i += 1
        if i >= len(t):
            break
        on = t[i]
        while i < len(t) and not ((not above[i]) and above[i-1]):
            i += 1
        if i >= len(t):
            break
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
        if not cand:
            continue
        b_on, b_off = cand[0]
        out["P"].append(a_next_on - a_on)
        out["BA"].append(a_off - a_on)
        out["BB"].append(b_off - b_on)
        out["D_AB"].append(b_on - a_off)
        out["D_BA"].append(a_next_on - b_off)
    return {k: np.array(v, float) for k, v in out.items() if len(v) > 0}

def linfit(x, y):
    if len(x) < 2:
        return np.nan, np.nan, np.nan
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

# ===================== PRESETS =====================
PRESETS = {
    "HCO multiespike (moderado v2)": dict(e1=3.27,  e2=3.31,  mu=0.0018, g_syn=0.30, g_el=0.00),
    "HCO multiespike (moderado)":    dict(e1=3.26,  e2=3.30,  mu=0.0020, g_syn=0.35, g_el=0.00),
    "Dos HR sin acoplo (control)":   dict(e1=3.281, e2=3.281, mu=0.0021, g_syn=0.00, g_el=0.00),
}

# ===================== UI =====================
st.title("Hindmarsh–Rose (HCO) + Invariantes dinámicos (IDS)")

colA, colB = st.columns([1,1])
with colA:
    engine = st.radio(
        "Motor de gráficos",
        ["Plotly (interactivo)", "Matplotlib (estático)"],
        index=0, horizontal=True,
    )
with colB:
    integrator_choice = st.radio(
        "Integrador",
        ["LSODA" + (" (SciPy)" if HAVE_SCIPY else " (no disponible)"), "RK4 (NumPy)"],
        index=0 if HAVE_SCIPY else 1, horizontal=True,
    )
    integrator = "LSODA" if integrator_choice.startswith("LSODA") and HAVE_SCIPY else "RK4"

IS_PLOTLY = engine.startswith("Plotly")

st.sidebar.header("Presets y parámetros")

# Inicialización de estado por defecto
_defaults = dict(
    preset=list(PRESETS.keys())[0], e1=3.27, e2=3.31, mu=0.0018, gsyn=0.30, gel=0.00,
    th=-0.25, kk=10.0, Esyn=-2.0, Tover=3000, Nover=30001, Tdet=80, Ndet=120001,
    vth=-0.60, minon=0.10, minoff=0.05, mingap=0.010,
)
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def _apply_preset():
    cfg = PRESETS[st.session_state.preset]
    st.session_state.e1   = cfg["e1"]
    st.session_state.e2   = cfg["e2"]
    st.session_state.mu   = cfg["mu"]
    st.session_state.gsyn = cfg["g_syn"]
    st.session_state.gel  = cfg["g_el"]

st.sidebar.selectbox("Preset", options=list(PRESETS.keys()), key="preset", on_change=_apply_preset)

# Sliders principales
st.sidebar.subheader("Neuronas y acoplos")
st.sidebar.slider("I1 (A)", 2.8, 3.4, key="e1", step=0.005, format="%.3f")
st.sidebar.slider("I2 (B)", 2.8, 3.4, key="e2", step=0.005, format="%.3f")
st.sidebar.slider("r (μ)", 0.0008, 0.0030, key="mu", step=0.0001, format="%.4f")
st.sidebar.slider("g_syn", 0.0, 1.0, key="gsyn", step=0.01)
st.sidebar.slider("g_el", 0.0, 0.5, key="gel", step=0.01)

st.sidebar.subheader("Sinapsis química")
st.sidebar.slider("θ", -1.0, 0.5, key="th", step=0.01)
st.sidebar.slider("k", 1.0, 15.0, key="kk", step=0.5)
st.sidebar.slider("E_syn", -3.0, 0.0, key="Esyn", step=0.1)

st.sidebar.subheader("Tiempos: panorámica y detalle")
st.sidebar.slider("T_over (pan)", 300, 15000, key="Tover", step=100)
st.sidebar.slider("N_over", 5001, 80001, key="Nover", step=2000)
st.sidebar.slider("T_detalle", 20, 300, key="Tdet", step=5)
st.sidebar.slider("N_det", 20001, 200001, key="Ndet", step=10000)

st.sidebar.subheader("Detección (spikes / bursts)")
st.sidebar.slider("v_th", -1.0, -0.2, key="vth", step=0.01)
st.sidebar.slider("min ON", 0.02, 0.40, key="minon", step=0.01)
st.sidebar.slider("min OFF", 0.01, 0.30, key="minoff", step=0.01)
st.sidebar.slider("min gap spike", 0.004, 0.030, key="mingap", step=0.001, format="%.3f")

# ===================== Simulación =====================
# 1) PANORÁMICA (primero)
T0_over = -min(1000.0, 0.2*st.session_state.Tover)
y0 = np.array([-1.2, -10, 1.8, -1.0, -9.5, 1.7], float)

tt_over, sol_over = simulate_segment(
    st.session_state.e1, st.session_state.e2, st.session_state.mu, 4.0,
    st.session_state.gsyn, st.session_state.gel, st.session_state.th,
    st.session_state.kk, st.session_state.Esyn,
    T0_over, float(st.session_state.Tover), int(st.session_state.Nover), y0,
    integrator=integrator, have_scipy=HAVE_SCIPY,
)

x1_over, x2_over = sol_over[:, 0], sol_over[:, 3]
nbins = int(min(3000, st.session_state.Nover//10))
tc1, mn1, mx1 = minmax_envelope(tt_over, x1_over, n_bins=nbins)
tc2, mn2, mx2 = minmax_envelope(tt_over, x2_over, n_bins=nbins)

if IS_PLOTLY:
    fig_over = go.Figure()
    # A (envolvente, menos transparente que B)
    fig_over.add_trace(go.Scatter(x=tc1, y=mx1, mode='lines', line=dict(width=0.5), name='A max', showlegend=False))
    fig_over.add_trace(go.Scatter(x=tc1, y=mn1, mode='lines', fill='tonexty', name='A (envolvente)', opacity=0.60))
    fig_over.add_trace(go.Scatter(x=tc1, y=0.5*(mn1+mx1), mode='lines', name='A contorno', line=dict(width=1)))
    # B (envolvente)
    fig_over.add_trace(go.Scatter(x=tc2, y=mx2, mode='lines', line=dict(width=0.5), name='B max', showlegend=False))
    fig_over.add_trace(go.Scatter(x=tc2, y=mn2, mode='lines', fill='tonexty', name='B (envolvente)', opacity=0.40))
    fig_over.add_trace(go.Scatter(x=tc2, y=0.5*(mn2+mx2), mode='lines', name='B contorno', line=dict(width=1)))
    fig_over.update_layout(title="Vista panorámica (envolventes min–max)", xaxis_title="tiempo", yaxis_title="x",
                           height=320, margin=dict(l=10, r=10, b=10, t=40))
    st.plotly_chart(fig_over, use_container_width=True, theme=None)
else:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.fill_between(tc1, mn1, mx1, alpha=0.60, step='mid', label="A (envolvente)")
    ax.plot(tc1, 0.5*(mn1+mx1), lw=0.9, alpha=0.9)
    ax.fill_between(tc2, mn2, mx2, alpha=0.40, step='mid', label="B (envolvente)")
    ax.plot(tc2, 0.5*(mn2+mx2), lw=0.9, alpha=0.9)
    ax.set(title="Vista panorámica (envolventes min–max)", xlabel="tiempo", ylabel="x")
    ax.legend(); fig.tight_layout()
    st.pyplot(fig)

# 2) DETALLE (para spikes/invariantes)
T0_det = -min(300.0, 0.2*st.session_state.Tdet)

tt, sol = simulate_segment(
    st.session_state.e1, st.session_state.e2, st.session_state.mu, 4.0,
    st.session_state.gsyn, st.session_state.gel, st.session_state.th,
    st.session_state.kk, st.session_state.Esyn,
    T0_det, float(st.session_state.Tdet), int(st.session_state.Ndet), y0,
    integrator=integrator, have_scipy=HAVE_SCIPY,
)

x1, y1, z1, x2, y2, z2 = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3], sol[:, 4], sol[:, 5]
sp1 = detect_spikes(tt, x1, thr=0.20, min_gap=st.session_state.mingap)
sp2 = detect_spikes(tt, x2, thr=0.20, min_gap=st.session_state.mingap)
BA  = detect_bursts(tt, x1, v_th=st.session_state.vth, min_on=st.session_state.minon, min_off=st.session_state.minoff)
BB  = detect_bursts(tt, x2, v_th=st.session_state.vth, min_on=st.session_state.minon, min_off=st.session_state.minoff)

# Downsample visual para rendimiento
max_pts = 8000
step_vis = max(1, int(len(tt)/max_pts))
Tvis = tt[::step_vis]; X1vis = x1[::step_vis]; X2vis = x2[::step_vis]

if IS_PLOTLY:
    fig_det = go.Figure()
    fig_det.add_trace(go.Scatter(x=Tvis, y=X1vis, name='x1 (A)', mode='lines'))
    fig_det.add_trace(go.Scatter(x=Tvis, y=X2vis, name='x2 (B)', mode='lines'))
    for on, off in BA:
        fig_det.add_vline(x=on, line_dash='dash', opacity=0.35)
        fig_det.add_vline(x=off, line_dash='dash', opacity=0.35)
    for on, off in BB:
        fig_det.add_vline(x=on, line_dash='dot', opacity=0.35)
        fig_det.add_vline(x=off, line_dash='dot', opacity=0.35)
    if len(sp1):
        fig_det.add_trace(go.Scatter(x=tt[sp1], y=x1[sp1], mode='markers', marker=dict(size=5), name='picos A'))
    if len(sp2):
        fig_det.add_trace(go.Scatter(x=tt[sp2], y=x2[sp2], mode='markers', marker=dict(size=5), name='picos B'))
    fig_det.add_hline(y=st.session_state.vth, line_dash='dash', opacity=0.45)
    fig_det.update_layout(title="Detalle — señal con picos y límites de ráfaga", xaxis_title="tiempo", yaxis_title="x",
                          height=420, margin=dict(l=10, r=10, b=10, t=40))
    st.plotly_chart(fig_det, use_container_width=True, theme=None)
else:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(tt, x1, label='x1 (A)')
    ax.plot(tt, x2, label='x2 (B)')
    for on, off in BA:
        ax.axvline(on, ls='--', alpha=0.35)
        ax.axvline(off, ls='--', alpha=0.35)
    for on, off in BB:
        ax.axvline(on, ls=':', alpha=0.35)
        ax.axvline(off, ls=':', alpha=0.35)
    if len(sp1):
        ax.scatter(tt[sp1], x1[sp1], s=14, zorder=3)
    if len(sp2):
        ax.scatter(tt[sp2], x2[sp2], s=14, zorder=3)
    ax.axhline(st.session_state.vth, ls='--', alpha=0.45, color='k')
    ax.set(title="Detalle — señal con picos y límites de ráfaga", xlabel="tiempo", ylabel="x")
    ax.legend(); fig.tight_layout(); st.pyplot(fig)

# 3) Raster de spikes
if IS_PLOTLY:
    fig_raster = go.Figure()
    fig_raster.add_trace(go.Scatter(x=(tt[sp1] if len(sp1) else []), y=np.ones(len(sp1)), mode='markers', name='A'))
    fig_raster.add_trace(go.Scatter(x=(tt[sp2] if len(sp2) else []), y=np.zeros(len(sp2)), mode='markers', name='B'))
    fig_raster.update_layout(title="Raster de spikes", xaxis_title="tiempo",
                             yaxis=dict(tickmode='array', tickvals=[0,1], ticktext=["B","A"]),
                             height=260, margin=dict(l=10, r=10, b=10, t=40))
    st.plotly_chart(fig_raster, use_container_width=True, theme=None)
else:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 2.6))
    if len(sp1):
        ax.scatter(tt[sp1], np.ones(len(sp1)), s=8)
    if len(sp2):
        ax.scatter(tt[sp2], np.zeros(len(sp2)), s=8)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["B (spikes)", "A (spikes)"])
    ax.set(xlabel="tiempo", title="Raster de spikes"); fig.tight_layout(); st.pyplot(fig)

# 4) Invariantes + comparativas IDS
Iv = pair_intervals(BA, BB)
if not Iv:
    st.warning("No se emparejaron ciclos A→B→A. Ajusta g_syn (0.25–0.40), r (0.0016–0.0022) o asegúrate de I2>I1.")
else:
    P, BAi, BBi, DAB, DBA = Iv["P"], Iv["BA"], Iv["BB"], Iv["D_AB"], Iv["D_BA"]

    # Estadísticos y títulos con m,q,R^2 (ASCII) para evitar anotaciones por subgráfico
    def _mk_title(name, x, y):
        m,q,r2 = linfit(x, y)
        return f"{name} | m={m:.3f}, q={q:.3f}, R^2={r2:.3f}", (m,q,r2)

    t1, s1 = _mk_title("BA_vs_P", P, BAi)
    t2, s2 = _mk_title("BB_vs_P", P, BBi)
    t3, s3 = _mk_title("D_AB_vs_P", P, DAB)
    t4, s4 = _mk_title("D_BA_vs_P", P, DBA)
    t5, s5 = _mk_title("BA_plus_D_AB_vs_P", P, BAi + DAB)
    t6, s6 = _mk_title("BB_plus_D_BA_vs_P", P, BBi + DBA)

    if IS_PLOTLY:
        fig_cmp = make_subplots(rows=2, cols=3, subplot_titles=(t1, t2, t3, t4, t5, t6))
        def _scatter_fit(fig, row, col, x, y, stats):
            m,q,r2 = stats
            xx = np.linspace(np.min(x), np.max(x), 100)
            fig.add_trace(go.Scatter(x=x,  y=y,        mode='markers', name=f"{row},{col}"), row=row, col=col)
            fig.add_trace(go.Scatter(x=xx, y=m*xx+q,   mode='lines',   showlegend=False), row=row, col=col)
            fig.update_xaxes(title_text="P", row=row, col=col)
            fig.update_yaxes(title_text="valor", row=row, col=col)
        _scatter_fit(fig_cmp, 1,1, P, BAi,       s1)
        _scatter_fit(fig_cmp, 1,2, P, BBi,       s2)
        _scatter_fit(fig_cmp, 1,3, P, DAB,       s3)
        _scatter_fit(fig_cmp, 2,1, P, DBA,       s4)
        _scatter_fit(fig_cmp, 2,2, P, BAi + DAB, s5)
        _scatter_fit(fig_cmp, 2,3, P, BBi + DBA, s6)
        fig_cmp.update_layout(height=560, title_text="Comparativas de invariantes (dispersión + ajuste)",
                              showlegend=False, margin=dict(l=10, r=10, b=10, t=60))
        st.plotly_chart(fig_cmp, use_container_width=True, theme=None)
    else:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 3, figsize=(13, 6))
        def _fit_ax(ax, x, y, ttl, stats):
            m,q,r2 = stats
            xx = np.linspace(np.min(x), np.max(x), 100)
            ax.scatter(x, y, s=18, alpha=0.9)
            ax.plot(xx, m*xx+q, lw=1.2)
            ax.grid(alpha=0.25)
            ax.set_title(f"{ttl} | m={m:.3f}, q={q:.3f}, R^2={r2:.3f}")
            ax.set_xlabel("P"); ax.set_ylabel("valor")
        _fit_ax(axs[0,0], P, BAi,          "BA_vs_P",          s1)
        _fit_ax(axs[0,1], P, BBi,          "BB_vs_P",          s2)
        _fit_ax(axs[0,2], P, DAB,          "D_AB_vs_P",        s3)
        _fit_ax(axs[1,0], P, DBA,          "D_BA_vs_P",        s4)
        _fit_ax(axs[1,1], P, BAi + DAB,    "BA_plus_D_AB_vs_P",s5)
        _fit_ax(axs[1,2], P, BBi + DBA,    "BB_plus_D_BA_vs_P",s6)
        fig.suptitle("Comparativas de invariantes (dispersión + ajuste)", y=1.02)
        fig.tight_layout(); st.pyplot(fig)

    # Resumen estadístico
    stats = {k: (float(np.mean(v)), float(np.std(v)), int(len(v))) for k, v in Iv.items()}
    df = pd.DataFrame({k: {'media': m, 'sd': s, 'n': n} for k, (m, s, n) in stats.items()}).T
    st.subheader("Resumen estadístico (s)")
    st.dataframe(df, use_container_width=True)
    st.download_button("Descargar CSV (IDS)", df.to_csv().encode('utf-8'), file_name="ids_resumen.csv", mime="text/csv")

# Nota de rendimiento + sanity checks
with st.expander("Notas de rendimiento y visualización"):
    st.markdown(
        f"""- Los gráficos se actualizan automáticamente con cada cambio.
- Para señales muy densas, se aplica un *downsample* visual (no afecta a la detección ni a IDS).
- Integrador actual: **{integrator}**. Puedes alternar a RK4 si SciPy no está disponible.
- Plotly permite zoom/pan de forma fluida; Matplotlib genera PNG estáticos."""
    )

# ====== Sanity checks (opcionales) ======
with st.sidebar.expander("Sanity checks (opcional)"):
    run_tests = st.checkbox("Ejecutar pruebas rápidas")
    if run_tests:
        # Test 1: linfit en una relación lineal exacta
        x_t = np.linspace(0, 9, 10)
        y_t = 2.0*x_t + 1.0
        m_t, q_t, r2_t = linfit(x_t, y_t)
        st.write({"linfit_m": m_t, "linfit_q": q_t, "linfit_r2": r2_t})
        assert abs(m_t-2.0) < 1e-9 and abs(q_t-1.0) < 1e-9 and r2_t > 0.999999999, "linfit fallo en caso lineal"

        # Test 2: pair_intervals con intervalos simples (1 emparejamiento mínimo)
        BA_t = [(0.0, 0.5), (1.0, 1.4), (2.0, 2.6)]
        BB_t = [(0.6, 0.9), (1.5, 1.8)]
        Iv_t = pair_intervals(BA_t, BB_t)
        st.write(Iv_t)
        assert "P" in Iv_t and len(Iv_t["P"]) >= 1, "pair_intervals no devolvió periodos"

        # Test 3: detect_spikes sobre señal sintética con 3 picos separados
        t_s = np.linspace(0, 1, 1001)
        x_s = np.sin(2*np.pi*5*t_s) * 0.0
        x_s[100] = 1.0; x_s[500] = 1.0; x_s[900] = 1.0
        pk = detect_spikes(t_s, x_s, thr=0.5, min_gap=0.05)
        st.write({"spikes_detectados": len(pk), "pos": pk.tolist()[:5]})
        assert len(pk) == 3, "detect_spikes no detectó 3 picos"

        # Test 4: detect_bursts con umbral simple (dos ráfagas separadas)
        x_b = np.zeros_like(t_s)
        x_b[(t_s>=0.10) & (t_s<=0.25)] = 1.0
        x_b[(t_s>=0.60) & (t_s<=0.80)] = 1.0
        bursts = detect_bursts(t_s, x_b, v_th=0.5, min_on=0.05, min_off=0.05)
        st.write({"bursts": bursts})
        assert len(bursts) == 2, "detect_bursts no detectó 2 ráfagas"

        st.success("Pruebas rápidas OK")
