# -*- coding: utf-8 -*-
"""
Hindmarsh–Rose (3D) — 2 neuronas
Sinapsis: Química (inhibidora sigmoidal; según C original) y Eléctrica (difusiva).
Integradores: LSODA (SciPy), RK4 y RK6 (coeficientes del integrador original).
Análisis: picos, ráfagas e invariantes (IDS) con métricas (m, q, R², RMSE, medias, SD, CV y fracciones normalizadas).

Dependencias:
    streamlit, numpy, plotly, pandas
    scipy (opcional, recomendado para LSODA y find_peaks)

Ejecutar:
    streamlit run invariantesh.py
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass

import streamlit as st
from plotly import graph_objects as go
from plotly.subplots import make_subplots

# ========================= SciPy opcional =========================
try:
    from scipy.integrate import odeint as _odeint
    from scipy.signal import find_peaks
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False
    def find_peaks(x, height=None, distance=None):
        """Fallback simple de picos (máximos locales) si no hay SciPy."""
        x = np.asarray(x, float)
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

# ===================== Parámetros del modelo =====================
@dataclass
class HRParams:
    e: float = 3.282     # corriente
    u: float = 0.0021    # escala lenta
    s1: float = 1.0
    s2: float = 1.0
    v1: float = 0.1
    v2: float = 0.1
    # Química
    Esyn: float = -1.8   # potencial reversa inhibidor
    Vfast: float = -1.1  # umbral sigmoide
    sfast: float = 0.2   # pendiente sigmoide

# ============================ RHS ============================
def rhs_quimica(state, t, p: HRParams):
    x1, y1, z1, x2, y2, z2 = state
    # gating sigmoidal exactamente como en el C: 1/(1+exp(sfast*(Vfast - x_pre)))
    s2 = 1.0 / (1.0 + np.exp(p.sfast * (p.Vfast - x2)))
    s1 = 1.0 / (1.0 + np.exp(p.sfast * (p.Vfast - x1)))
    dx1 = y1 + 3.0*x1*x1 - x1*x1*x1 - z1 + p.e - 0.1*(x1 - p.Esyn)*s2
    dy1 = 1.0 - 5.0*x1*x1 - y1
    dz1 = p.u * (-p.v1*z1 + p.s1*(x1 + 1.6))
    dx2 = y2 + 3.0*x2*x2 - x2*x2*x2 - z2 + p.e - 0.1*(x2 - p.Esyn)*s1
    dy2 = 1.0 - 5.0*x2*x2 - y2
    dz2 = p.u * (-p.v2*z2 + p.s2*(x2 + 1.6))
    return np.array([dx1, dy1, dz1, dx2, dy2, dz2], float)

def rhs_electrica(state, t, p: HRParams):
    x1, y1, z1, x2, y2, z2 = state
    dx1 = y1 + 3.0*x1*x1 - x1*x1*x1 - z1 + p.e + 0.05*(x1 - x2)
    dy1 = 1.0 - 5.0*x1*x1 - y1
    dz1 = p.u * (-p.v1*z1 + p.s1*(x1 + 1.6))
    dx2 = y2 + 3.0*x2*x2 - x2*x2*x2 - z2 + p.e + 0.05*(x2 - x1)
    dy2 = 1.0 - 5.0*x2*x2 - y2
    dz2 = p.u * (-p.v2*z2 + p.s2*(x2 + 1.6))
    return np.array([dx1, dy1, dz1, dx2, dy2, dz2], float)

# ========================= Integradores =========================
def rk4_step(y, t, dt, f, *args):
    k1 = f(y, t, *args)
    k2 = f(y + 0.5*dt*k1, t + 0.5*dt, *args)
    k3 = f(y + 0.5*dt*k2, t + 0.5*dt, *args)
    k4 = f(y + dt*k3,     t + dt,     *args)
    return y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def rk4(f, y0, t, *args):
    y = np.zeros((len(t), len(y0)), float); y[0] = y0
    for i in range(len(t)-1):
        y[i+1] = rk4_step(y[i], t[i], t[i+1]-t[i], f, *args)
    return y

def rk6(f, y0, t, *args):
    """RK6 con coeficientes del integrador original (C)."""
    y = np.zeros((len(t), len(y0)), float); y[0] = y0
    for i in range(len(t)-1):
        dt = t[i+1]-t[i]; x = y[i]
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
    """Decimación aproximada para trazados largos (LTTB)."""
    t = np.asarray(t); y = np.asarray(y)
    n = len(t)
    if n_out >= n or n_out < 3:
        return t, y
    bucket = (n - 2) / (n_out - 2)
    out_t = np.empty(n_out); out_y = np.empty(n_out)
    out_t[0] = t[0]; out_y[0] = y[0]; a = 0
    for i in range(1, n_out - 1):
        s = int(np.floor((i - 1) * bucket)) + 1
        e = int(np.floor(i * bucket)) + 1; e = min(e, n-1)
        s2 = int(np.floor(i * bucket)) + 1
        e2 = int(np.floor((i + 1) * bucket)) + 1; e2 = min(e2, n)
        ta = t[a]; ya = y[a]
        # área del triángulo
        area = np.abs((ta - t[s2:e2]) * (y[s:e].mean() - ya) - (ya - y[s2:e2]) * (t[s:e].mean() - ta))
        a = s if area.size == 0 else s + int(np.argmax(area))
        out_t[i] = t[a]; out_y[i] = y[a]
    out_t[-1] = t[-1]; out_y[-1] = y[-1]
    return out_t, out_y

def detect_spikes(t, x, thr=0.20, min_gap=0.010):
    """Detección de picos (umbral en x)."""
    dt = float(np.median(np.diff(t)))
    min_samples = max(1, int(np.ceil(min_gap/dt)))
    pk, _ = find_peaks(np.asarray(x), height=thr, distance=min_samples)
    return pk.astype(int)

def detect_bursts(t, x, v_th=-0.60, min_on=0.10, min_off=0.05):
    """Ráfagas por cruce sobre v_th con duraciones mínimas."""
    above = np.asarray(x) > v_th
    bursts = []; i = 1
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

# ---- Fallback robusto por picos/ISI (cuando el cruce falla) ----
def auto_isi_thresholds(t, peaks):
    """Umbrales automáticos para agrupar por ISI (en segundos)."""
    if len(peaks) < 3:
        return 0.06, 0.12
    isi = np.diff(t[peaks])
    base = np.percentile(isi, 30)  # ISI intra-ráfaga
    isi_on  = 1.5 * float(base)    # seguir en la misma ráfaga
    ibi_off = 3.0 * float(base)    # cortar ráfaga
    return isi_on, ibi_off

def bursts_from_peaks(t, peaks, isi_on, ibi_off, min_on=0.05):
    """Agrupa picos en ráfagas usando ISI. Devuelve [(on, off), ...]."""
    if len(peaks) < 2:
        return []
    tt = t[peaks]; isi = np.diff(tt)
    bursts = []; start_idx = 0
    for k in range(len(isi)):
        if isi[k] > ibi_off:
            on, off = tt[start_idx], tt[k]
            if (off - on) >= min_on:
                bursts.append((on, off))
            start_idx = k + 1
    on, off = tt[start_idx], tt[-1]
    if (off - on) >= min_on:
        bursts.append((on, off))
    return bursts

# ---- IDS ----
def pair_intervals_anchor(A, B):
    """Empareja ciclos con anclaje en A respecto a B: P, BA, BB, D_AB, D_BA."""
    out = {"P": [], "BA": [], "BB": [], "D_AB": [], "D_BA": []}
    for i in range(len(A)-1):
        a_on, a_off = A[i]
        a_next_on   = A[i+1][0]
        cand = [b for b in B if b[0] >= a_on and b[0] < a_next_on]
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
    """Ajuste lineal y→m·x+q con R²."""
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    X = np.vstack([x, np.ones_like(x)]).T
    m, q = np.linalg.lstsq(X, y, rcond=None)[0]
    yhat = m*x + q
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2) + 1e-12
    return float(m), float(q), float(1 - ss_res/ss_tot)

def ids_metrics(P, BAi, BBi, DAB, DBA):
    """Métricas de IDS: ajustes, básicos y fracciones normalizadas."""
    def _fit(x, y):
        if len(x) < 2:
            return dict(m=np.nan, q=np.nan, r2=np.nan, rmse=np.nan)
        X = np.vstack([x, np.ones_like(x)]).T
        m, q = np.linalg.lstsq(X, y, rcond=None)[0]
        yhat = m*x + q
        res  = y - yhat
        ss_tot = np.sum((y - np.mean(y))**2) + 1e-12
        r2  = 1.0 - np.sum(res**2)/ss_tot
        rmse = np.sqrt(np.mean(res**2))
        return dict(m=float(m), q=float(q), r2=float(r2), rmse=float(rmse))

    fits = {
        "BA vs P":      _fit(P, BAi),
        "BB vs P":      _fit(P, BBi),
        "D_AB vs P":    _fit(P, DAB),
        "D_BA vs P":    _fit(P, DBA),
        "BA+D_AB vs P": _fit(P, BAi + DAB),
        "BB+D_BA vs P": _fit(P, BBi + DBA),
    }

    def _stats(v):
        v = np.asarray(v, float)
        return dict(mean=float(np.mean(v)),
                    sd=float(np.std(v)),
                    cv=float(np.std(v)/(np.mean(v)+1e-12)))
    basic = {
        "P": _stats(P), "BA": _stats(BAi), "BB": _stats(BBi),
        "D_AB": _stats(DAB), "D_BA": _stats(DBA)
    }

    with np.errstate(divide='ignore', invalid='ignore'):
        frac = {
            "BA/P":       float(np.nanmean(BAi / P)),
            "BB/P":       float(np.nanmean(BBi / P)),
            "BA+D_AB/P":  float(np.nanmean((BAi + DAB) / P)),
            "BB+D_BA/P":  float(np.nanmean((BBi + DBA) / P)),
        }
    return fits, basic, frac

def minmax_envelope(t, x, n_bins=2500):
    n_bins = int(max(50, n_bins))
    idx = np.linspace(0, len(x), n_bins+1, dtype=int)
    tc  = 0.5*(t[idx[:-1]] + t[np.clip(idx[1:]-1, 0, len(t)-1)])
    xmin = np.array([x[a:b].min() for a, b in zip(idx[:-1], idx[1:])])
    xmax = np.array([x[a:b].max() for a, b in zip(idx[:-1], idx[1:])])
    return tc, xmin, xmax

# ==================== Simulación segmentada (caché) ====================
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

# ================================ UI ================================
st.title("Hindmarsh–Rose (2 neuronas) — Invariantes y gráficas")

colA, colB, colC = st.columns([1.2,1.2,1.0])
with colA:
    modo_lbl = st.selectbox("Sinapsis", ["Química (sigmoide)", "Eléctrica (difusiva)"], index=0)
    mode = "quimica" if modo_lbl.startswith("Química") else "electrica"
with colB:
    integ_lbl = st.radio("Integrador", ["LSODA (SciPy)", "RK4 (NumPy)", "RK6"], 
                         index=0 if HAVE_SCIPY else 1, horizontal=True)
    integrator = "LSODA" if (integ_lbl.startswith("LSODA") and HAVE_SCIPY) else ("RK6" if "RK6" in integ_lbl else "RK4")
with colC:
    theme_lbl = st.radio("Tema Plotly", ["Claro","Oscuro"], index=1, horizontal=True)
    THEME_ARG = "streamlit" if theme_lbl=="Claro" else None
    TEMPLATE  = "plotly_white" if theme_lbl=="Claro" else None

st.sidebar.header("Parámetros del modelo")
e     = st.sidebar.slider("e (corriente)", 2.8, 3.5, 3.282, 0.001)
u     = st.sidebar.slider("u (escala lenta)", 0.0008, 0.0040, 0.0021, 0.0001, format="%.4f")
Esyn  = st.sidebar.slider("Esyn (química)", -3.0, 0.0, -1.8, 0.1)
Vfast = st.sidebar.slider("Vfast (química)", -2.0, 1.0, -1.1, 0.05)
sfast = st.sidebar.slider("sfast (química)", 0.05, 1.0, 0.20, 0.05)

st.sidebar.header("Tiempo y resolución")
T_over = st.sidebar.slider("T_over (s, panorámica)", 100.0, 15000.0, 3000.0, 50.0)
N_over = st.sidebar.slider("N_over (puntos)", 2001, 80001, 30001, 2000)
T_det  = st.sidebar.slider("T_detalle (s)", 20.0, 300.0, 80.0, 5.0)
N_det  = st.sidebar.slider("N_det (puntos, detalle)", 20001, 1_000_000, 1_000_000, 10_000)

st.sidebar.header("Detección")
auto_vth = st.sidebar.checkbox("Umbral de ráfaga automático (robusto)", value=True)
v_th     = st.sidebar.slider("v_th (manual)", -3.0, 2.0, -0.60, 0.02)
pk_thr   = st.sidebar.slider("Umbral de picos (x)", -2.0, 2.0, 0.20, 0.01)
min_on   = st.sidebar.slider("Duración mínima ON (s)", 0.02, 0.40, 0.10, 0.01)
min_off  = st.sidebar.slider("Duración mínima OFF (s)", 0.01, 0.30, 0.05, 0.01)
min_gap  = st.sidebar.slider("Separación mínima entre picos (s)", 0.004, 0.050, 0.010, 0.001, format="%.3f")

st.sidebar.header("Vistas opcionales")
show_phase = st.sidebar.checkbox("Mostrar retratos de fase", value=False)
show_hist  = st.sidebar.checkbox("Mostrar histogramas ISI", value=False)

# Condiciones iniciales (idénticas al C)
y0 = np.array([-0.915325, -3.208968, 3.350784, -1.307949, -7.580493, 3.068898], float)
params = dict(e=e, u=u, Esyn=Esyn, Vfast=Vfast, sfast=sfast)

# ======================== PANORÁMICA ========================
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
                       height=320, margin=dict(l=10, r=10, b=10, t=50), template=TEMPLATE or None)
st.plotly_chart(fig_over, use_container_width=True, theme=THEME_ARG)

# ========================= DETALLE =========================
# El detalle parte del estado final de la panorámica
y0_det = sol_over[-1, :].copy()
T0_det = 0.0
tt, sol = simulate_segment(mode, integrator, params, T0_det, T_det, N_det, y0_det)
x1, y1, z1, x2, y2, z2 = sol.T

# Umbrales robustos (percentiles) si auto_vth
qA_lo, qA_hi = float(np.quantile(x1, 0.20)), float(np.quantile(x1, 0.90))
qB_lo, qB_hi = float(np.quantile(x2, 0.20)), float(np.quantile(x2, 0.90))
v_th_A = (qA_lo + 0.45*(qA_hi - qA_lo)) if auto_vth else v_th
v_th_B = (qB_lo + 0.45*(qB_hi - qB_lo)) if auto_vth else v_th

# --- Picos (con fallback de umbral) ---
def _peaks_with_fallback(x, thr, qlo, qhi):
    pk = detect_spikes(tt, x, thr=thr, min_gap=min_gap)
    if len(pk) < 3:
        alt_thr = max(thr*0.5, qlo + 0.30*(qhi - qlo))
        pk2 = detect_spikes(tt, x, thr=alt_thr, min_gap=min_gap)
        if len(pk2) > len(pk):
            st.warning("Fallback de picos activado (umbral relajado).")
            return pk2
    return pk

sp1 = _peaks_with_fallback(x1, pk_thr, qA_lo, qA_hi)
sp2 = _peaks_with_fallback(x2, pk_thr, qB_lo, qB_hi)

# --- Ráfagas por cruce (umbral robusto/manual) ---
BA  = detect_bursts(tt, x1, v_th=v_th_A, min_on=min_on, min_off=min_off)
BB  = detect_bursts(tt, x2, v_th=v_th_B, min_on=min_on, min_off=min_off)

# --- Fallback por ISI si no se detectó nada en una neurona ---
if len(BA) == 0 and len(sp1) >= 3:
    on_thr, off_thr = auto_isi_thresholds(tt, sp1)
    BA = bursts_from_peaks(tt, sp1, on_thr, off_thr, min_on=max(0.04, 0.8*min_on))
    if len(BA):
        st.warning("Ráfagas N1 obtenidas por ISI (fallback).")
if len(BB) == 0 and len(sp2) >= 3:
    on_thr, off_thr = auto_isi_thresholds(tt, sp2)
    BB = bursts_from_peaks(tt, sp2, on_thr, off_thr, min_on=max(0.04, 0.8*min_on))
    if len(BB):
        st.warning("Ráfagas N2 obtenidas por ISI (fallback).")

# Downsample visual para series
max_pts = 8000
step_vis = max(1, int(len(tt)/max_pts))
Tvis = tt[::step_vis]; X1vis = x1[::step_vis]; X2vis = x2[::step_vis]

# ============================ TABS ============================
tab_ts, tab_raster, tab_ids, tab_tables = st.tabs(
    ["Series temporales", "Raster de picos", "IDS (invariantes)", "Tablas"]
)

with tab_ts:
    fig_det = go.Figure()
    fig_det.add_trace(go.Scatter(x=Tvis, y=X1vis, name='x1 (A)', mode='lines'))
    fig_det.add_trace(go.Scatter(x=Tvis, y=X2vis, name='x2 (B)', mode='lines'))
    for on, off in BA: fig_det.add_vline(x=on, line_dash='dash', opacity=0.35); fig_det.add_vline(x=off, line_dash='dash', opacity=0.35)
    for on, off in BB: fig_det.add_vline(x=on, line_dash='dot',  opacity=0.35); fig_det.add_vline(x=off, line_dash='dot',  opacity=0.35)
    if len(sp1): fig_det.add_trace(go.Scatter(x=tt[sp1], y=x1[sp1], mode='markers', marker=dict(size=5), name='picos A'))
    if len(sp2): fig_det.add_trace(go.Scatter(x=tt[sp2], y=x2[sp2], mode='markers', marker=dict(size=5), name='picos B'))
    fig_det.add_hline(y=v_th_A, line_dash='dash', opacity=0.45)
    fig_det.update_layout(title=f"Detalle — {modo_lbl} | integ={integ_lbl}",
                          xaxis_title="tiempo (s)", yaxis_title="x",
                          height=420, margin=dict(l=10, r=10, b=10, t=60), template=TEMPLATE or None)
    st.plotly_chart(fig_det, use_container_width=True, theme=THEME_ARG)

with tab_raster:
    fig_r = go.Figure()
    fig_r.add_trace(go.Scatter(x=(tt[sp1] if len(sp1) else []), y=np.ones(len(sp1)), mode='markers', name='A'))
    fig_r.add_trace(go.Scatter(x=(tt[sp2] if len(sp2) else []), y=np.zeros(len(sp2)), mode='markers', name='B'))
    fig_r.update_layout(title="Raster de picos", xaxis_title="tiempo (s)",
                        yaxis=dict(tickmode='array', tickvals=[0,1], ticktext=["B","A"]),
                        height=260, margin=dict(l=10, r=10, b=10, t=60), template=TEMPLATE or None)
    st.plotly_chart(fig_r, use_container_width=True, theme=THEME_ARG)

with tab_ids:
    # intentamos ambos anclajes y elegimos el que produzca más ciclos
    IvA = pair_intervals_anchor(BA, BB) if (len(BA) and len(BB)) else {}
    IvB = pair_intervals_anchor(BB, BA) if (len(BA) and len(BB)) else {}
    candidates = [("A", IvA), ("B", IvB)]
    anchor, Iv = max(candidates, key=lambda kv: len(kv[1].get("P", [])) if kv[1] else 0)

    if not Iv:
        st.warning("No se emparejaron ciclos completos en la ventana de detalle. Ajusta umbrales o la ventana temporal.")
    else:
        st.caption(f"Anclaje usado para IDS: **{anchor}**.")
        P, BAi, BBi, DAB, DBA = Iv["P"], Iv["BA"], Iv["BB"], Iv["D_AB"], Iv["D_BA"]

        # 1) Dispersión + ajustes (6 relaciones)
        titles = ["BA vs P", "BB vs P", "D_AB vs P", "D_BA vs P", "BA + D_AB vs P", "BB + D_BA vs P"]
        fig_cmp = make_subplots(rows=2, cols=3, subplot_titles=titles)
        pairs = [(BAi,P),(BBi,P),(DAB,P),(DBA,P),(BAi+DAB,P),(BBi+DBA,P)]
        for k,(y,x) in enumerate(pairs, 1):
            m,q,r2 = linfit(x,y)
            xx = np.linspace(np.min(x), np.max(x), 100)
            r = 1 if k<=3 else 2; c = ((k-1)%3)+1
            fig_cmp.add_trace(go.Scatter(x=x, y=y, mode='markers', name=titles[k-1]), row=r, col=c)
            fig_cmp.add_trace(go.Scatter(x=xx, y=m*xx+q, mode='lines', showlegend=False), row=r, col=c)
            fig_cmp.update_xaxes(title_text="P (s)", row=r, col=c); fig_cmp.update_yaxes(title_text="valor (s)", row=r, col=c)
        fig_cmp.update_layout(height=560, title="Comparativas IDS (dispersión + ajuste)",
                              showlegend=False, margin=dict(l=10, r=10, b=10, t=60), template=TEMPLATE or None)
        st.plotly_chart(fig_cmp, use_container_width=True, theme=THEME_ARG)

        # 2) Métricas
        fits, basic, frac = ids_metrics(P, BAi, BBi, DAB, DBA)
        df_fits  = pd.DataFrame(fits).T[["m","q","r2","rmse"]]
        df_basic = pd.DataFrame(basic).T[["mean","sd","cv"]]
        df_frac  = pd.DataFrame([frac]).T; df_frac.columns = ["media"]

        c1,c2,c3 = st.columns([1.2,1.2,0.8])
        with c1:
            st.subheader("Ajustes lineales")
            st.dataframe(df_fits.style.format({"m":"{:.3f}","q":"{:.3f}","r2":"{:.3f}","rmse":"{:.3f}"}),
                         use_container_width=True)
        with c2:
            st.subheader("Estadísticos básicos")
            st.dataframe(df_basic.style.format({"mean":"{:.3f}","sd":"{:.3f}","cv":"{:.3f}"}),
                         use_container_width=True)
        with c3:
            st.subheader("Fracciones normalizadas")
            st.dataframe(df_frac.style.format({"media":"{:.3f}"}), use_container_width=True)

with tab_tables:
    c1,c2,c3 = st.columns([1,1,1])
    def bursts_df(B):
        if not B: return pd.DataFrame(columns=["burst_start","burst_end","dur"])
        arr = np.array(B)
        return pd.DataFrame({"burst_start":arr[:,0], "burst_end":arr[:,1], "dur":arr[:,1]-arr[:,0]})
    dfA = bursts_df(BA); dfB = bursts_df(BB)
    with c1:
        st.subheader("Intervalos N1"); st.dataframe(dfA, use_container_width=True, height=280)
    with c2:
        st.subheader("Intervalos N2"); st.dataframe(dfB, use_container_width=True, height=280)
    with c3:
        IvA = pair_intervals_anchor(BA, BB) if (len(BA) and len(BB)) else {}
        IvB = pair_intervals_anchor(BB, BA) if (len(BA) and len(BB)) else {}
        anchor, Iv = max([("A", IvA), ("B", IvB)],
                         key=lambda kv: len(kv[1].get("P", [])) if kv[1] else 0)
        st.subheader(f"Secuencias (anclaje {anchor})")
        if Iv:
            dfIv = pd.DataFrame({"P":Iv["P"], "BA":Iv["BA"], "BB":Iv["BB"], "D_AB":Iv["D_AB"], "D_BA":Iv["D_BA"]})
            st.dataframe(dfIv, use_container_width=True, height=280)
        else:
            st.info("Sin secuencias completas en la ventana de detalle.")

# ========================= Vistas opcionales =========================
if show_phase:
    st.subheader("Retratos de fase (opcional)")
    figp = make_subplots(rows=1, cols=2, subplot_titles=("N1 (x–y)", "N2 (x–y)"))
    figp.add_trace(go.Scatter(x=x1, y=y1, mode='lines', name='N1'), row=1, col=1)
    figp.add_trace(go.Scatter(x=x2, y=y2, mode='lines', name='N2'), row=1, col=2)
    figp.update_xaxes(title_text="x1", row=1, col=1); figp.update_yaxes(title_text="y1", row=1, col=1)
    figp.update_xaxes(title_text="x2", row=1, col=2); figp.update_yaxes(title_text="y2", row=1, col=2)
    figp.update_layout(height=360, showlegend=False, margin=dict(l=10, r=10, b=10, t=60), template=TEMPLATE or None)
    st.plotly_chart(figp, use_container_width=True, theme=THEME_ARG)

if show_hist:
    st.subheader("Histogramas ISI (opcional)")
    fig_h = make_subplots(rows=1, cols=2, subplot_titles=("ISI — N1", "ISI — N2"))
    if len(sp1) > 1:
        isi1 = np.diff(tt[sp1]); fig_h.add_trace(go.Histogram(x=isi1, nbinsx=min(40, max(10, len(isi1)//2))), row=1, col=1)
    if len(sp2) > 1:
        isi2 = np.diff(tt[sp2]); fig_h.add_trace(go.Histogram(x=isi2, nbinsx=min(40, max(10, len(isi2)//2))), row=1, col=2)
    fig_h.update_xaxes(title_text="ISI (s)", row=1, col=1); fig_h.update_yaxes(title_text="Frecuencia", row=1, col=1)
    fig_h.update_xaxes(title_text="ISI (s)", row=1, col=2); fig_h.update_yaxes(title_text="Frecuencia", row=1, col=2)
    fig_h.update_layout(height=360, showlegend=False, bargap=0.05, margin=dict(l=10, r=10, b=10, t=60), template=TEMPLATE or None)
    st.plotly_chart(fig_h, use_container_width=True, theme=THEME_ARG)

# ============================ Diagnóstico ============================
st.info(f"Detección — picos A={len(sp1)}, B={len(sp2)} | ráfagas A={len(BA)}, B={len(BB)} | v_th usados: A={v_th_A:.3f}, B={v_th_B:.3f}")
st.success(f"Muestras detalle: {len(tt):,} | Integrador: {integ_lbl} | Downsample visual: {step_vis}×.")
