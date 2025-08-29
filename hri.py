# -*- coding: utf-8 -*-
# Hindmarsh–Rose (2 neuronas) + invariantes intrínsecos (A–D) + diagrama de ciclo (anclado)

import numpy as np
import pandas as pd
from dataclasses import dataclass
import streamlit as st
from plotly import graph_objects as go
from plotly.subplots import make_subplots

# -------- SciPy opcional (LSODA) --------
try:
    from scipy.integrate import odeint as _odeint
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

st.set_page_config(page_title="HR (2 neuronas) + invariantes", layout="wide")

# ---- Estilo anotaciones (legible)
def _anno(x, y, text, align="center"):
    return dict(
        x=x, y=y, text=text, xref="x", yref="y", showarrow=False,
        bgcolor="rgba(0,0,0,0.70)", bordercolor="white", borderwidth=1.2, borderpad=3,
        font=dict(size=13, color="white"), align=align,
    )

# ============================
#        MODELO HR
# ============================
@dataclass
class HRParams:
    e1: float = 3.282; e2: float = 3.282; mu: float = 0.0021
    s1: float = 1.0;   s2: float = 1.0
    v1: float = 0.1;   v2: float = 0.1

@dataclass
class SynChemSigm:
    g_syn: float = 0.35; theta: float = -0.25; k: float = 10.0; E_syn: float = -2.0

@dataclass
class SynChemCPP:
    g_fast: float = 0.10; Esyn: float = -1.8; Vfast: float = -1.1; sfast: float = 0.2

@dataclass
class SynElec:
    g_el: float = 0.05

def sigm(x, th, k): return 1.0/(1.0 + np.exp(-k*(x - th)))

def rhs_hr(y, t, prm: HRParams, mode: str, sc: SynChemSigm, cc: SynChemCPP, se: SynElec):
    x1,y1,z1, x2,y2,z2 = y
    if mode == "quimica_sigmoidal":
        s1s = sigm(x1, sc.theta, sc.k); s2s = sigm(x2, sc.theta, sc.k)
        Isyn1 = sc.g_syn * s2s * (sc.E_syn - x1); Isyn2 = sc.g_syn * s1s * (sc.E_syn - x2)
        Iel1 = Iel2 = 0.0
    elif mode == "quimica_cpp":
        Isyn1 = -cc.g_fast * (x1 - cc.Esyn) / (1.0 + np.exp(cc.sfast*(cc.Vfast - x2)))
        Isyn2 = -cc.g_fast * (x2 - cc.Esyn) / (1.0 + np.exp(cc.sfast*(cc.Vfast - x1)))
        Iel1 = Iel2 = 0.0
    elif mode == "electrica":
        Iel1  = se.g_el * (x2 - x1); Iel2  = se.g_el * (x1 - x2); Isyn1 = Isyn2 = 0.0
    else:
        Isyn1 = Isyn2 = Iel1 = Iel2 = 0.0
    dx1 = y1 + 3*x1**2 - x1**3 - z1 + prm.e1 + Isyn1 + Iel1
    dy1 = 1 - 5*x1**2 - y1
    dz1 = prm.mu * (-prm.v1*z1 + prm.s1*(x1 + 1.6))
    dx2 = y2 + 3*x2**2 - x2**3 - z2 + prm.e2 + Isyn2 + Iel2
    dy2 = 1 - 5*x2**2 - y2
    dz2 = prm.mu * (-prm.v2*z2 + prm.s2*(x2 + 1.6))
    return np.array([dx1,dy1,dz1, dx2,dy2,dz2], dtype=float)

# ============================
#     INTEGRADORES
# ============================
def rk4(f, y0, t, *args):
    y = np.empty((len(t), len(y0)), dtype=float); y[0] = y0
    for i in range(len(t)-1):
        dt = t[i+1]-t[i]
        k1 = f(y[i], t[i], *args)
        k2 = f(y[i] + 0.5*dt*k1, t[i] + 0.5*dt, *args)
        k3 = f(y[i] + 0.5*dt*k2, t[i] + 0.5*dt, *args)
        k4 = f(y[i] + dt*k3,     t[i] + dt,   *args)
        y[i+1] = y[i] + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return y

@st.cache_data(show_spinner=False)
def simulate(mode, e1,e2, mu,s1,s2,v1,v2,
             g_syn,theta,k,Esyn,
             g_fast, Esyn_c, Vfast, sfast,
             g_el,
             x10,y10,z10, x20,y20,z20,
             nsteps, dt, decim, use_lsoda):

    prm = HRParams(e1=float(e1), e2=float(e2), mu=float(mu), s1=float(s1), s2=float(s2),
                   v1=float(v1), v2=float(v2))
    sc  = SynChemSigm(g_syn=float(g_syn), theta=float(theta), k=float(k), E_syn=float(Esyn))
    cc  = SynChemCPP(g_fast=float(g_fast), Esyn=float(Esyn_c), Vfast=float(Vfast), sfast=float(sfast))
    se  = SynElec(g_el=float(g_el))
    y0  = np.array([x10,y10,z10, x20,y20,z20], float)

    # rejillas
    t_full = np.linspace(0.0, float(nsteps)*dt, int(nsteps)+1)
    step   = max(1, int(decim))
    t_out  = t_full[::step]

    rhs = lambda Y,tt: rhs_hr(Y,tt,prm,mode,sc,cc,se)

    if use_lsoda and HAVE_SCIPY:
        # integrar directamente en t_out (más ligero en memoria)
        sol_out = _odeint(lambda Y,tt: rhs(Y,tt), y0, t_out, atol=1e-6, rtol=1e-6)
        return t_out, sol_out

    # RK4: integro fino pero SOLO guardo cada "step"
    y = np.empty((len(t_out), len(y0)), dtype=float)
    y_curr = y0.copy()
    y[0] = y_curr
    t_curr = t_full[0]
    out_i = 1
    for i in range(0, len(t_full)-1):
        dt_i = t_full[i+1]-t_full[i]
        k1 = rhs(y_curr, t_curr)
        k2 = rhs(y_curr + 0.5*dt_i*k1, t_curr + 0.5*dt_i)
        k3 = rhs(y_curr + 0.5*dt_i*k2, t_curr + 0.5*dt_i)
        k4 = rhs(y_curr + dt_i*k3,     t_curr + dt_i)
        y_curr = y_curr + (dt_i/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        t_curr = t_full[i+1]
        if ((i+1) % step) == 0:
            y[out_i] = y_curr
            out_i += 1
            if out_i >= len(t_out):
                break
    return t_out, y

# ============================
#  DETECCIÓN DE RÁFAGAS
# ============================
def detect_bursts(t, x, v_th=-0.60, min_on=0.10, min_off=0.05):
    above = x > v_th
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

# ============================
#  INVARIANTES POR NEURONA (A–D)
# ============================
def _cycles_from_bursts(bursts):
    if not bursts or len(bursts) < 2:
        return np.array([]), np.array([]), np.array([]), [], []
    on  = np.array([b[0] for b in bursts], float)
    off = np.array([b[1] for b in bursts], float)
    P   = on[1:]  - on[:-1]
    B   = (off-on)[:-1]
    IBI = on[1:]  - off[:-1]
    win = [(on[i], on[i+1]) for i in range(len(on)-1)]
    seg = [(on[i], off[i])   for i in range(len(on)-1)]
    return P, B, IBI, win, seg

def _resample_segment(t, x, t0, t1, n=300):
    m = (t >= t0) & (t <= t1)
    if m.sum() < 4: return None, None
    tt = t[m]; xx = x[m]
    tt_new = np.linspace(t0, t1, n); xx_new = np.interp(tt_new, tt, xx)
    return tt_new, xx_new

def plot_single_neuron_invariants(t, x, bursts, tag="N1", n_overlay=10):
    P, B, IBI, win, seg = _cycles_from_bursts(bursts)
    st.subheader(f"Invariantes — {tag}")
    figA = go.Figure()
    figA.add_trace(go.Scatter(x=t, y=x, mode="lines", name=f"{tag}", line=dict(width=1.1)))
    for (on, off) in bursts:
        figA.add_vrect(x0=on, x1=off, fillcolor="#E74C3C" if tag.endswith("1") else "#3498DB",
                       opacity=0.18, line_width=0)
    figA.update_layout(height=260, title="A) Serie temporal con ráfagas",
                       xaxis_title="tiempo (s)", yaxis_title="x")
    st.plotly_chart(figA, use_container_width=True)
    if len(P) == 0:
        st.info("No hay ciclos completos (al menos 2 ráfagas) para esta neurona.")
        return
    figB = go.Figure()
    figB.add_trace(go.Box(y=P,   name="Periodo", boxmean=True))
    figB.add_trace(go.Box(y=B,   name="Burst",   boxmean=True))
    figB.add_trace(go.Box(y=IBI, name="IBI",     boxmean=True))
    figB.update_layout(height=260, title="B) Distribuciones ciclo-a-ciclo")
    st.plotly_chart(figB, use_container_width=True)
    def _hist_trace(v, nb=30): return go.Histogram(x=v, nbinsx=int(nb), showlegend=False)
    def _sct(xv, yv): return go.Scatter(x=xv, y=yv, mode='markers',
                                        marker=dict(size=4, opacity=0.7), showlegend=False)
    figC = make_subplots(rows=3, cols=3,
        subplot_titles=("P","P vs B","P vs IBI","B vs P","B","B vs IBI","IBI vs P","IBI vs B","IBI"),
        horizontal_spacing=0.04, vertical_spacing=0.08)
    figC.add_trace(_hist_trace(P), row=1, col=1);  figC.add_trace(_sct(P,B),   row=1, col=2);  figC.add_trace(_sct(P,IBI), row=1, col=3)
    figC.add_trace(_sct(B,P),      row=2, col=1);  figC.add_trace(_hist_trace(B), row=2, col=2); figC.add_trace(_sct(B,IBI), row=2, col=3)
    figC.add_trace(_sct(IBI,P),    row=3, col=1);  figC.add_trace(_sct(IBI,B), row=3, col=2);   figC.add_trace(_hist_trace(IBI), row=3, col=3)
    figC.update_layout(height=560, title="C) Relación entre Periodo, Burst e IBI")
    st.plotly_chart(figC, use_container_width=True)
    k = min(n_overlay, len(seg))
    figD = go.Figure()
    for i in range(k):
        t0, t1 = seg[i]; tt, xx = _resample_segment(t, x, t0, t1, n=300)
        if tt is None: continue
        tau = (tt - tt[0]) / (tt[-1] - tt[0] + 1e-12)
        figD.add_trace(go.Scatter(x=tau, y=xx, mode='lines', line=dict(width=1), showlegend=False))
    figD.update_layout(height=300, title=f"D) Superposición de {len(figD.data)} ráfagas — {tag}",
                       xaxis_title="tiempo normalizado dentro de la ráfaga", yaxis_title="x")
    st.plotly_chart(figD, use_container_width=True)
    st.caption("Resumen (medias±sd) — Periodo, Burst, IBI")
    df_sum = pd.DataFrame({"Periodo":[P.mean(),P.std(),len(P)],
                           "Burst":[B.mean(),B.std(),len(B)],
                           "IBI":[IBI.mean(),IBI.std(),len(IBI)]},
                           index=["media","sd","n"]).round(4).T
    st.dataframe(df_sum, use_container_width=True)

# ============================
#  DIAGRAMA DE CICLO (anclado X) + tabla
# ============================
COL = {"X":"#E74C3C","Y":"#3498DB","TXT":"#ECF0F1","NEU":"#7F8C8D"}

def _x_anchored_cycles(burstsX, burstsY):
    """
    Ciclos X→Y→X anclados a X. Para cada par consecutivo de X (x_on0, x_on1)
    ubicamos el primer Y dentro de [x_on0, x_on1) y además guardamos el
    siguiente inicio de ráfaga de Y (y_on_next) aunque caiga después de x_on1.
    """
    if not burstsX or len(burstsX) < 2 or not burstsY:
        return []

    X_on  = np.array([b[0] for b in burstsX], float)
    X_off = np.array([b[1] for b in burstsX], float)
    Y_on  = np.array([b[0] for b in burstsY], float)
    Y_off = np.array([b[1] for b in burstsY], float)

    out = []
    j = 0  # índice sobre Y
    for i in range(len(X_on) - 1):
        x_on0, x_off0 = X_on[i],   X_off[i]
        x_on1, x_off1 = X_on[i+1], X_off[i+1]

        # avanza Y hasta la primera ráfaga que empieza dentro del ciclo de X
        while j < len(Y_on) and Y_on[j] < x_on0:
            j += 1
        if j >= len(Y_on) or Y_on[j] >= x_on1:
            # no hay Y dentro de este ciclo de X
            continue

        y_on, y_off = Y_on[j], Y_off[j]

        # siguiente inicio de Y (puede caer después de x_on1)
        if j + 1 < len(Y_on):
            y_on_next  = Y_on[j+1]
            y_off_next = Y_off[j+1]
        else:
            y_on_next = np.nan
            y_off_next = np.nan

        out.append(dict(
            x_on0=x_on0, x_off0=x_off0, x_on1=x_on1, x_off1=x_off1,
            y_on=y_on, y_off=y_off, y_on_next=y_on_next, y_off_next=y_off_next
        ))
    return out

def _cycle_metrics(cyc):
    x_on0, x_off0 = cyc["x_on0"], cyc["x_off0"]
    x_on1, x_off1 = cyc["x_on1"], cyc["x_off1"]
    y_on,  y_off  = cyc["y_on"],  cyc["y_off"]
    y_on_next     = cyc.get("y_on_next", np.nan)

    # X
    B_X   = x_off0 - x_on0
    P_X   = x_on1  - x_on0
    IBI_X = P_X - B_X

    # intervalos/retardos
    I_XY = y_on  - x_on0
    D_XY = y_on  - x_off0
    I_YX = x_on1 - y_on
    D_YX = x_on1 - y_off

    # Y (exacto si tenemos y_on_next)
    B_Y = y_off - y_on
    if np.isfinite(y_on_next):
        P_Y   = y_on_next - y_on
        IBI_Y = P_Y - B_Y
        approx_PY = False
    else:
        P_Y   = I_XY + I_YX
        IBI_Y = P_Y - B_Y
        approx_PY = True

    return dict(
        P_X=P_X, IBI_X=IBI_X, B_X=B_X,
        P_Y=P_Y, IBI_Y=IBI_Y, B_Y=B_Y,
        D_XY=D_XY, D_YX=D_YX, I_XY=I_XY, I_YX=I_YX,
        approx_PY=approx_PY,
        y_on_next=y_on_next
    )

def _span(fig, x0, x1, y, text, color):
    fig.add_shape(type="line", x0=x0, x1=x1, y0=y, y1=y,
                  line=dict(color=color, width=3))
    fig.add_annotation(**_anno((x0+x1)/2, y, text))

def _span_clipped(fig, x0, x1, y, text, color, tmin, tmax):
    xa = max(float(x0), float(tmin))
    xb = min(float(x1), float(tmax))
    if xb > xa:  # hay tramo visible
        fig.add_shape(type="line", x0=xa, x1=xb, y0=y, y1=y,
                      line=dict(color=color, width=3))
        fig.add_annotation(**_anno((xa+xb)/2.0, y, text))
    else:
        # No hay tramo visible: deja un recordatorio al borde
        edge_x = tmax if x0 > tmax else tmin
        fig.add_annotation(**_anno(edge_x, y, text+" (fuera de ventana)", align="left"))

def draw_cycle_diagram_pretty(t, x1, x2, burstsA, burstsB, title, sim_sig=None):
    st.subheader("Diagrama de ciclo y métricas (anclado en X)")

    # clave base para sliders (cambia cuando cambia la simulación)
    base_key = str(sim_sig) if sim_sig is not None else f"{len(t)}_{int(round(float(t[-1])*1000))}"

    # Ventana temporal (por defecto: TODO el registro)
    step_f = float(t[1] - t[0]) if len(t) > 1 else 0.01
    t0, t1 = st.slider(
        "Ventana temporal para detectar ciclos",
        min_value=float(t[0]),
        max_value=float(t[-1]),
        value=(float(t[0]), float(t[-1])),
        step=step_f,
        key=f"win_{base_key}"
    )

    # Filtrar ráfagas en la ventana y emparejar X→Y→X
    BA = [(on, off) for (on, off) in burstsA if (on >= t0 and on < t1)]
    BB = [(on, off) for (on, off) in burstsB if (on >= t0 and on < t1)]
    cycles = _x_anchored_cycles(BA, BB)

    st.caption(f"Ráfagas en ventana: X={len(BA)} | Y={len(BB)} | Ciclos X→Y→X detectados={len(cycles)}")
    if not cycles:
        st.info("No se encontró ningún ciclo X→Y→X dentro de la ventana seleccionada. Ajusta la ventana o los parámetros.")
        return

    # Selección de ciclo (por defecto: el último detectado en la ventana)
    if len(cycles) == 1:
        idx = 0
        st.caption("Se detectó **un** ciclo en la ventana (no se muestra slider).")
    else:
        idx = st.slider(
            "Ciclo (índice en X)",
            min_value=0, max_value=len(cycles)-1,
            value=len(cycles)-1, step=1,
            key=f"cyc_{base_key}_{len(cycles)}_{int(round(t0*1000))}_{int(round(t1*1000))}"
        )

    cyc = cycles[idx]
    met = _cycle_metrics(cyc)

    # Destino REAL del periodo Y: y_on -> y_on_next (si existe)
    y_end = met["y_on_next"] if np.isfinite(met["y_on_next"]) else (cyc["y_on"] + met["P_Y"])

    # Panorámico vs. zoom al ciclo (extiende para cubrir y_end si cae más allá)
    zoom_ciclo = st.toggle(
        "Zoom al ciclo seleccionado", value=False,
        help="Apagado = muestra la ventana [t0, t1] completa."
    )
    if zoom_ciclo:
        x0, x1n = cyc["x_on0"], cyc["x_on1"]
        P = x1n - x0
        tmin = max(t0, x0 - 0.20*P)
        tmax = min(t1, max(x1n + 0.20*P, y_end + 0.05*P))
    else:
        tmin, tmax = t0, t1

    m = (t >= tmin) & (t <= tmax)
    if not np.any(m):
        m = slice(None)
        tmin, tmax = float(t[0]), float(t[-1])

    # Figura
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t[m], y=x1[m], name="X (N1)", line=dict(color=COL["X"], width=2.2)))
    fig.add_trace(go.Scatter(x=t[m], y=x2[m], name="Y (N2)", line=dict(color=COL["Y"], width=2.2)))

    fig.add_vrect(
        x0=cyc["x_on0"], x1=cyc["x_off0"], fillcolor=COL["X"], opacity=0.18, line_width=0,
        annotation_text="X burst", annotation_position="top left"
    )
    fig.add_vrect(
        x0=cyc["y_on"],  x1=cyc["y_off"],  fillcolor=COL["Y"], opacity=0.18, line_width=0,
        annotation_text="Y burst", annotation_position="top left"
    )
    fig.add_vrect(x0=cyc["x_on1"], x1=cyc["x_off1"], fillcolor=COL["X"], opacity=0.12, line_width=0)

    y_top = max(np.max(x1[m]), np.max(x2[m])); y_bot = min(np.min(x1[m]), np.min(x2[m]))
    dy = (y_top - y_bot); yb = y_bot - 0.10*dy

    # Spans recortados al rango visible
    _span_clipped(fig, cyc["x_on0"], cyc["x_on1"], y_top + 0.12*dy, f"Periodo X = {met['P_X']:.3f} s", COL["X"], tmin, tmax)
    _span_clipped(fig, cyc["x_off0"], cyc["x_on1"], y_top + 0.06*dy, f"IBI X = {met['IBI_X']:.3f} s", COL["X"], tmin, tmax)

    texto_py = f"Periodo Y{' ≈' if met['approx_PY'] else ''} = {met['P_Y']:.3f} s"
    _span_clipped(fig, cyc["y_on"], y_end, y_top + 0.00*dy, texto_py, COL["Y"], tmin, tmax)

    _span_clipped(fig, cyc["x_on0"], cyc["x_off0"], yb - 0.00*dy, f"Duración ráfaga X = {met['B_X']:.3f} s", COL["X"], tmin, tmax)
    _span_clipped(fig, cyc["y_on"],  cyc["y_off"],  yb - 0.06*dy, f"Duración ráfaga Y = {met['B_Y']:.3f} s", COL["Y"], tmin, tmax)
    _span_clipped(fig, cyc["x_off0"],cyc["y_on"],   yb - 0.16*dy, f"Retardo X→Y = {met['D_XY']:.3f} s", COL["NEU"], tmin, tmax)
    _span_clipped(fig, cyc["y_off"], cyc["x_on1"], yb - 0.22*dy, f"Retardo Y→X = {met['D_YX']:.3f} s", COL["NEU"], tmin, tmax)
    _span_clipped(fig, cyc["x_on0"], cyc["y_on"],  yb - 0.32*dy, f"Intervalo X→Y = {met['I_XY']:.3f} s",  "#2C3E50", tmin, tmax)
    _span_clipped(fig, cyc["y_on"],  cyc["x_on1"], yb - 0.38*dy, f"Intervalo Y→X = {met['I_YX']:.3f} s", "#2C3E50", tmin, tmax)

    fig.update_layout(
        title=title, xaxis_title="tiempo (s)", yaxis_title="señal x",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        height=480, margin=dict(l=10, r=10, b=10, t=50)
    )
    fig.update_xaxes(range=[tmin, tmax])  # fuerza el rango del eje X
    st.plotly_chart(fig, use_container_width=True)

    # Tabla de métricas (robusta)
    approx_py = bool(met.get("approx_PY", not np.isfinite(met.get("y_on_next", np.nan))))
    name_py = "Periodo Y (≈)" if approx_py else "Periodo Y"

    info = [
        ("Periodo X",          "Tiempo entre inicios de ráfaga de X: onₙ(X)→onₙ₊₁(X)",        "P_X"),
        ("IBI X",              "Intervalo inter-ráfaga de X: Periodo X − Duración X",         "IBI_X"),
        ("Duración ráfaga X",  "Duración de la ráfaga de X en el ciclo n",                    "B_X"),
        (name_py,              "onₙ(Y)→onₙ₊₁(Y) (si no hay siguiente Y, se aproxima I_XY+I_YX)", "P_Y"),
        ("IBI Y (≈)",          "Periodo Y − Duración Y",                                      "IBI_Y"),
        ("Duración ráfaga Y",  "Duración de la ráfaga de Y en el ciclo n",                    "B_Y"),
        ("Retardo X→Y",        "Fin de X → inicio de Y: offₙ(X)→onₙ(Y)",                      "D_XY"),
        ("Retardo Y→X",        "Fin de Y → inicio de X: offₙ(Y)→onₙ₊₁(X)",                    "D_YX"),
        ("Intervalo X→Y",      "Inicio X → inicio Y: onₙ(X)→onₙ(Y)",                          "I_XY"),
        ("Intervalo Y→X",      "Inicio Y → inicio X: onₙ(Y)→onₙ₊₁(X)",                        "I_YX"),
    ]
    rows = []
    for nice, desc, key in info:
        val = met.get(key, np.nan)
        try:
            val = float(val)
        except Exception:
            val = np.nan
        rows.append({"Magnitud": nice, "Definición": desc, "Valor [s]": val})
    df = pd.DataFrame(rows).round({"Valor [s]": 6})
    st.caption("Métricas del ciclo (anclado en X) — definiciones y valores")
    st.dataframe(df, use_container_width=True)


# ============================
#              UI
# ============================
st.title("Hindmarsh–Rose (2 neuronas) — invariantes y coordinación")

with st.sidebar:
    st.header("Configuración")
    syn_mode = st.radio("Sinapsis", ["Química (sigmoidal, HCO)","Química (núcleo C)","Eléctrica (difusiva)"])
    mode_key = {"Química (sigmoidal, HCO)":"quimica_sigmoidal", "Química (núcleo C)":"quimica_cpp", "Eléctrica (difusiva)":"electrica"}[syn_mode]

    nsteps = st.number_input("Nº de pasos (TIME)",  min_value=10_000, max_value=5_500_000, value=1_000_000, step=50_000)
    dt     = st.selectbox("dt (s)", [0.001, 0.0005, 0.0002], index=0)
    decim  = st.number_input("Decimación (cada N pasos)", min_value=50, max_value=500, value=150, step=10)

    st.subheader("Parámetros neurales")
    e1 = st.number_input("I1 (N1)", value=3.282, step=0.001, format="%.3f")
    e2 = st.number_input("I2 (N2)", value=3.282, step=0.001, format="%.3f")
    mu = st.number_input("u (escala lenta)", value=0.0021, step=0.0001, format="%.4f")
    s1 = st.number_input("s1", value=1.0, step=0.1)
    s2 = st.number_input("s2", value=1.0, step=0.1)
    v1 = st.number_input("v1", value=0.1, step=0.01)
    v2 = st.number_input("v2", value=0.1, step=0.01)

    g_el = 0.0
    if mode_key == "quimica_sigmoidal":
        st.subheader("Química sigmoidal")
        g_syn = st.number_input("g_syn", value=0.35, step=0.01)
        theta = st.number_input("θ", value=-0.25, step=0.01)
        kk    = st.number_input("k", value=10.0, step=0.5)
        Esy   = st.number_input("E_syn", value=-2.0, step=0.1)
        g_fast = 0.10; Esy_c=-1.8; Vfast=-1.1; sfast=0.2
    elif mode_key == "quimica_cpp":
        st.subheader("Química (núcleo C)")
        g_fast = st.number_input("g_fast (C)", value=0.10, step=0.01)
        Esy_c  = st.number_input("Esyn (C)",   value=-1.8, step=0.1)
        Vfast  = st.number_input("Vfast (C)",  value=-1.1, step=0.1)
        sfast  = st.number_input("sfast (C)",  value=0.2,  step=0.01)
        g_syn=0.35; theta=-0.25; kk=10.0; Esy=-2.0
    else:
        st.subheader("Eléctrica (difusiva)")
        g_el = st.number_input("g_el", value=0.05, step=0.01)
        g_syn=0.35; theta=-0.25; kk=10.0; Esy=-2.0
        g_fast=0.10; Esy_c=-1.8; Vfast=-1.1; sfast=0.2

    st.subheader("Detección de ráfagas")
    v_th   = st.number_input("Umbral v_th", value=-0.60, step=0.01)
    min_on = st.number_input("Duración mínima ON (s)", value=0.10, step=0.01)
    min_off= st.number_input("Duración mínima OFF (s)", value=0.05, step=0.01)

    st.subheader("Integrador")
    integ = st.radio("Método", ["LSODA (SciPy)"+("" if HAVE_SCIPY else " — no disponible"), "RK4 (NumPy)"],
                     index=0 if HAVE_SCIPY else 1, horizontal=True)
    use_lsoda = (integ.startswith("LSODA") and HAVE_SCIPY)

# Condiciones iniciales (como en C)
y0_def = np.array([-0.915325, -3.208968, 3.350784, -1.307949, -7.580493, 3.068898], float)

# Simulación
t, sol = simulate(mode_key, e1,e2, mu,s1,s2,v1,v2,
                  g_syn,theta,kk,Esy,
                  g_fast,Esy_c,Vfast,sfast,
                  g_el,
                  *y0_def,
                  nsteps, dt, decim, use_lsoda)

x1,y1,z1, x2,y2,z2 = sol[:,0], sol[:,1], sol[:,2], sol[:,3], sol[:,4], sol[:,5]

# Firma de simulación para resetear sliders al cambiar parámetros/datos
sim_sig = f"{mode_key}_{len(t)}_{int(t[-1]*1e6)}_{len(x1)}_{len(x2)}_{round(float(x1[-1]-x1[0]),6)}_{round(float(x2[-1]-x2[0]),6)}"

# Ráfagas (sobre señal decimada)
burstsA = detect_bursts(t, x1, v_th=v_th, min_on=min_on, min_off=min_off)
burstsB = detect_bursts(t, x2, v_th=v_th, min_on=min_on, min_off=min_off)

# ===== Gráficos
st.header("Series temporales (decimadas para visualización)")
fig_over = go.Figure()
fig_over.add_trace(go.Scatter(x=t, y=x1, name="X (N1)", line=dict(color=COL["X"])))
fig_over.add_trace(go.Scatter(x=t, y=x2, name="Y (N2)", line=dict(color=COL["Y"])))
for (on, off) in burstsA: fig_over.add_vrect(x0=on, x1=off, fillcolor=COL["X"], opacity=0.10, line_width=0)
for (on, off) in burstsB: fig_over.add_vrect(x0=on, x1=off, fillcolor=COL["Y"], opacity=0.10, line_width=0)
fig_over.update_layout(height=340, xaxis_title="tiempo (s)", yaxis_title="x",
                       margin=dict(l=10,r=10,b=10,t=10))
st.plotly_chart(fig_over, use_container_width=True)

st.divider()
draw_cycle_diagram_pretty(
    t, x1, x2, burstsA, burstsB,
    title=("Diagrama de ciclo — " + ("Química (sigmoidal)" if mode_key=="quimica_sigmoidal"
                                     else "Química (núcleo C)" if mode_key=="quimica_cpp"
                                     else "Eléctrica (difusiva)")),
    sim_sig=sim_sig
)

st.divider()
st.header("Invariantes por neurona (A–D)")
who = st.radio("Selecciona neurona", ["N1 (X)","N2 (Y)"], horizontal=True)
if who.startswith("N1"):
    plot_single_neuron_invariants(t, x1, burstsA, tag="N1")
else:
    plot_single_neuron_invariants(t, x2, burstsB, tag="N2")

st.info(f"Muestras visuales: {len(t):,}  |  Integrador: {'LSODA' if use_lsoda else 'RK4'}  |  Decimación: {decim}×")
