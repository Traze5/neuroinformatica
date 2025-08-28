# app.py — Streamlit: Hindmarsh–Rose 2N (química / eléctrica) con invariantes y gráficos
# Autor: tú + tu copiloto :D
# Requisitos: streamlit, numpy, matplotlib
# Ejecuta: streamlit run app.py

import math
from dataclasses import dataclass
from typing import Tuple, Dict, Literal

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ===========================
# Estilo general (matplotlib)
# ===========================
plt.rcParams.update({
    "figure.figsize": (8.5, 4.8),
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.titlesize": 14,
    "legend.frameon": False,
    "lines.linewidth": 1.2,
})

# ===========================
# Modelo Hindmarsh–Rose 2N
# ===========================
@dataclass
class HRParams:
    # Parámetros del modelo
    e: float = 3.282
    u: float = 0.0021
    s1: float = 1.0
    s2: float = 1.0
    v1: float = 0.1
    v2: float = 0.1
    # Sinapsis química
    Esyn: float = -1.8
    Vfast: float = -1.1
    sfast: float = 0.2
    # Integración
    dt: float = 0.001
    t0: float = 0.00005

def rhs_chemical(x: np.ndarray, p: HRParams) -> np.ndarray:
    """Acoplamiento químico: término sigmoide con Esyn, Vfast, sfast."""
    x1, y1, z1, x2, y2, z2 = x
    dx1 = y1 + 3.0*x1*x1 - x1*x1*x1 - z1 + p.e - 0.1*(x1 - p.Esyn) / (1.0 + np.exp(p.sfast*(p.Vfast - x2)))
    dy1 = 1.0 - 5.0*x1*x1 - y1
    dz1 = p.u * (-p.v1*z1 + p.s1*(x1 + 1.6))

    dx2 = y2 + 3.0*x2*x2 - x2*x2*x2 - z2 + p.e - 0.1*(x2 - p.Esyn) / (1.0 + np.exp(p.sfast*(p.Vfast - x1)))
    dy2 = 1.0 - 5.0*x2*x2 - y2
    dz2 = p.u * (-p.v2*z2 + p.s2*(x2 + 1.6))
    return np.array([dx1, dy1, dz1, dx2, dy2, dz2], dtype=np.float64)

def rhs_electrical(x: np.ndarray, p: HRParams) -> np.ndarray:
    """Acoplamiento eléctrico: término difusivo lineal."""
    x1, y1, z1, x2, y2, z2 = x
    dx1 = y1 + 3.0*x1*x1 - x1*x1*x1 - z1 + p.e + 0.05*(x1 - x2)
    dy1 = 1.0 - 5.0*x1*x1 - y1
    dz1 = p.u * (-p.v1*z1 + p.s1*(x1 + 1.6))

    dx2 = y2 + 3.0*x2*x2 - x2*x2*x2 - z2 + p.e + 0.05*(x2 - x1)
    dy2 = 1.0 - 5.0*x2*x2 - y2
    dz2 = p.u * (-p.v2*z2 + p.s2*(x2 + 1.6))
    return np.array([dx1, dy1, dz1, dx2, dy2, dz2], dtype=np.float64)

def rk6_step(x: np.ndarray, dt: float, f, p: HRParams) -> np.ndarray:
    """Runge–Kutta de 6 etapas con coeficientes idénticos a tu C."""
    k = np.zeros((6, x.size), dtype=np.float64)
    k[0] = dt * f(x, p)

    a = x + 0.2*k[0]
    k[1] = dt * f(a, p)

    a = x + 0.075*k[0] + 0.225*k[1]
    k[2] = dt * f(a, p)

    a = x + 0.3*k[0] - 0.9*k[1] + 1.2*k[2]
    k[3] = dt * f(a, p)

    a = x + 0.075*k[0] + 0.675*k[1] - 0.6*k[2] + 0.75*k[3]
    k[4] = dt * f(a, p)

    a = x + 0.660493827160493*k[0] + 2.5*k[1] - 5.185185185185185*k[2] + 3.888888888888889*k[3] - 0.864197530864197*k[4]
    k[5] = dt * f(a, p)

    x_new = x + 0.098765432098765*k[0] + 0.396825396825396*k[2] + 0.231481481481481*k[3] + 0.308641975308641*k[4] - 0.035714285714285*k[5]
    return x_new

def simulate(
    n_steps: int,
    params: HRParams,
    variant: Literal["chemical", "electrical"] = "chemical",
    sample_every: int = 1,
    record_pre_time: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simula y devuelve (t, Y) con columnas [x1,y1,z1,x2,y2,z2]."""
    x = np.array([-0.915325, -3.208968, 3.350784, -1.307949, -7.580493, 3.068898], dtype=np.float64)
    t = params.t0
    dt = params.dt

    f = rhs_chemical if variant == "chemical" else rhs_electrical

    # Prealoca
    n_store = n_steps // sample_every + int(n_steps % sample_every != 0)
    t_out = np.zeros(n_store, dtype=np.float64)
    y_out = np.zeros((n_store, 6), dtype=np.float64)

    w = 0
    serie = 0
    for i in range(n_steps):
        t_pre = t
        x = rk6_step(x, dt, f, params)
        t += dt

        serie = (serie + 1) % sample_every
        if serie == sample_every - 1:
            t_out[w] = (t_pre if record_pre_time else t)
            y_out[w] = x
            w += 1

    return t_out[:w], y_out[:w]

# ===========================
# Detección de picos e ISI
# ===========================
def detect_spikes(x: np.ndarray, t: np.ndarray, thr: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Detecta máximos locales por encima de umbral."""
    if x.size < 3:
        return np.array([]), np.array([])
    left = x[1:-1] > x[:-2]
    right = x[1:-1] > x[2:]
    above = x[1:-1] > thr
    idx = np.where(left & right & above)[0] + 1
    return t[idx], idx

def isi_stats(t_spk: np.ndarray) -> Dict[str, float]:
    if t_spk.size < 2:
        return {k: np.nan for k in ["n_spikes","rate_hz","mean","std","cv","median","q25","q75","skew","kurt"]}
    isi = np.diff(t_spk)
    n_spikes = int(t_spk.size)
    duration = t_spk[-1] - t_spk[0]
    rate_hz = (n_spikes - 1) / duration if duration > 0 else np.nan
    mean = float(np.mean(isi))
    std = float(np.std(isi, ddof=1)) if isi.size > 1 else 0.0
    cv = std / mean if mean > 0 else np.nan
    median = float(np.median(isi))
    q25, q75 = float(np.percentile(isi, 25)), float(np.percentile(isi, 75))
    # Asimetría y curtosis (exceso)
    m = isi - mean
    m2 = np.mean(m**2)
    m3 = np.mean(m**3)
    m4 = np.mean(m**4)
    skew = m3 / (m2**1.5 + 1e-12)
    kurt = m4 / (m2**2 + 1e-12) - 3.0
    return dict(n_spikes=n_spikes, rate_hz=rate_hz, mean=mean, std=std, cv=cv,
                median=median, q25=q25, q75=q75, skew=float(skew), kurt=float(kurt))

def return_map(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return (arr[:-1], arr[1:]) if arr.size >= 2 else (np.array([]), np.array([]))

# ===========================
# Correlación cruzada
# ===========================
def xcorr(a: np.ndarray, b: np.ndarray, max_lag_s: float, dt: float, decim: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Correlación cruzada normalizada de series (con decimación opcional)."""
    if decim > 1:
        a = a[::decim]
        b = b[::decim]
        dt = dt * decim
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    n = len(a)
    max_lag = int(max_lag_s / dt)
    # correlación por FFT sería más eficiente, pero para tamaños moderados esto basta
    corr_full = np.correlate(a, b, mode="full") / max(1, n)
    lags = np.arange(-n + 1, n)
    mask = (lags >= -max_lag) & (lags <= max_lag)
    return lags[mask] * dt, corr_full[mask]

# ===========================
# Streamlit UI
# ===========================
st.set_page_config(page_title="Hindmarsh–Rose 2N • Invariantes", layout="wide")

st.title("Hindmarsh–Rose (2 neuronas) — invariantes y gráficas")
st.caption("Selección de sinapsis, simulación con RK6, detección de picos, ISI, mapas de retorno y correlación cruzada.")

with st.sidebar:
    st.header("Parámetros")
    variant = st.selectbox("Sinapsis", ["chemical", "electrical"], format_func=lambda s: "Química (sigmoide)" if s=="chemical" else "Eléctrica (difusiva)")
    n_steps = st.slider("Pasos de integración (dt=0.001 s)", min_value=20_000, max_value=600_000, value=200_000, step=10_000)
    sample_every = st.selectbox("Guardar cada (decimación para series)", [1, 2, 5, 10], index=0, help="1 = guardar todo; >1 reduce tamaño/tiempo de gráficos.")
    thr_method = st.selectbox("Umbral de picos", ["0.0 (fijo)", "media + 0.5·std", "mediana"], index=0)
    max_lag_s = st.slider("Máx. retardo para xcorr (s)", 0.1, 5.0, 2.0, 0.1)
    xcorr_decim = st.selectbox("Decimación para xcorr", [1, 2, 5, 10], index=2)
    do_zoom = st.checkbox("Mostrar zoom final (8 s)", value=True)
    st.divider()
    st.caption("IC y parámetros por defecto clonados del código C. Puedes ajustar si quieres:")
    e_val = st.number_input("e (corriente)", value=3.282, step=0.01, format="%.3f")
    u_val = st.number_input("u (lento)", value=0.0021, step=0.0001, format="%.4f")
    Esyn_val = st.number_input("Esyn (química)", value=-1.8, step=0.1, format="%.1f")
    Vfast_val = st.number_input("Vfast (química)", value=-1.1, step=0.1, format="%.1f")
    sfast_val = st.number_input("sfast (química)", value=0.2, step=0.05, format="%.2f")
    st.divider()
    run_btn = st.button("🚀 Simular / Actualizar", type="primary", use_container_width=True)

# Parámetros del modelo
params = HRParams(e=e_val, u=u_val, Esyn=Esyn_val, Vfast=Vfast_val, sfast=sfast_val)

@st.cache_data(show_spinner=False)
def run_sim(n_steps: int, params: HRParams, variant: str, sample_every: int):
    # Guardamos todo para análisis fino; record_pre_time=False (solo relevante si quisieras clonar .out del C)
    return simulate(n_steps=n_steps, params=params, variant=variant, sample_every=sample_every, record_pre_time=False)

if run_btn or "last_run" not in st.session_state:
    t, Y = run_sim(n_steps, params, variant, sample_every)
    st.session_state["t"] = t
    st.session_state["Y"] = Y
    st.session_state["last_run"] = True

t = st.session_state["t"]
Y = st.session_state["Y"]
dt_eff = params.dt * sample_every

x1, y1, z1, x2, y2, z2 = Y[:,0], Y[:,1], Y[:,2], Y[:,3], Y[:,4], Y[:,5]

# Umbral de picos
if thr_method.startswith("0.0"):
    thr = 0.0
elif "media" in thr_method:
    thr = float(np.mean(x1)) + 0.5*float(np.std(x1))
else:
    thr = float(np.median(x1))

# Detectar picos e ISI
t_sp1, idx1 = detect_spikes(x1, t, thr=thr)
t_sp2, idx2 = detect_spikes(x2, t, thr=thr)
isi1 = np.diff(t_sp1) if t_sp1.size > 1 else np.array([])
isi2 = np.diff(t_sp2) if t_sp2.size > 1 else np.array([])
stats1 = isi_stats(t_sp1)
stats2 = isi_stats(t_sp2)

# ============
# Layout tabs
# ============
tab_ts, tab_isi, tab_return, tab_phase, tab_xcorr, tab_stats = st.tabs(
    ["Series temporales", "Histogramas ISI", "Mapas de retorno", "Retratos de fase", "Correlación cruzada", "Resumen estadístico"]
)

with tab_ts:
    c1, c2 = st.columns([2,1])
    with c1:
        fig, ax = plt.subplots()
        ax.plot(t, x1, label="x1")
        ax.plot(t, x2, label="x2")
        # marcar picos (si hay)
        if t_sp1.size:
            ax.plot(t_sp1, x1[idx1], "o", ms=3, alpha=0.6, label="picos x1")
        if t_sp2.size:
            ax.plot(t_sp2, x2[idx2], "o", ms=3, alpha=0.6, label="picos x2")
        ax.set_xlabel("Tiempo (s)"); ax.set_ylabel("x(t)")
        ax.set_title(f"Series temporales — {variant}")
        ax.legend(ncol=3, fontsize=9)
        st.pyplot(fig, clear_figure=True)

        if do_zoom and t.size > 0:
            fig2, ax2 = plt.subplots()
            mask = t > (t[-1] - 8.0)
            ax2.plot(t[mask], x1[mask], label="x1")
            ax2.plot(t[mask], x2[mask], label="x2")
            ax2.set_xlabel("Tiempo (s)"); ax2.set_ylabel("x(t)")
            ax2.set_title("Zoom final (8 s)")
            ax2.legend()
            st.pyplot(fig2, clear_figure=True)

    with c2:
        fig3, ax3 = plt.subplots()
        ax3.plot(t, z1, label="z1")
        ax3.plot(t, z2, label="z2")
        ax3.set_xlabel("Tiempo (s)"); ax3.set_ylabel("z(t)")
        ax3.set_title("Variable lenta z(t)")
        ax3.legend()
        st.pyplot(fig3, clear_figure=True)

with tab_isi:
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        if isi1.size:
            ax.hist(isi1, bins=min(40, max(5, isi1.size//2)))
        ax.set_xlabel("ISI N1 (s)"); ax.set_ylabel("Frecuencia")
        ax.set_title("Histograma ISI — N1")
        st.pyplot(fig, clear_figure=True)
    with c2:
        fig, ax = plt.subplots()
        if isi2.size:
            ax.hist(isi2, bins=min(40, max(5, isi2.size//2)))
        ax.set_xlabel("ISI N2 (s)"); ax.set_ylabel("Frecuencia")
        ax.set_title("Histograma ISI — N2")
        st.pyplot(fig, clear_figure=True)

with tab_return:
    c1, c2 = st.columns(2)
    Pn1, Pn1n = return_map(isi1)
    Pn2, Pn2n = return_map(isi2)
    with c1:
        fig, ax = plt.subplots()
        if Pn1.size:
            ax.scatter(Pn1, Pn1n, s=14, alpha=0.9)
            lims = [min(Pn1.min(), Pn1n.min()), max(Pn1.max(), Pn1n.max())]
            ax.plot(lims, lims, "--", lw=1)  # y=x
            ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel(r"$P_n$ (s)"); ax.set_ylabel(r"$P_{n+1}$ (s)")
        ax.set_title("Mapa de retorno — N1")
        st.pyplot(fig, clear_figure=True)
    with c2:
        fig, ax = plt.subplots()
        if Pn2.size:
            ax.scatter(Pn2, Pn2n, s=14, alpha=0.9)
            lims = [min(Pn2.min(), Pn2n.min()), max(Pn2.max(), Pn2n.max())]
            ax.plot(lims, lims, "--", lw=1)
            ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel(r"$P_n$ (s)"); ax.set_ylabel(r"$P_{n+1}$ (s)")
        ax.set_title("Mapa de retorno — N2")
        st.pyplot(fig, clear_figure=True)

with tab_phase:
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        ax.plot(x1, y1, lw=0.8)
        ax.set_xlabel("x1"); ax.set_ylabel("y1")
        ax.set_title("Retrato de fase N1 (x–y)")
        st.pyplot(fig, clear_figure=True)
    with c2:
        fig, ax = plt.subplots()
        ax.plot(x2, y2, lw=0.8)
        ax.set_xlabel("x2"); ax.set_ylabel("y2")
        ax.set_title("Retrato de fase N2 (x–y)")
        st.pyplot(fig, clear_figure=True)

with tab_xcorr:
    # Correlación cruzada de x1 con x2
    lags, corr = xcorr(x1, x2, max_lag_s=max_lag_s, dt=dt_eff, decim=xcorr_decim)
    fig, ax = plt.subplots()
    ax.plot(lags, corr)
    ax.set_xlabel("Retardo (s)")
    ax.set_ylabel("Correlación normalizada")
    ax.set_title(f"Correlación cruzada x1–x2 (decim={xcorr_decim}×)")
    st.pyplot(fig, clear_figure=True)

with tab_stats:
    c1, c2, c3 = st.columns([1.4, 1.4, 1.2])

    def nice_stats_table(title, stats):
        st.subheader(title)
        st.write(
            f"- **Spikes:** {stats['n_spikes']}"
            f"\n- **Firing rate:** {stats['rate_hz']:.3f} Hz"
            f"\n- **ISI mean ± std:** {stats['mean']:.4f} ± {stats['std']:.4f} s"
            f"\n- **CV:** {stats['cv']:.3f}"
            f"\n- **Mediana [Q25–Q75]:** {stats['median']:.4f} s [{stats['q25']:.4f}–{stats['q75']:.4f}]"
            f"\n- **Asimetría (skew):** {stats['skew']:.3f}"
            f"\n- **Curtosis (exceso):** {stats['kurt']:.3f}"
        )

    with c1:
        nice_stats_table("N1 — Estadísticos ISI", stats1)
    with c2:
        nice_stats_table("N2 — Estadísticos ISI", stats2)
    with c3:
        st.subheader("Resumen simulación")
        st.write(
            f"- Variante: **{variant}**"
            f"\n- Pasos: **{n_steps}** (dt={params.dt}s, decim={sample_every}× ⇒ dt_eff={dt_eff:.4f}s)"
            f"\n- Duración: **{t[-1]-t[0]:.2f} s**"
            f"\n- e={params.e}, u={params.u}, Esyn={params.Esyn}, Vfast={params.Vfast}, sfast={params.sfast}"
        )

st.success("Simulación completada. Ajusta parámetros en la barra lateral y vuelve a pulsar “Simular / Actualizar”.")
