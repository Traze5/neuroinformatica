# streamlit_rulkov_hco_invariantes.py
# Rulkov HCO (2 neuronas) + IDS (invariantes dinámicos secuenciales)
# Autor: Miguel Calderón + GPT-5 Thinking (asistente)
# Licencia: MIT
#
# Descripción
# ----------
# - Simula dos neuronas tipo Rulkov (mapa 2D) acopladas para formar un HCO (inhibición mutua),
#   con opción de acoplamiento eléctrico (gap) y ruido.
# - Extrae eventos (picos) y agrupa en ráfagas (bursts) para cada neurona.
# - Calcula intervalos ciclo-a-ciclo: Duración de ráfaga (B), IBI (inter-burst interval),
#   Período (T) y retardos cruzados (Delay_AB, Delay_BA).
# - Visualiza series temporales, raster de picos, fase (x vs y), histogramas y
#   gráficos de dispersión con regresión (candidatos a invariantes).
# - Incluye "presets" reproducibles.
#
# Referencias conceptuales (para la redacción del TFM; NO se usan aquí como citas formales):
# - Berbel, Latorre, Varona (2024, 2025): Invariantes dinámicos secuenciales en CPGs; rol de sinapsis químicas y eléctricas.
# - Rulkov (2002): Mapa 2D para spiking-bursting.
#
# Instrucciones
# ------------
# Ejecutar con:  streamlit run streamlit_rulkov_hco_invariantes.py
# Requisitos: streamlit, numpy, pandas, plotly, scipy (opcional)
#
# Notas de implementación
# -----------------------
# - Para robustez y velocidad, la detección de picos se basa en umbral sobre x y cruce ascendente.
# - La separación de ráfagas se determina por un umbral de ISI (gap_threshold); por defecto, 3× el ISI mediano intrarráfaga estimado.
# - El acoplamiento químico inhibitorio se implementa con una sigmoide presináptica S(x_pre) y un término (E_syn - x_post).
# - El acoplamiento eléctrico (opcional) usa un término g_el * (x_j - x_i) sobre la variable rápida x.
# - Se incluyen "presets" con configuraciones simétricas/asimétricas y con/sin gap para favorecer variabilidad + robustez.
#
import streamlit as st
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, List
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------
# Núcleo del modelo de Rulkov
# ---------------------------
@dataclass
class RulkovParams:
    alpha: float = 4.1       # Control de no linealidad (4.1–4.6 suele dar bursting)
    mu: float = 0.001        # Ritmo lento (0<mu<<1)
    sigma: float = -1.2      # Umbral lento (excitabilidad)
    I: float = 0.0           # Sesgo externo
    # Sinapsis química inhibidora i<-j
    g_inh_12: float = 0.35   # j=2 -> i=1
    g_inh_21: float = 0.35   # j=1 -> i=2
    E_syn: float = -1.5
    k_sig: float = 10.0      # pendiente sigmoide
    theta_sig: float = -0.1  # umbral sigmoide
    # Acoplamiento eléctrico (gap)
    g_el: float = 0.0        # bi-direccional simétrico
    # Ruido
    sigma_noise: float = 0.0 # desviación estándar del ruido gaussiano en x
    # Inicialización
    x1_0: float = -1.5
    y1_0: float = -2.0
    x2_0: float = -1.0
    y2_0: float = -2.2

def rulkov_update(x, y, p: RulkovParams, x_other=None):
    """
    Un paso del mapa de Rulkov para una neurona, con acoplamientos opcionales.
    Forma usada (común en literatura):
        x_{n+1} = alpha/(1 + x_n^2) + y_n + I + I_chem + I_el + ruido
        y_{n+1} = y_n - mu * (x_n - sigma)
    """
    # Sinapsis química desde la otra neurona
    I_chem = 0.0
    if x_other is not None:
        S_pre = 1.0 / (1.0 + np.exp(-p.k_sig * (x_other - p.theta_sig)))
        # corriente inhibidora (E_syn < x => hiperpolariza)
        I_chem = (p.E_syn - x) * S_pre

    # Acoplamiento eléctrico (gap)
    I_el = 0.0
    if x_other is not None and p.g_el > 0:
        I_el = (x_other - x)

    # Actualización
    xn1 = p.alpha / (1.0 + x*x) + y + p.I + p.g_el * I_el
    # Ganancias sinápticas dirigidas se aplican fuera (para 2 neuronas).
    yn1 = y - p.mu * (x - p.sigma)
    return xn1, yn1, I_chem

def simulate_rulkov_2neurons(steps: int, p: RulkovParams, seed: int = 0) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x1 = np.empty(steps); y1 = np.empty(steps)
    x2 = np.empty(steps); y2 = np.empty(steps)

    # init
    x1[0], y1[0], x2[0], y2[0] = p.x1_0, p.y1_0, p.x2_0, p.y2_0

    for n in range(steps-1):
        # Paso provisional sin aplicar g_inh (para obtener corrientes químicas dirigidas)
        x1n1, y1n1, Ichem_1_from2 = rulkov_update(x1[n], y1[n], p, x_other=x2[n])
        x2n1, y2n1, Ichem_2_from1 = rulkov_update(x2[n], y2[n], p, x_other=x1[n])

        # Aplicar inhibición dirigida (ganancias distintas por dirección)
        x1[n+1] = x1n1 + p.g_inh_12 * Ichem_1_from2 + (p.sigma_noise * rng.normal() if p.sigma_noise>0 else 0.0)
        y1[n+1] = y1n1
        x2[n+1] = x2n1 + p.g_inh_21 * Ichem_2_from1 + (p.sigma_noise * rng.normal() if p.sigma_noise>0 else 0.0)
        y2[n+1] = y2n1

    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

# ---------------------------
# Detección de picos y ráfagas
# ---------------------------

def detect_spikes(x: np.ndarray, thr: float = 0.0) -> np.ndarray:
    """Índices de cruce ascendente del umbral thr en la señal x."""
    above = x > thr
    crossings = np.where((~above[:-1]) & (above[1:]))[0] + 1
    return crossings


def group_bursts(spike_idx: np.ndarray, gap_threshold: int) -> List[Tuple[int,int]]:
    """
    Agrupa spikes en ráfagas: nueva ráfaga cuando ISI > gap_threshold.
    Retorna lista de pares (start_idx_in_series, end_idx_in_series) usando los índices de spikes.
    """
    if len(spike_idx) == 0:
        return []
    bursts = []
    start = spike_idx[0]
    for i in range(1, len(spike_idx)):
        if spike_idx[i] - spike_idx[i-1] > gap_threshold:
            end = spike_idx[i-1]
            bursts.append((start, end))
            start = spike_idx[i]
    bursts.append((start, spike_idx[-1]))
    return bursts


def estimate_gap_threshold(spike_idx: np.ndarray) -> int:
    """Heurística: 3× la mediana del ISI intrarráfaga estimado (si hay suficientes spikes)."""
    if len(spike_idx) < 3:
        return 30
    isi = np.diff(spike_idx)
    med = np.median(isi)
    # cap en rango razonable
    val = int(max(20, min(200, 3*med)))
    return val


def intervals_from_bursts(bursts: List[Tuple[int,int]]) -> pd.DataFrame:
    """Calcula B (duración), IBI y T a partir de los límites de ráfagas (índices de spikes)."""
    if len(bursts) < 2:
        return pd.DataFrame(columns=["burst_start","burst_end","B","IBI","T"])
    rows = []
    for k in range(len(bursts)-1):
        s0,e0 = bursts[k]
        s1,e1 = bursts[k+1]
        B = e0 - s0
        IBI = s1 - e0
        T = s1 - s0
        rows.append({"burst_start": s0, "burst_end": e0, "B": B, "IBI": IBI, "T": T})
    # última ráfaga sin siguiente: sólo B conocido
    s_last,e_last = bursts[-1]
    rows.append({"burst_start": s_last, "burst_end": e_last, "B": e_last - s_last, "IBI": np.nan, "T": np.nan})
    return pd.DataFrame(rows)


def build_sequence_pairs(bursts_A: List[Tuple[int,int]], bursts_B: List[Tuple[int,int]]) -> pd.DataFrame:
    """
    Define ciclos anclados a A: para cada ráfaga de A (k) busca la ráfaga de B cuyo inicio cae en (endA_k, startA_{k+1}).
    Calcula Delay_AB = startB - endA_k y Delay_BA = startA_{k+1} - endB.
    """
    rows=[]
    for k in range(len(bursts_A)-1):
        sA, eA = bursts_A[k]
        sA_next, eA_next = bursts_A[k+1]
        # elegir ráfaga B con inicio entre eA y sA_next
        candidates = [ (sB,eB) for (sB,eB) in bursts_B if (sB>eA and sB<sA_next) ]
        if len(candidates)==0:
            continue
        # asumir alternancia: tomar la primera
        sB, eB = candidates[0]
        Delay_AB = sB - eA
        Delay_BA = sA_next - eB
        rows.append({
            "A_burst_start": sA, "A_burst_end": eA, "A_next_start": sA_next,
            "B_burst_start": sB, "B_burst_end": eB,
            "Delay_AB": Delay_AB, "Delay_BA": Delay_BA,
            "Period_A": sA_next - sA
        })
    return pd.DataFrame(rows)


def linregress_np(x: np.ndarray, y: np.ndarray) -> Tuple[float,float,float]:
    """Regresión lineal simple (pendiente, intercepto, R^2)."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum()<2:
        return np.nan, np.nan, np.nan
    x1 = x[mask]; y1 = y[mask]
    A = np.vstack([x1, np.ones_like(x1)]).T
    # mínimos cuadrados
    m, b = np.linalg.lstsq(A, y1, rcond=None)[0]
    # R^2
    y_pred = m*x1 + b
    ss_res = np.sum((y1-y_pred)**2)
    ss_tot = np.sum((y1 - y1.mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot>0 else np.nan
    return float(m), float(b), float(r2)

# ---------------------------
# Presets
# ---------------------------
PRESETS: Dict[str, Dict] = {
    "Químico simétrico (variabilidad alta)": dict(
        alpha=4.2, mu=0.001, sigma=-1.2, I=0.0,
        g_inh_12=0.40, g_inh_21=0.40, g_el=0.0,
        sigma_noise=0.002, theta_sig=-0.1, k_sig=10.0
    ),
    "Químico asimétrico (coordinación)": dict(
        alpha=4.25, mu=0.001, sigma=-1.2, I=0.0,
        g_inh_12=0.45, g_inh_21=0.30, g_el=0.0,
        sigma_noise=0.0015, theta_sig=-0.1, k_sig=10.0
    ),
    "Mixto: inhibición + gap débil": dict(
        alpha=4.1, mu=0.001, sigma=-1.15, I=0.0,
        g_inh_12=0.35, g_inh_21=0.35, g_el=0.015,
        sigma_noise=0.001, theta_sig=-0.1, k_sig=10.0
    ),
    "Gap moderado (más regular)": dict(
        alpha=4.1, mu=0.001, sigma=-1.1, I=0.0,
        g_inh_12=0.30, g_inh_21=0.30, g_el=0.030,
        sigma_noise=0.0005, theta_sig=-0.1, k_sig=10.0
    ),
}

# ---------------------------
# UI de Streamlit
# ---------------------------
st.set_page_config(page_title="Rulkov HCO + IDS", layout="wide")
st.title("Rulkov HCO (2 neuronas) + IDS (invariantes)")

with st.sidebar:
    st.header("Configuración")
    preset_name = st.selectbox("Preset", list(PRESETS.keys()), index=1)
    steps = st.slider("Pasos de simulación", 2000, 20000, 6000, step=500)
    seed = st.number_input("Semilla RNG", 0, 10_000_000, 0, step=1)
    st.divider()
    st.subheader("Parámetros modelo")
    pvals = PRESETS[preset_name].copy()
    # Permitir override fino
    alpha = st.number_input("alpha", 3.5, 6.0, float(pvals["alpha"]), step=0.05)
    mu = st.number_input("mu", 1e-5, 0.01, float(pvals["mu"]), step=1e-4, format="%.5f")
    sigma = st.number_input("sigma", -2.0, 0.0, float(pvals["sigma"]), step=0.01)
    I = st.number_input("I (bias)", -1.0, 1.0, float(pvals["I"]), step=0.01)
    g12 = st.number_input("g_inh 2→1", 0.0, 1.0, float(pvals["g_inh_12"]), step=0.01)
    g21 = st.number_input("g_inh 1→2", 0.0, 1.0, float(pvals["g_inh_21"]), step=0.01)
    g_el = st.number_input("g_el (gap)", 0.0, 0.2, float(pvals["g_el"]), step=0.005)
    k_sig = st.number_input("k_sig (pendiente sigmoide)", 1.0, 40.0, float(pvals["k_sig"]), step=1.0)
    theta_sig = st.number_input("theta_sig (umbral sigmoide)", -2.0, 2.0, float(pvals["theta_sig"]), step=0.05)
    sigma_noise = st.number_input("ruido σ_x", 0.0, 0.05, float(pvals["sigma_noise"]), step=0.001, format="%.3f")

    st.subheader("Detección de ráfagas")
    thr = st.number_input("Umbral de pico (x)", -2.0, 2.0, 0.0, step=0.05)
    gap_manual = st.checkbox("Configurar gap_threshold manualmente", value=False)
    gap_thr = st.slider("gap_threshold (pasos)", 10, 400, 80, step=5, disabled=not gap_manual)

    st.subheader("Ploteo")
    zoom_ini = st.slider("Zoom inicio", 0, steps-100, 500, step=50)
    zoom_len = st.slider("Zoom ventana", 100, min(steps, 4000), 1200, step=50)

# Construir parámetros
params = RulkovParams(
    alpha=alpha, mu=mu, sigma=sigma, I=I,
    g_inh_12=g12, g_inh_21=g21, g_el=g_el,
    k_sig=k_sig, theta_sig=theta_sig, sigma_noise=sigma_noise
)

# Simular
sim = simulate_rulkov_2neurons(steps=steps, p=params, seed=int(seed))
x1, y1, x2, y2 = sim["x1"], sim["y1"], sim["x2"], sim["y2"]

# Detectar picos y ráfagas
spk1 = detect_spikes(x1, thr=thr)
spk2 = detect_spikes(x2, thr=thr)

gap1 = gap_thr if gap_manual else estimate_gap_threshold(spk1)
gap2 = gap_thr if gap_manual else estimate_gap_threshold(spk2)

bursts1 = group_bursts(spk1, gap_threshold=gap1)
bursts2 = group_bursts(spk2, gap_threshold=gap2)

df1 = intervals_from_bursts(bursts1)
df2 = intervals_from_bursts(bursts2)
df_seq = build_sequence_pairs(bursts1, bursts2)

# ---------------------------------
# Gráficos principales (tiempo real)
# ---------------------------------

tab_ts, tab_raster, tab_phase, tab_hist, tab_ids = st.tabs(
    ["Series temporales", "Raster de picos", "Fase (x–y)", "Histogramas", "IDS (invariantes)"]
)

with tab_ts:
    st.caption("Serie completa (x) y ventana de zoom ajustable.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=x1, mode="lines", name="x1", line=dict(width=1)))
    fig.add_trace(go.Scatter(y=x2, mode="lines", name="x2", line=dict(width=1)))
    st.plotly_chart(fig, use_container_width=True)
    # zoom
    a = zoom_ini; b = min(steps, zoom_ini+zoom_len)
    figz = go.Figure()
    figz.add_trace(go.Scatter(x=np.arange(a,b), y=x1[a:b], mode="lines", name="x1", line=dict(width=1)))
    figz.add_trace(go.Scatter(x=np.arange(a,b), y=x2[a:b], mode="lines", name="x2", line=dict(width=1)))
    st.plotly_chart(figz, use_container_width=True)

with tab_raster:
    st.caption("Eventos de pico detectados (umbral y cruce ascendente).")
    figR = go.Figure()
    figR.add_trace(go.Scatter(x=spk1, y=np.ones_like(spk1), mode="markers", name="N1", marker=dict(symbol="line-ns-open")))
    figR.add_trace(go.Scatter(x=spk2, y=2*np.ones_like(spk2), mode="markers", name="N2", marker=dict(symbol="line-ns-open")))
    figR.update_yaxes(tickvals=[1,2], ticktext=["N1","N2"], range=[0.5,2.5])
    st.plotly_chart(figR, use_container_width=True)

with tab_phase:
    st.caption("Retrato de fase (x vs y).")
    figP = go.Figure()
    figP.add_trace(go.Scatter(x=x1, y=y1, mode="lines", name="N1", opacity=0.8))
    figP.add_trace(go.Scatter(x=x2, y=y2, mode="lines", name="N2", opacity=0.8))
    st.plotly_chart(figP, use_container_width=True)

with tab_hist:
    st.caption("Distribuciones de intervalos (por neurona).")
    if not df1.empty:
        st.subheader("Neurona 1")
        st.plotly_chart(px.histogram(df1.dropna(), x="B", nbins=30, title="Burst duration (B) N1"), use_container_width=True)
        st.plotly_chart(px.histogram(df1.dropna(), x="IBI", nbins=30, title="IBI N1"), use_container_width=True)
        st.plotly_chart(px.histogram(df1.dropna(), x="T", nbins=30, title="Period (T) N1"), use_container_width=True)
    if not df2.empty:
        st.subheader("Neurona 2")
        st.plotly_chart(px.histogram(df2.dropna(), x="B", nbins=30, title="Burst duration (B) N2"), use_container_width=True)
        st.plotly_chart(px.histogram(df2.dropna(), x="IBI", nbins=30, title="IBI N2"), use_container_width=True)
        st.plotly_chart(px.histogram(df2.dropna(), x="T", nbins=30, title="Period (T) N2"), use_container_width=True)

with tab_ids:
    st.caption("Relaciones candidato a invariantes (lineales).")
    cols = st.columns(2)

    # Candidatos clásicos en HCO: retardo cruzado vs periodo
    if not df_seq.empty and not df1.empty and not df2.empty:
        # Convertir a DataFrame con columnas de interés
        A_T = []
        A_B = []
        for k in range(min(len(df1)-1, len(df_seq))):
            A_T.append(df_seq.loc[k, "Period_A"])
            A_B.append(df1.loc[k, "B"])
        A_T = np.array(A_T, dtype=float)
        A_B = np.array(A_B, dtype=float)

        Delay_AB = df_seq["Delay_AB"].values.astype(float)
        Delay_BA = df_seq["Delay_BA"].values.astype(float)

        # Regressiones
        m1,b1,r21 = linregress_np(A_T, Delay_AB)
        m2,b2,r22 = linregress_np(A_T, Delay_BA)
        m3,b3,r23 = linregress_np(A_T, A_B)

        with cols[0]:
            st.markdown(f"**Delay_AB vs Period_A**  \n slope={m1:.3f}, R²={r21:.3f}")
            fig1 = px.scatter(x=A_T, y=Delay_AB, labels={'x':'Period_A','y':'Delay_AB'})
            if np.isfinite(m1):
                xs = np.linspace(np.nanmin(A_T), np.nanmax(A_T), 50)
                fig1.add_trace(go.Scatter(x=xs, y=m1*xs+b1, mode="lines", name="fit"))
            st.plotly_chart(fig1, use_container_width=True)

        with cols[1]:
            st.markdown(f"**Delay_BA vs Period_A**  \n slope={m2:.3f}, R²={r22:.3f}")
            fig2 = px.scatter(x=A_T, y=Delay_BA, labels={'x':'Period_A','y':'Delay_BA'})
            if np.isfinite(m2):
                xs = np.linspace(np.nanmin(A_T), np.nanmax(A_T), 50)
                fig2.add_trace(go.Scatter(x=xs, y=m2*xs+b2, mode="lines", name="fit"))
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown(f"**B_A vs Period_A**  \n slope={m3:.3f}, R²={r23:.3f}")
        fig3 = px.scatter(x=A_T, y=A_B, labels={'x':'Period_A','y':'B_A'})
        if np.isfinite(m3):
            xs = np.linspace(np.nanmin(A_T), np.nanmax(A_T), 50)
            fig3.add_trace(go.Scatter(x=xs, y=m3*xs+b3, mode="lines", name="fit"))
        st.plotly_chart(fig3, use_container_width=True)

        st.divider()
        st.subheader("Pairplot rápido (A)")
        dfA = pd.DataFrame({"Period_A":A_T, "B_A":A_B, "Delay_AB":Delay_AB, "Delay_BA":Delay_BA})
        st.plotly_chart(px.scatter_matrix(dfA), use_container_width=True)
    else:
        st.info("Aún no hay suficientes ráfagas para caracterizar IDS. Ajusta parámetros o ejecuta más pasos.")

# ---------------------------
# Datos y exportación
# ---------------------------

st.divider()
colA, colB, colC = st.columns([1,1,1])
with colA:
    st.subheader("Intervalos N1")
    st.dataframe(df1.head(50))
    st.download_button("Descargar N1 (CSV)", data=df1.to_csv(index=False).encode("utf-8"), file_name="intervalos_N1.csv", mime="text/csv")
with colB:
    st.subheader("Intervalos N2")
    st.dataframe(df2.head(50))
    st.download_button("Descargar N2 (CSV)", data=df2.to_csv(index=False).encode("utf-8"), file_name="intervalos_N2.csv", mime="text/csv")
with colC:
    st.subheader("Secuencias (A ancla)")
    st.dataframe(df_seq.head(50))
    st.download_button("Descargar Secuencias (CSV)", data=df_seq.to_csv(index=False).encode("utf-8"), file_name="secuencias_A.csv", mime="text/csv")

st.caption("Tip: usa 'Químico asimétrico' o 'Mixto' para ver variabilidad + coordinación (buen terreno para invariantes).")
