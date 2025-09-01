# -*- coding: utf-8 -*-
# HR (2 neuronas): series + ciclos (paper, con 1er/último spike por cruces) + tabla + pairplots

import numpy as np, pandas as pd, streamlit as st, warnings
from dataclasses import dataclass
from plotly import graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="HR (2 neuronas) — invariantes y pairplots", layout="wide")

# ===== SciPy (LSODA) =====
try:
    from scipy.integrate import odeint as _odeint
    HAVE_SCIPY = True
    try:
        from scipy.integrate._odepack_py import ODEintWarning
    except Exception:
        class ODEintWarning(UserWarning): ...
except Exception:
    HAVE_SCIPY = False
    class ODEintWarning(UserWarning): ...

# ===== Utilidades =====
COL = {"X":"#E74C3C","Y":"#3498DB","NEU":"#7F8C8D"}
def _anno(x,y,text):
    return dict(x=x,y=y,text=text,xref="x",yref="y",showarrow=False,
                bgcolor="rgba(0,0,0,0.72)",bordercolor="white",borderwidth=1.1,borderpad=3,
                font=dict(size=13,color="white"))

# ===== Modelo HR =====
@dataclass
class HRParams:
    e1:float=3.282; e2:float=3.282
    mu:float=0.0021; s1:float=1.0; s2:float=1.0; v1:float=0.1; v2:float=0.1
@dataclass
class SynChemSigm:
    g_syn:float=0.35; theta:float=-0.25; k:float=10.0; E_syn:float=-2.0
@dataclass
class SynChemCPP:
    g_fast:float=0.10; Esyn:float=-1.8; Vfast:float=-1.1; sfast:float=0.2
@dataclass
class SynElec:
    g_el:float=0.05

# Micro-variabilidad lenta (ligera)
DRIFT_A = 0.012
DRIFT_T = 600.0

def _exp_clip(a): return np.exp(np.clip(a, -60.0, 60.0))
def sigm_stable(x, th, k): return 1.0/(1.0 + _exp_clip(-k*(x - th)))

def rhs_hr(y, t, prm, mode, sc, cc, se):
    e1_eff = prm.e1
    e2_eff = prm.e2 + (DRIFT_A*np.sin(2*np.pi*t/max(DRIFT_T,1e-6)))
    x1,y1,z1,x2,y2,z2 = [float(v) for v in y]
    if mode=="quimica_sigmoidal":
        s1s = sigm_stable(x1, sc.theta, sc.k); s2s = sigm_stable(x2, sc.theta, sc.k)
        Isyn1 = sc.g_syn*s2s*(sc.E_syn-x1); Isyn2 = sc.g_syn*s1s*(sc.E_syn-x2); Iel1=Iel2=0.0
    elif mode=="quimica_cpp":
        Isyn1 = -cc.g_fast*(x1-cc.Esyn)/(1.0 + _exp_clip(cc.sfast*(cc.Vfast-x2)))
        Isyn2 = -cc.g_fast*(x2-cc.Esyn)/(1.0 + _exp_clip(cc.sfast*(cc.Vfast-x1))); Iel1=Iel2=0.0
    elif mode=="electrica":
        Iel1 = se.g_el*(x2-x1); Iel2 = se.g_el*(x1-x2); Isyn1=Isyn2=0.0
    else:
        Isyn1=Isyn2=Iel1=Iel2=0.0
    dx1 = y1 + 3*x1**2 - x1**3 - z1 + e1_eff + Isyn1 + Iel1
    dy1 = 1 - 5*x1**2 - y1
    dz1 = prm.mu * (-prm.v1*z1 + prm.s1*(x1 + 1.6))
    dx2 = y2 + 3*x2**2 - x2**3 - z2 + e2_eff + Isyn2 + Iel2
    dy2 = 1 - 5*x2**2 - y2
    dz2 = prm.mu * (-prm.v2*z2 + prm.s2*(x2 + 1.6))
    return np.array([dx1,dy1,dz1, dx2,dy2,dz2], float)

# ===== Integración =====
@st.cache_data(show_spinner=False)
def simulate(mode, prm_dict, sc_dict, cc_dict, se_dict,
             y0, nsteps, dt, decim_evt, burn_in_s, use_lsoda=True):
    prm = HRParams(**prm_dict); sc  = SynChemSigm(**sc_dict)
    cc  = SynChemCPP(**cc_dict); se  = SynElec(**se_dict)
    nsteps=int(nsteps); dt=float(dt); step=int(max(1,decim_evt))
    t_full=np.linspace(0.0,nsteps*dt,nsteps+1)
    t_out=t_full[::step]  # malla de salida/detección

    def _rhs(Y, tt, prm=prm, mode=mode, sc=sc, cc=cc, se=se): return rhs_hr(Y, tt, prm, mode, sc, cc, se)

    def _rk4_on_grid():
        y=np.empty((len(t_out),len(y0)),float); y[0]=np.array(y0,float)
        for i in range(len(t_out)-1):
            h=t_out[i+1]-t_out[i]
            k1=_rhs(y[i],t_out[i]); k2=_rhs(y[i]+0.5*h*k1,t_out[i]+0.5*h)
            k3=_rhs(y[i]+0.5*h*k2,t_out[i]+0.5*h); k4=_rhs(y[i]+h*k3,t_out[i]+h)
            y[i+1]=y[i]+(h/6.0)*(k1+2*k2+2*k3+k4)
        return y

    if use_lsoda and HAVE_SCIPY:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ODEintWarning)
                sol_out=_odeint(_rhs,np.array(y0,float),t_out,atol=1e-6,rtol=1e-6,mxstep=50000)
        except Exception:
            warnings.warn("LSODA falló; uso RK4 sobre la malla decimada.")
            sol_out=_rk4_on_grid()
    else:
        sol_out=_rk4_on_grid()

    if burn_in_s>0:
        skip=int(np.clip(np.floor(burn_in_s/(dt*step)),0,len(t_out)-2))
        t_out=t_out[skip:]; sol_out=sol_out[skip:]
    return t_out,sol_out

# ===== Ráfagas (umbral bajo) =====
def detect_bursts_interp(t,x,v_th=-0.60,min_on=0.10,min_off=0.05):
    t=np.asarray(t); x=np.asarray(x); up=[]; dn=[]
    for i in range(1,len(t)):
        if x[i-1]<=v_th and x[i]>v_th:
            frac=(v_th-x[i-1])/(x[i]-x[i-1]+1e-12); up.append(t[i-1]+frac*(t[i]-t[i-1]))
        if x[i-1]>v_th and x[i]<=v_th:
            frac=(v_th-x[i-1])/(x[i]-x[i-1]+1e-12); dn.append(t[i-1]+frac*(t[i]-t[i-1]))
    bursts=[]; i=j=0; last_off=-1e9
    while i<len(up) and j<len(dn):
        if up[i]<=dn[j]:
            on=up[i]
            while j<len(dn) and (dn[j]-on)<min_on: j+=1
            if j>=len(dn): break
            off=dn[j]
            if (on-last_off)>=min_off: bursts.append((on,off)); last_off=off
            i+=1; j+=1
        else: j+=1
    return bursts

# ===== 1er y último spike por cruces con interpolación (robusto + fallback) =====
def _cross_time(t0,x0,t1,x1,thr):
    den = (x1 - x0)
    if abs(den) < 1e-12: return t0
    a = (thr - x0) / den
    a = np.clip(a, 0.0, 1.0)
    return t0 + a*(t1 - t0)

def spike_edges_in_bursts_cross(t, x, bursts, v_spike=0.0):
    """
    Para cada burst [on, off] devuelve (first_spike, last_spike) como:
    - first: primer cruce ascendente de v_spike
    - last : último cruce descendente de v_spike
    Si no hay cruces, intenta umbral adaptativo; si no, usa máximo local; si no, on/off.
    """
    t=np.asarray(t); x=np.asarray(x)
    first=[]; last=[]
    for (a,b) in bursts:
        m=(t>=a) & (t<=b)
        tt=t[m]; xx=x[m]
        if tt.size<2:
            first.append(a); last.append(b); continue

        def _edges_for(thr):
            above = xx >= thr
            if not above.any(): return None
            idx = np.where(above)[0]
            i0, i1 = int(idx[0]), int(idx[-1])
            # primer cruce ascendente
            if i0>0:
                t_first = _cross_time(tt[i0-1], xx[i0-1], tt[i0], xx[i0], thr)
            else:
                t_first = tt[i0]
            # último cruce descendente
            if i1 < len(xx)-1:
                t_last = _cross_time(tt[i1], xx[i1], tt[i1+1], xx[i1+1], thr)
            else:
                t_last = tt[i1]
            return t_first, t_last

        out = _edges_for(v_spike)
        if out is None:
            xm = float(np.median(xx)); xM = float(np.max(xx))
            thr = xm + 0.4*(xM - xm)
            out = _edges_for(thr)
        if out is None:
            kmax=int(np.nanargmax(xx)); tpk=tt[kmax]
            first.append(tpk); last.append(tpk)
        else:
            first.append(out[0]); last.append(out[1])
    return np.array(first,float), np.array(last,float)

# ===== Ciclos (paper + spikes) =====
def build_cycles_Xleader(burstsX, burstsY, t, x, y, v_spike=0.0, tmin=None, tmax=None):
    if tmin is not None or tmax is not None:
        def _crop(bb):
            out=[]
            for (on,off) in bb:
                if (tmin is not None and off<=tmin) or (tmax is not None and on>=tmax): continue
                out.append((on,off))
            return out
        burstsX=_crop(burstsX); burstsY=_crop(burstsY)
    if len(burstsX)<3 or len(burstsY)<2: return []

    X_on  = np.array([on  for on,off in burstsX], float)
    X_off = np.array([off for on,off in burstsX], float)
    Y_on  = np.array([on  for on,off in burstsY], float)
    Y_off = np.array([off for on,off in burstsY], float)

    X_first, X_last = spike_edges_in_bursts_cross(t, x, burstsX, v_spike=v_spike)
    Y_first, Y_last = spike_edges_in_bursts_cross(t, y, burstsY, v_spike=v_spike)

    def pick_Y_in(a,b):
        j=np.searchsorted(Y_on, a, side="left")
        if j<len(Y_on) and Y_on[j] < b: return j
        jl=np.searchsorted(Y_on, a, side="right")-1
        if 0<=jl<len(Y_on) and Y_off[jl] > a: return jl
        return j if j<len(Y_on) else None

    cycles=[]
    for i in range(len(X_on)-2):
        x0, xf0 = X_on[i],   X_off[i]
        x1, xf1 = X_on[i+1], X_off[i+1]
        x2      = X_on[i+2]
        if (tmin is not None and x1 <= tmin) or (tmax is not None and x1 >= tmax): continue
        j0 = pick_Y_in(x0, x1); j1 = pick_Y_in(x1, x2)
        if j0 is None or j1 is None: continue
        if Y_on[j1] <= Y_on[j0]:
            if j1+1 < len(Y_on): j1 += 1
            else: continue
        cycles.append(dict(
            # sombreado (umbral bajo)
            x_on0=x0, x_off0=xf0, x_on1=x1, x_off1=xf1, x_on2=x2,
            y0_on=Y_on[j0], y0_off=Y_off[j0], y1_on=Y_on[j1], y1_off=Y_off[j1],
            # spikes (altos)
            X1_first=X_first[i],   X1_last=X_last[i],
            X2_first=X_first[i+1], X2_last=X_last[i+1],
            Y1_first=Y_first[j0],  Y1_last=Y_last[j0],
            Y2_first=Y_first[j1],  Y2_last=Y_last[j1],
        ))
    return cycles

# ===== Métricas (paper con spikes) =====
def metrics_from_cycle(c):
    P_X = c["X2_first"] - c["X1_first"]
    P_Y = c["Y2_first"] - c["Y1_first"]
    B_X = c["X2_last"]  - c["X2_first"]
    B_Y = c["Y1_last"]  - c["Y1_first"]
    IBI_X = c["X2_first"] - c["X1_last"]
    IBI_Y = c["Y2_first"] - c["Y1_last"]
    I_YX = c["X2_first"] - c["Y1_first"]
    D_YX = c["X2_first"] - c["Y1_last"]
    I_XY = c["Y2_first"] - c["X2_first"]
    D_XY = c["Y2_first"] - c["X2_last"]
    return dict(P_X=P_X, P_Y=P_Y, B_X=B_X, B_Y=B_Y,
                IBI_X=IBI_X, IBI_Y=IBI_Y, I_YX=I_YX, D_YX=D_YX, I_XY=I_XY, D_XY=D_XY)

def cycles_dataframe(cycles):
    rows=[]
    for c in cycles:
        m=metrics_from_cycle(c)
        rows.append({
            "Periodo X": m["P_X"], "Hiperpol. X (IBI)": m["IBI_X"], "Burst X": m["B_X"],
            "Periodo Y": m["P_Y"], "Hiperpol. Y (IBI)": m["IBI_Y"], "Burst Y": m["B_Y"],
            "Intervalo Y→X": m["I_YX"], "Retardo Y→X": m["D_YX"],
            "Intervalo X→Y": m["I_XY"], "Retardo X→Y": m["D_XY"],
        })
    return pd.DataFrame(rows)

# ===== Pairgrid Plotly =====
def plotly_pairgrid(df, cols, corner=False, nbins=24, marker_size=3, opacity=0.7,
                    height_per_row=190, share_axes=True):
    cols=[c for c in cols if c in df.columns]
    df=df[cols].replace([np.inf,-np.inf],np.nan).dropna(how="any")
    m=len(cols)
    if m==0 or df.shape[0]==0: return go.Figure()
    rng, data = {}, {}; rs=np.random.RandomState(0)
    for c in cols:
        x=df[c].to_numpy(float)
        spread=float(np.nanmax(x)-np.nanmin(x)) if x.size else 0.0
        std=float(np.nanstd(x)) if x.size else 0.0
        if not np.isfinite(std) or std<1e-9 or spread<1e-9:
            scale=max(1e-3, abs(float(np.nanmean(x)))*1e-3, spread*0.01)
            x=x+rs.normal(0.0,scale,size=x.shape)
        data[c]=x
        lo=float(np.nanmin(x)); hi=float(np.nanmax(x))
        d=hi-lo; pad=0.02*d if d>0 else max(1e-3, abs(hi if hi!=0 else 1.0)*0.02)
        rng[c]=[lo-pad, hi+pad]
    fig=make_subplots(rows=m, cols=m, shared_xaxes=False, shared_yaxes=False,
                      horizontal_spacing=0.03, vertical_spacing=0.05)
    for i in range(m):
        for j in range(m):
            if corner and j>i: continue
            xi,yj=cols[j],cols[i]
            if i==j:
                fig.add_trace(go.Histogram(x=data[xi], nbinsx=int(nbins), marker_line_width=0, showlegend=False),
                              row=i+1,col=j+1)
            else:
                fig.add_trace(go.Scatter(x=data[xi], y=data[yj], mode="markers",
                                         marker=dict(size=marker_size, opacity=opacity),
                                         hovertemplate="%{x:.4g}, %{y:.4g}<extra></extra>",
                                         showlegend=False),
                              row=i+1,col=j+1)
            if i==m-1: fig.update_xaxes(title_text=xi, row=i+1, col=j+1)
            if j==0:   fig.update_yaxes(title_text=yj, row=i+1, col=j+1)
            if share_axes:
                fig.update_xaxes(range=rng[xi], row=i+1, col=j+1)
                fig.update_yaxes(range=rng[yj], row=i+1, col=j+1)
    fig.update_layout(height=max(360,int(height_per_row*m)), margin=dict(l=10,r=10,t=18,b=10),
                      plot_bgcolor="#111418", paper_bgcolor="#111418",
                      font=dict(color="#E6E6E6"), bargap=0.02)
    for ax in fig.layout:
        if ax.startswith("xaxis") or ax.startswith("yaxis"):
            fig.layout[ax].showgrid=True; fig.layout[ax].gridcolor="#333A40"
            fig.layout[ax].zeroline=False; fig.layout[ax].linecolor="#70757a"; fig.layout[ax].tickcolor="#9aa0a6"
    return fig

# ======================== UI ========================
st.title("Hindmarsh–Rose (2 neuronas) — invariantes por ciclo")

with st.sidebar:
    st.header("Configuración")
    syn_mode=st.radio("Sinapsis",["Química (sigmoidal, HCO)","Química (núcleo C)","Eléctrica (difusiva)"])
    mode={"Química (sigmoidal, HCO)":"quimica_sigmoidal",
          "Química (núcleo C)":"quimica_cpp",
          "Eléctrica (difusiva)":"electrica"}[syn_mode]

    nsteps=st.number_input("Nº de pasos (TIME)", min_value=100_000, value=1_200_000, step=100_000)
    dt = st.number_input("dt (s)", min_value=0.0002, max_value=0.01, value=0.001, step=0.0002, format="%.4f")
    gamma = st.number_input("Escala temporal γ (solo visualización/métricas)", min_value=0.5, max_value=40.0, value=18.0, step=0.5)
    decim_evt=st.number_input("Muestreo para detección (cada N pasos)", min_value=1, max_value=10_000, value=60, step=5)
    burn_in_s = st.number_input("Descartar transitorio (s, tiempo físico)", min_value=0.0, value=0.0, step=1.0)

    st.subheader("Parámetros neurales")
    e1=st.number_input("I1 (N1)",value=3.282,step=0.001,format="%.3f")
    e2=st.number_input("I2 (N2)",value=3.282,step=0.001,format="%.3f")
    mu=st.number_input("u (escala lenta)",value=0.0021,step=0.0001,format="%.4f")
    s1=st.number_input("s1",value=1.0,step=0.1); s2=st.number_input("s2",value=1.0,step=0.1)
    v1=st.number_input("v1",value=0.1,step=0.01); v2=st.number_input("v2",value=0.1,step=0.01)

    if mode=="quimica_sigmoidal":
        st.subheader("Química sigmoidal")
        g_syn=st.number_input("g_syn",value=0.35,step=0.01)
        theta=st.number_input("θ",value=-0.25,step=0.01)
        kk=st.number_input("k",value=10.0,step=0.5)
        Esy=st.number_input("E_syn",value=-2.0,step=0.1)
        g_fast=0.10; Esy_c=-1.8; Vfast=-1.1; sfast=0.2
    elif mode=="quimica_cpp":
        st.subheader("Química (núcleo C)")
        g_fast=st.number_input("g_fast (C)",value=0.10,step=0.01)
        Esy_c=st.number_input("Esyn (C)",value=-1.8,step=0.1)
        Vfast=st.number_input("Vfast (C)",value=-1.1,step=0.1)
        sfast=st.number_input("sfast (C)",value=0.2,step=0.01)
        g_syn=0.35; theta=-0.25; kk=10.0; Esy=-2.0
    else:
        st.subheader("Eléctrica (difusiva)")
        g_el=st.number_input("g_el",value=0.05,step=0.01)
        g_syn=0.35; theta=-0.25; kk=10.0; Esy=-2.0
        g_fast=0.10; Esy_c=-1.8; Vfast=-1.1; sfast=0.2

    st.subheader("Detección de ráfagas")
    v_th=st.number_input("Umbral bajo v_th",value=-0.60,step=0.01)
    min_on=st.number_input("Duración mínima ON (s)",value=0.10,step=0.01)
    min_off=st.number_input("Duración mínima OFF (s)",value=0.05,step=0.01)

    st.subheader("Picos dentro del burst (1er/último spike)")
    v_spike = st.number_input("Umbral alto v_spike", value=0.00, step=0.05, format="%.2f")

    st.subheader("Integrador")
    use_lsoda = st.radio("Método",["LSODA (recomendado)","RK4 (fallback)"],index=0,horizontal=True).startswith("LSODA")

# Empaquetado
prm_dict=dict(e1=float(e1),e2=float(e2),mu=float(mu),s1=float(s1),s2=float(s2),v1=float(v1),v2=float(v2))
sigm_par=dict(g_syn=float(locals().get("g_syn",0.35)),theta=float(locals().get("theta",-0.25)),
              k=float(locals().get("kk",10.0)),E_syn=float(locals().get("Esy",-2.0)))
cpp_par =dict(g_fast=float(locals().get("g_fast",0.10)),Esyn=float(locals().get("Esy_c",-1.8)),
              Vfast=float(locals().get("Vfast",-1.1)),sfast=float(locals().get("sfast",0.2)))
elec_par=dict(g_el=float(locals().get("g_el",0.0)))
y0=np.array([-0.915325,-3.208968,3.350784,-1.307949,-7.580493,3.068898],float)

# Simulación
t_phys, sol_phys = simulate(mode,prm_dict,sigm_par,cpp_par,elec_par,
                            y0,int(nsteps),float(dt),int(decim_evt),float(burn_in_s),use_lsoda)
x1, x2 = sol_phys[:,0], sol_phys[:,3]
gamma = float(locals().get("gamma",18.0))
t_disp = t_phys / gamma

# ===== Detección previa para pintar spikes en las series =====
burstsX = detect_bursts_interp(t_phys, x1, v_th=float(v_th), min_on=float(min_on), min_off=float(min_off))
burstsY = detect_bursts_interp(t_phys, x2, v_th=float(v_th), min_on=float(min_on), min_off=float(min_off))
Xf, Xl = spike_edges_in_bursts_cross(t_phys, x1, burstsX, v_spike=float(v_spike))
Yf, Yl = spike_edges_in_bursts_cross(t_phys, x2, burstsY, v_spike=float(v_spike))

# ===== Series + marcadores de spikes (arriba) =====
st.header("Series temporales con 1er/último spike por burst")
fig_over=go.Figure()
fig_over.add_trace(go.Scatter(x=t_disp,y=x1,name="X (N1)",line=dict(color=COL["X"],width=1.8)))
fig_over.add_trace(go.Scatter(x=t_disp,y=x2,name="Y (N2)",line=dict(color=COL["Y"],width=1.8)))
# X first / last
if Xf.size:
    fig_over.add_trace(go.Scatter(x=Xf/gamma, y=np.interp(Xf,t_phys,x1), mode="markers",
                                  marker=dict(symbol="triangle-up", size=7, color="#FF9AA2"),
                                  name="X primer spike"))
if Xl.size:
    fig_over.add_trace(go.Scatter(x=Xl/gamma, y=np.interp(Xl,t_phys,x1), mode="markers",
                                  marker=dict(symbol="triangle-down", size=7, color="#C0392B"),
                                  name="X último spike"))
# Y first / last
if Yf.size:
    fig_over.add_trace(go.Scatter(x=Yf/gamma, y=np.interp(Yf,t_phys,x2), mode="markers",
                                  marker=dict(symbol="triangle-up", size=7, color="#9AD0FF"),
                                  name="Y primer spike"))
if Yl.size:
    fig_over.add_trace(go.Scatter(x=Yl/gamma, y=np.interp(Yl,t_phys,x2), mode="markers",
                                  marker=dict(symbol="triangle-down", size=7, color="#2162B0"),
                                  name="Y último spike"))
fig_over.update_layout(height=330,xaxis_title="tiempo (s)",yaxis_title="x",
                       margin=dict(l=10,r=10,b=10,t=10),
                       legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1.0))
st.plotly_chart(fig_over,use_container_width=True)

# ===== Ventana y ciclos =====
st.subheader("Ventana temporal para detección de ciclos (se muestra en t/γ)")
t0, t1 = float(t_disp[0]), float(t_disp[-1])
step_slider = max((t1-t0)/1000.0, 1e-6)
win_disp = st.slider("Selecciona ventana", min_value=t0, max_value=t1, value=(t0, t1),
                     step=step_slider, format="%.2f")
win_phys = (win_disp[0]*gamma, win_disp[1]*gamma)

# ciclos con spikes (paper)
cycles = build_cycles_Xleader(burstsX, burstsY, t_phys, x1, x2,
                              v_spike=float(v_spike),
                              tmin=win_phys[0], tmax=win_phys[1])

# escalar a t/γ
def _scale_cycle(c, s):
    out={}
    for k,v in c.items():
        try:
            vv=float(v); out[k]= (vv/s) if np.isfinite(vv) else vv
        except Exception:
            out[k]=v
    return out
cycles_disp=[_scale_cycle(c, gamma) for c in cycles]
df_cycles = cycles_dataframe(cycles_disp)

# ===== Diagrama =====
st.header("Diagrama de ciclo (definición del paper)")
st.caption(f"Ráfagas detectadas: X={len(burstsX)} | Y={len(burstsY)} | Ciclos={len(cycles_disp)}")
if not cycles_disp:
    st.info("No hay ciclos válidos en la ventana.")
else:
    idx = 1 if len(cycles_disp)==1 else st.slider("Ciclo",1,len(cycles_disp),len(cycles_disp),step=1)
    cyc = cycles_disp[idx-1]; m=metrics_from_cycle(cyc)

    P = cyc["X2_first"] - cyc["X1_first"]
    tmin=max(t_disp[0], cyc["X1_first"]-0.2*P)
    tmax=min(t_disp[-1], max(cyc["X2_first"]+0.3*P, cyc["Y2_first"]+0.1*P))
    mm=(t_disp>=tmin)&(t_disp<=tmax)

    fig=go.Figure()
    fig.add_trace(go.Scatter(x=t_disp[mm],y=x1[mm],name="X (N1)",line=dict(color=COL["X"],width=2.0)))
    fig.add_trace(go.Scatter(x=t_disp[mm],y=x2[mm],name="Y (N2)",line=dict(color=COL["Y"],width=2.0)))

    # sombreado (umbral bajo)
    fig.add_vrect(x0=cyc["x_on0"],x1=cyc["x_off0"],fillcolor=COL["X"],opacity=0.16,line_width=0)
    fig.add_vrect(x0=cyc["y0_on"],x1=cyc["y0_off"],fillcolor=COL["Y"],opacity=0.16,line_width=0)
    fig.add_vrect(x0=cyc["x_on1"],x1=cyc["x_off1"],fillcolor=COL["X"],opacity=0.10,line_width=0)
    fig.add_vrect(x0=cyc["y1_on"],x1=cyc["y1_off"],fillcolor=COL["Y"],opacity=0.10,line_width=0)

    # líneas de métrica (spikes)
    y_top=max(np.max(x1[mm]),np.max(x2[mm])); y_bot=min(np.min(x1[mm]),np.min(x2[mm])); dy=y_top-y_bot; yb=y_bot-0.10*dy
    def _span(a,b,y,txt,col):
        a=max(float(a),float(tmin)); b=min(float(b),float(tmax))
        if np.isfinite(a) and np.isfinite(b) and b>a:
            fig.add_shape(type="line",x0=a,x1=b,y0=y,y1=y,line=dict(color=col,width=3))
            fig.add_annotation(**_anno((a+b)/2.0,y,txt))

    _span(cyc["X1_first"], cyc["X2_first"], y_top+0.12*dy, f"Periodo X = {m['P_X']:.3f} s", COL["X"])
    _span(cyc["X1_last"],  cyc["X2_first"], y_top+0.06*dy, f"IBI X = {m['IBI_X']:.3f} s", COL["X"])
    _span(cyc["Y1_first"], cyc["Y2_first"], y_top+0.00*dy, f"Periodo Y = {m['P_Y']:.3f} s", COL["Y"])

    _span(cyc["Y1_first"], cyc["Y1_last"], yb-0.00*dy, f"Y burst = {m['B_Y']:.3f} s", COL["Y"])
    _span(cyc["Y1_first"], cyc["X2_first"], yb-0.06*dy, f"Y→X intervalo = {m['I_YX']:.3f} s", "#2C3E50")
    _span(cyc["Y1_last"],  cyc["X2_first"], yb-0.12*dy, f"Y→X retardo = {m['D_YX']:.3f} s", COL["NEU"])

    _span(cyc["X2_first"], cyc["X2_last"], yb-0.20*dy, f"X burst = {m['B_X']:.3f} s", COL["X"])
    _span(cyc["X2_first"], cyc["Y2_first"], yb-0.26*dy, f"X→Y intervalo = {m['I_XY']:.3f} s", "#2C3E50")
    _span(cyc["X2_last"],  cyc["Y2_first"], yb-0.32*dy, f"X→Y retardo = {m['D_XY']:.3f} s", COL["NEU"])

    fig.update_layout(height=420,margin=dict(l=10,r=10,b=10,t=10),
                      legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1.0),
                      xaxis_title="tiempo (s)",yaxis_title="x")
    st.plotly_chart(fig,use_container_width=True)

# ===== Tabla =====
st.subheader("Tabla de ciclos (una fila por ciclo)")
if df_cycles.empty:
    st.info("No hay filas en la ventana seleccionada.")
else:
    df_show = df_cycles.round(4).copy()
    df_show.index = np.arange(1, len(df_show)+1); df_show.index.name = "ciclo"
    st.dataframe(df_show, use_container_width=True, height=280)
    st.download_button("Descargar CSV de ciclos", data=df_show.to_csv().encode("utf-8"),
                       file_name="ciclos_xy_spikes.csv", mime="text/csv")

# ===== Pairplots =====
st.header("Relaciones ciclo-a-ciclo (10×10)")
if df_cycles.shape[0] >= 4:
    all_vars = ["Periodo X","Periodo Y","Hiperpol. X (IBI)","Hiperpol. Y (IBI)",
                "Burst X","Burst Y","Intervalo Y→X","Retardo Y→X","Intervalo X→Y","Retardo X→Y"]
    default_vars = [v for v in all_vars if v in df_cycles.columns]
    sel_vars = st.multiselect("Variables a cruzar", options=default_vars, default=default_vars,
                              key="pair_vars_seaborn_order")
    max_c = st.slider("Máximo de ciclos a graficar", 4, int(df_cycles.shape[0]),
                      min(int(df_cycles.shape[0]), 150), step=1)
    triangular = st.toggle("Vista triangular (reduce paneles)", value=False)
    nbins = st.slider("Bins (diagonal)",10,60,24,step=2)
    marker_size = st.slider("Tamaño de marcador",2,8,3,step=1)
    opacity = st.slider("Opacidad",0.2,1.0,0.70,step=0.05)
    if len(sel_vars)>0:
        order = {name:i for i,name in enumerate(all_vars)}
        sel_vars_sorted = sorted(sel_vars, key=lambda c: order.get(c, 1e9))
        dfp = df_cycles.iloc[:max_c][sel_vars_sorted].copy()
        fig = plotly_pairgrid(dfp, sel_vars_sorted, corner=triangular, nbins=nbins,
                              marker_size=marker_size, opacity=opacity,
                              height_per_row=190, share_axes=True)
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Se requieren al menos 4 ciclos para generar los cruces.")

st.caption(f"Muestras (t/γ): {len(t_disp):,} | Ciclos válidos en ventana: {len(cycles_disp)} | "
           f"Integrador: {'LSODA' if locals().get('use_lsoda',True) else 'RK4'} | dt={float(locals().get('dt',0.001)):g}s | "
           f"decimación={int(locals().get('decim_evt',60))}× | γ={gamma:g} | ventana=[{win_disp[0]:.2f}, {win_disp[1]:.2f}] s")
