# -*- coding: utf-8 -*-
# Hindmarsh–Rose (2 neuronas): series + ciclo X→Y→X + tabla (una fila por ciclo) + pairplots 10×10

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
def auto_stride(n_points,max_plot_points=5000): return max(1,int(np.ceil(n_points/max_plot_points)))

# ===== Modelo HR (mismo que el tuyo) =====
@dataclass
class HRParams: e1:float=3.282; e2:float=3.282; mu:float=0.0021; s1:float=1.0; s2:float=1.0; v1:float=0.1; v2:float=0.1
@dataclass
class SynChemSigm: g_syn:float=0.35; theta:float=-0.25; k:float=10.0; E_syn:float=-2.0
@dataclass
class SynChemCPP: g_fast:float=0.10; Esyn:float=-1.8; Vfast:float=-1.1; sfast:float=0.2
@dataclass
class SynElec: g_el:float=0.05

# --- sigmoides numéricamente estables (evita overflow)
def _exp_clip(a): return np.exp(np.clip(a, -60.0, 60.0))
def sigm_stable(x, th, k): return 1.0/(1.0 + _exp_clip(-k*(x - th)))

def rhs_hr(y, t, prm, mode, sc, cc, se):
    x1,y1,z1,x2,y2,z2 = y
    if mode=="quimica_sigmoidal":
        s1s = sigm_stable(x1, sc.theta, sc.k); s2s = sigm_stable(x2, sc.theta, sc.k)
        Isyn1 = sc.g_syn*s2s*(sc.E_syn-x1); Isyn2 = sc.g_syn*s1s*(sc.E_syn-x2); Iel1=Iel2=0.0
    elif mode=="quimica_cpp":
        Isyn1 = -cc.g_fast*(x1-cc.Esyn)/(1.0 + _exp_clip(cc.sfast*(cc.Vfast-x2)))
        Isyn2 = -cc.g_fast*(x2-cc.Esyn)/(1.0 + _exp_clip(cc.sfast*(cc.Vfast-x1))); Iel1=Iel2=0.0
    elif mode=="electrica":
        Iel1=se.g_el*(x2-x1); Iel2=se.g_el*(x1-x2); Isyn1=Isyn2=0.0
    else:
        Isyn1=Isyn2=Iel1=Iel2=0.0
    dx1=y1+3*x1**2-x1**3-z1+prm.e1+Isyn1+Iel1; dy1=1-5*x1**2-y1; dz1=prm.mu*(-prm.v1*z1+prm.s1*(x1+1.6))
    dx2=y2+3*x2**2-x2**3-z2+prm.e2+Isyn2+Iel2; dy2=1-5*x2**2-y2; dz2=prm.mu*(-prm.v2*z2+prm.s2*(x2+1.6))
    return np.array([dx1,dy1,dz1,dx2,dy2,dz2],float)

# ===== Integrador (LSODA por defecto). Warm-up opcional =====
@st.cache_data(show_spinner=False)
def simulate(mode, prm_dict, sc_dict, cc_dict, se_dict,
             y0, nsteps, dt, decim_evt, burn_in_s, use_lsoda=True):

    prm = HRParams(**prm_dict)
    sc  = SynChemSigm(**sc_dict)
    cc  = SynChemCPP(**cc_dict)
    se  = SynElec(**se_dict)

    # Malla fina (para definir el tiempo total) y malla de salida (decimada fija)
    t_full = np.linspace(0.0, float(nsteps)*float(dt), int(nsteps)+1)
    step   = int(max(1, decim_evt))
    t_out  = t_full[::step]

    # Integramos directamente en t_out (ahorra RAM y es suficiente para ráfagas/métricas)
    def _rhs(Y, tt, prm=prm, mode=mode, sc=sc, cc=cc, se=se):
        return rhs_hr(Y, tt, prm, mode, sc, cc, se)

    if use_lsoda and HAVE_SCIPY:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ODEintWarning)
            sol_out = _odeint(_rhs, np.array(y0,float), t_out, atol=1e-6, rtol=1e-6, mxstep=50000)
    else:
        # Fallback RK4 en la misma malla de salida
        y = np.empty((len(t_out), len(y0)), float); y[0] = y0 = np.array(y0,float)
        for i in range(len(t_out)-1):
            dt_i = t_out[i+1]-t_out[i]
            k1=_rhs(y[i],t_out[i]); k2=_rhs(y[i]+0.5*dt_i*k1, t_out[i]+0.5*dt_i)
            k3=_rhs(y[i]+0.5*dt_i*k2, t_out[i]+0.5*dt_i); k4=_rhs(y[i]+dt_i*k3, t_out[i]+dt_i)
            y[i+1]=y[i]+(dt_i/6.0)*(k1+2*k2+2*k3+k4)
        sol_out = y

    # DESCARTAR TRANSITORIO (opcional)
    if burn_in_s>0:
        skip = int(np.clip(np.floor(burn_in_s/(dt*step)), 0, len(t_out)-2))
        t_out = t_out[skip:]
        sol_out = sol_out[skip:]

    return t_out, sol_out

# ===== Detección de ráfagas y ciclos (tu lógica original) =====
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

def cycles_xyx(burstsX,burstsY, tmin=None, tmax=None):
    if tmin is not None or tmax is not None:
        def _crop(bb):
            out=[]
            for (on,off) in bb:
                if (tmin is not None and off<=tmin) or (tmax is not None and on>=tmax): 
                    continue
                out.append((on,off))
            return out
        burstsX=_crop(burstsX); burstsY=_crop(burstsY)
    if not burstsX or len(burstsX)<2 or not burstsY: return []
    X_on=np.array([b[0] for b in burstsX]); X_off=np.array([b[1] for b in burstsX])
    Y_on=np.array([b[0] for b in burstsY]); Y_off=np.array([b[1] for b in burstsY])
    out=[]; j=0
    for i in range(len(X_on)-1):
        x0,xf0=X_on[i],X_off[i]; x1,xf1=X_on[i+1],X_off[i+1]
        if (tmin is not None and (x0<tmin or x1>tmax)): 
            continue
        while j<len(Y_on) and Y_on[j]<x0: j+=1
        if j>=len(Y_on) or Y_on[j]>=x1: continue
        y0,yf=Y_on[j],Y_off[j]; y0n=Y_on[j+1] if (j+1)<len(Y_on) else np.nan
        out.append(dict(x_on0=x0,x_off0=xf0,x_on1=x1,x_off1=xf1,y_on=y0,y_off=yf,y_on_next=y0n))
    return out

def metrics_from_cycle(c):
    B_X=c["x_off0"]-c["x_on0"]; P_X=c["x_on1"]-c["x_on0"]; IBI_X=P_X-B_X
    B_Y=c["y_off"]-c["y_on"]
    P_Y=(c["y_on_next"]-c["y_on"]) if np.isfinite(c["y_on_next"]) else ( (c["y_on"]-c["x_on0"]) + (c["x_on1"]-c["y_on"]) )
    IBI_Y=P_Y-B_Y
    I_XY=c["y_on"]-c["x_on0"]; I_YX=c["x_on1"]-c["y_on"]; D_XY=c["y_on"]-c["x_off0"]; D_YX=c["x_on1"]-c["y_off"]
    return dict(P_X=P_X, IBI_X=IBI_X, B_X=B_X, P_Y=P_Y, IBI_Y=IBI_Y, B_Y=B_Y, I_XY=I_XY, I_YX=I_YX, D_XY=D_XY, D_YX=D_YX)

def cycles_dataframe(cycles):
    rows=[]
    for c in cycles:
        m=metrics_from_cycle(c)
        rows.append({
            "Periodo X":m["P_X"], "Periodo Y":m["P_Y"],
            "Hiperpol. X (IBI)":m["IBI_X"], "Hiperpol. Y (IBI)":m["IBI_Y"],
            "Burst X":m["B_X"], "Burst Y":m["B_Y"],
            "Intervalo X→Y":m["I_XY"], "Intervalo Y→X":m["I_YX"],
            "Retardo X→Y":m["D_XY"], "Retardo Y→X":m["D_YX"]
        })
    return pd.DataFrame(rows)

# ===== Pairgrid Plotly (10×10) =====
def plotly_pairgrid(df, cols, corner=False, nbins=24, marker_size=3, opacity=0.7,
                    height_per_row=190, share_axes=True):
    cols=list(cols); m=len(cols)
    if m==0 or df.shape[0]==0: return go.Figure()
    def _pad(lo,hi,frac=0.02):
        if not np.isfinite(lo) or not np.isfinite(hi): return None
        d=(hi-lo) if hi>lo else (abs(hi)+1.0); p=d*frac
        return [float(lo-p), float(hi+p)]
    rng={c:_pad(np.nanmin(df[c]), np.nanmax(df[c])) for c in cols}
    fig=make_subplots(rows=m, cols=m, shared_xaxes=False, shared_yaxes=False,
                      horizontal_spacing=0.03, vertical_spacing=0.05)
    for i in range(m):
        for j in range(m):
            if corner and j>i: 
                continue
            xi,yj=cols[j],cols[i]
            if i==j:
                fig.add_trace(go.Histogram(x=df[xi], nbinsx=int(nbins), marker_line_width=0, showlegend=False),
                              row=i+1,col=j+1)
            else:
                fig.add_trace(go.Scattergl(x=df[xi], y=df[yj], mode="markers",
                                           marker=dict(size=marker_size, opacity=opacity),
                                           showlegend=False),
                              row=i+1,col=j+1)
            if i==m-1: fig.update_xaxes(title_text=xi, row=i+1, col=j+1)
            if j==0:   fig.update_yaxes(title_text=yj, row=i+1, col=j+1)
            if share_axes:
                if rng[xi] is not None: fig.update_xaxes(range=rng[xi], row=i+1, col=j+1)
                if rng[yj] is not None: fig.update_yaxes(range=rng[yj], row=i+1, col=j+1)
    fig.update_layout(height=max(360,int(height_per_row*m)), margin=dict(l=10,r=10,t=18,b=10),
                      plot_bgcolor="#111418", paper_bgcolor="#111418",
                      font=dict(color="#E6E6E6"), bargap=0.02)
    for ax in fig.layout:
        if ax.startswith("xaxis") or ax.startswith("yaxis"):
            fig.layout[ax].showgrid=True; fig.layout[ax].gridcolor="#333A40"
            fig.layout[ax].zeroline=False; fig.layout[ax].linecolor="#70757a"; fig.layout[ax].tickcolor="#9aa0a6"
    return fig

# ===== UI =====
st.title("Hindmarsh–Rose (2 neuronas) — invariantes por ciclo")

with st.sidebar:
    st.header("Configuración")
    syn_mode=st.radio("Sinapsis",["Química (sigmoidal, HCO)","Química (núcleo C)","Eléctrica (difusiva)"])
    mode={"Química (sigmoidal, HCO)":"quimica_sigmoidal","Química (núcleo C)":"quimica_cpp","Eléctrica (difusiva)":"electrica"}[syn_mode]

    nsteps=st.number_input("Nº de pasos (TIME)", min_value=100_000, value=1_200_000, step=100_000)
    dt = st.number_input("dt (s)", min_value=0.0002, max_value=0.01,
                     value=0.001, step=0.0002, format="%.4f")
    decim_evt=st.number_input("Muestreo para detección (cada N pasos)", min_value=0, max_value=10_000, value=60, step=5)
    burn_in_s = st.number_input("Descartar transitorio (s)", min_value=0.0, value=0.0, step=1.0,
                                help="Si ves 'deriva' al aumentar pasos, descarta los primeros segundos para ver el régimen permanente.")

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
    v_th=st.number_input("Umbral v_th",value=-0.60,step=0.01)
    min_on=st.number_input("Duración mínima ON (s)",value=0.10,step=0.01)
    min_off=st.number_input("Duración mínima OFF (s)",value=0.05,step=0.01)

    st.subheader("Integrador")
    use_lsoda = st.radio("Método",["LSODA (recomendado)","RK4 (fallback)"],index=0,horizontal=True).startswith("LSODA")

# Empaquetado para cache estable
prm_dict=dict(e1=float(e1),e2=float(e2),mu=float(mu),s1=float(s1),s2=float(s2),v1=float(v1),v2=float(v2))
sigm_par=dict(g_syn=float(g_syn),theta=float(theta),k=float(kk),E_syn=float(Ey if (Ey:=Esy) else Esy))
cpp_par=dict(g_fast=float(g_fast),Esyn=float(Esy_c),Vfast=float(Vfast),sfast=float(sfast))
elec_par=dict(g_el=float(locals().get("g_el",0.0)))
y0=np.array([-0.915325,-3.208968,3.350784,-1.307949,-7.580493,3.068898],float)

# Simulación (una sola malla coherente)
t_evt,sol_evt = simulate(mode,prm_dict,sigm_par,cpp_par,elec_par,
                         y0,int(nsteps),float(dt),int(decim_evt),float(burn_in_s),use_lsoda)
x1_evt,x2_evt = sol_evt[:,0], sol_evt[:,3]

# Gráfico general
st.header("Series temporales (decimadas para visualización)")
fig_over=go.Figure()
fig_over.add_trace(go.Scatter(x=t_evt,y=x1_evt,name="X (N1)",line=dict(color=COL["X"])))
fig_over.add_trace(go.Scatter(x=t_evt,y=x2_evt,name="Y (N2)",line=dict(color=COL["Y"])))
fig_over.update_layout(height=330,xaxis_title="tiempo",yaxis_title="x",
                       margin=dict(l=10,r=10,b=10,t=10),
                       legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1.0))
st.plotly_chart(fig_over,use_container_width=True)

# Ventana y ciclos
st.subheader("Ventana temporal para detección de ciclos")
t0, t1 = float(t_evt[0]), float(t_evt[-1])
win = st.slider("Selecciona ventana", min_value=t0, max_value=t1, value=(t0, t1),
                step=(t1-t0)/1000.0, format="%.2f")

burstsA = detect_bursts_interp(t_evt, x1_evt, v_th=v_th, min_on=min_on, min_off=min_off)
burstsB = detect_bursts_interp(t_evt, x2_evt, v_th=v_th, min_on=min_on, min_off=min_off)
cycles  = cycles_xyx(burstsA, burstsB, tmin=win[0], tmax=win[1])
df_cycles = cycles_dataframe(cycles)

# Diagrama de ciclo (igual que tenías)
st.header("Diagrama de ciclo (anclado en X)")
if not cycles:
    st.info("No hay ciclos completos X→Y→X en la ventana. Ajusta el transitorio, la ventana o parámetros.")
else:
    idx = 1 if len(cycles)==1 else st.slider("Ciclo",1,len(cycles),len(cycles),step=1)
    cyc=cycles[idx-1]; m=metrics_from_cycle(cyc)
    y_end = cyc["y_on_next"] if np.isfinite(cyc["y_on_next"]) else (cyc["y_on"]+m["P_Y"])
    x0, x1n = cyc["x_on0"], cyc["x_on1"]; P=x1n-x0
    tmin=max(t_evt[0], x0-0.2*P); tmax=min(t_evt[-1], max(x1n+0.2*P, y_end+0.05*P))
    mm=(t_evt>=tmin)&(t_evt<=tmax)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=t_evt[mm],y=x1_evt[mm],name="X (N1)",line=dict(color=COL["X"],width=2.0)))
    fig.add_trace(go.Scatter(x=t_evt[mm],y=x2_evt[mm],name="Y (N2)",line=dict(color=COL["Y"],width=2.0)))
    fig.add_vrect(x0=cyc["x_on0"],x1=cyc["x_off0"],fillcolor=COL["X"],opacity=0.16,line_width=0)
    fig.add_vrect(x0=cyc["y_on"], x1=cyc["y_off"], fillcolor=COL["Y"],opacity=0.16,line_width=0)
    fig.add_vrect(x0=cyc["x_on1"],x1=cyc["x_off1"],fillcolor=COL["X"],opacity=0.10,line_width=0)
    y_top=max(np.max(x1_evt[mm]),np.max(x2_evt[mm])); y_bot=min(np.min(x1_evt[mm]),np.min(x2_evt[mm])); dy=y_top-y_bot; yb=y_bot-0.10*dy
    def _span(a,b,y,txt,col):
        a=max(float(a),float(tmin)); b=min(float(b),float(tmax))
        if b>a: fig.add_shape(type="line",x0=a,x1=b,y0=y,y1=y,line=dict(color=col,width=3)); fig.add_annotation(**_anno((a+b)/2.0,y,txt))
    _span(cyc["x_on0"],cyc["x_on1"], y_top+0.12*dy, f"Periodo X = {m['P_X']:.3f} s", COL["X"])
    _span(cyc["x_off0"],cyc["x_on1"], y_top+0.06*dy, f"IBI X = {m['IBI_X']:.3f} s", COL["X"])
    _span(cyc["y_on"],    y_end,       y_top+0.00*dy, f"Periodo Y = {m['P_Y']:.3f} s", COL["Y"])
    _span(cyc["x_on0"],cyc["x_off0"], yb-0.00*dy,      f"Duración X = {m['B_X']:.3f} s", COL["X"])
    _span(cyc["y_on"],  cyc["y_off"], yb-0.06*dy,      f"Duración Y = {m['B_Y']:.3f} s", COL["Y"])
    _span(cyc["x_off0"],cyc["y_on"],  yb-0.16*dy,      f"Retardo X→Y = {m['D_XY']:.3f} s", COL["NEU"])
    _span(cyc["y_on"],  cyc["x_on1"], yb-0.22*dy,      f"Intervalo Y→X = {m['I_YX']:.3f} s", "#2C3E50")
    _span(cyc["x_on0"],cyc["y_on"],   yb-0.28*dy,      f"Intervalo X→Y = {m['I_XY']:.3f} s", "#2C3E50")
    fig.update_layout(height=420,margin=dict(l=10,r=10,b=10,t=10),
                      legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1.0),
                      xaxis_title="tiempo (s)",yaxis_title="x")
    st.plotly_chart(fig,use_container_width=True)

# Tabla
st.subheader("Tabla de ciclos X→Y→X (una fila por ciclo)")
if df_cycles.empty:
    st.info("No hay filas en la ventana seleccionada.")
else:
    st.dataframe(df_cycles.round(4), use_container_width=True, height=280)
    st.download_button("Descargar CSV de ciclos", data=df_cycles.to_csv(index=False).encode("utf-8"),
                       file_name="ciclos_xyx.csv", mime="text/csv")

# Pairplots
st.header("Relaciones ciclo-a-ciclo (10×10)")
if df_cycles.shape[0] >= 4:
    all_vars = ["Periodo X","Periodo Y","Hiperpol. X (IBI)","Hiperpol. Y (IBI)","Burst X","Burst Y","Intervalo X→Y","Intervalo Y→X","Retardo X→Y","Retardo Y→X"]
    default_vars = [v for v in all_vars if v in df_cycles.columns]
    sel_vars = st.multiselect("Variables a cruzar", options=default_vars, default=default_vars, key="pair_vars_seaborn_order")
    max_c = st.slider("Máximo de ciclos a graficar", 4, int(df_cycles.shape[0]), min(int(df_cycles.shape[0]), 150), step=1)
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

st.caption(f"Muestras visuales: {len(t_evt):,} | Ciclos detectados en ventana: {len(cycles)} | Integrador: {'LSODA' if use_lsoda else 'RK4'} | dt={dt:g}s | decimación={decim_evt}× | warm-up={burn_in_s:g}s | ventana=[{win[0]:.2f}, {win[1]:.2f}] s")
