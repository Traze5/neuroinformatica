import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.express as px
import io

# --------------------------
# Helpers
# --------------------------
def run_module(path: str):
    """Ejecuta un .py externo mostrando errores en Streamlit sin romper toda la app."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()
        exec(compile(code, path, "exec"), {})
    except FileNotFoundError:
        st.error(f"No se encontró el archivo: {path}")
    except Exception as e:
        st.error(f"Error al ejecutar {path}: {e}")
        st.exception(e)

# --------------------------
# Funciones demo (Home)
# --------------------------
def generate_hindmarsh_rose_data():
    t_values = np.linspace(0, 20, 1000)
    x_values = np.sin(t_values) + np.random.normal(scale=0.1, size=t_values.shape)
    return t_values, x_values

def generate_rulkov_data():
    t_values = np.linspace(0, 20, 1000)
    x_values = np.cos(t_values) + np.random.normal(scale=0.1, size=t_values.shape)
    return t_values, x_values

def generate_hodgkin_huxley_data():
    t_values = np.linspace(0, 20, 1000)
    x_values = np.tanh(t_values) + np.random.normal(scale=0.1, size=t_values.shape)
    return t_values, x_values

def generate_izhikevich_data():
    t_values = np.linspace(0, 20, 1000)
    x_values = np.log(t_values + 1) + np.random.normal(scale=0.1, size=t_values.shape)
    return t_values, x_values

# --------------------------
# Sidebar / Menú
# --------------------------
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=[
            "Home",
            "Modelo Hindmarsh-Rose",
            "Modelo Hindmarsh-Rose Modificado",
            "Modelo Hindmarsh-Rose Caótico",
            "Modelo Izhikevich",
            "Modelo Rulkov",
            "Invariantes HR",
            "Invariantes Rulkov",
            "Miguel Angel Calderón",
        ],
        icons=[
            "house",
            "activity",
            "tools",
            "lightning",
            "triangle",
            "cpu",
            "bar-chart-line",
            "bar-chart-line",
            "person",
        ],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important"},
            "icon": {"color": "#6c757d", "font-size": "18px"},
            "nav-link": {"font-size": "15px", "text-align": "left", "margin": "0"},
            "nav-link-selected": {"background-color": "#1f2937"},
        },
    )

# --------------------------
# Ruteo de páginas
# --------------------------

# Home
if selected == "Home":
    st.header('Aplicación de Simulación de Neuronas')
    st.write("Bienvenido. Usa el menú de la izquierda para navegar por los modelos y análisis.")

    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)

    with c1:
        t_values, x_values = generate_hindmarsh_rose_data()
        fig = px.line(x=t_values, y=x_values, labels={'x':'Tiempo', 'y':'x'})
        fig.update_layout(title='Modelo Hindmarsh-Rose')
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        t_values, x_values = generate_rulkov_data()
        fig = px.line(x=t_values, y=x_values, labels={'x':'Tiempo', 'y':'x'})
        fig.update_layout(title='Modelo Rulkov')
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        t_values, x_values = generate_hodgkin_huxley_data()
        fig = px.line(x=t_values, y=x_values, labels={'x':'Tiempo', 'y':'x'})
        fig.update_layout(title='Modelo Hodgkin-Huxley')
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        t_values, x_values = generate_izhikevich_data()
        fig = px.line(x=t_values, y=x_values, labels={'x':'Tiempo', 'y':'x'})
        fig.update_layout(title='Modelo Izhikevich')
        st.plotly_chart(fig, use_container_width=True)

# Modelo Hindmarsh-Rose
elif selected == "Modelo Hindmarsh-Rose":
    st.header('Simulación del Modelo Hindmarsh-Rose')
    run_module("neu.py")

# Modelo Hindmarsh-Rose modificado
elif selected == "Modelo Hindmarsh-Rose Modificado":
    st.header('Simulación del Modelo Hindmarsh-Rose (Modificado)')
    run_module("neu_mod.py")

# Modelo HR caótico
elif selected == "Modelo Hindmarsh-Rose Caótico":
    st.header('Simulación del Modelo Hindmarsh-Rose Caótico')
    run_module("neu2.py")

# Modelo Izhikevich
elif selected == "Modelo Izhikevich":
    st.header('Simulación del Modelo Izhikevich')
    run_module("neu3.py")

# Modelo Rulkov
elif selected == "Modelo Rulkov":
    st.header('Simulación del Modelo Rulkov')
    run_module("neu4.py")

# Invariantes HR (nuevo)
elif selected == "Invariantes HR":
    st.header('Invariantes Dinámicos — Hindmarsh–Rose')
    run_module("invariantesh.py")

# Invariantes Rulkov (nuevo)
elif selected == "Invariantes Rulkov":
    st.header('Invariantes Dinámicos — Rulkov')
    run_module("invariantesr.py")

# Página de contacto
elif selected == "Miguel Angel Calderón":
    st.header("Miguel Angel Calderón")
    st.write("Data Scientist.")

