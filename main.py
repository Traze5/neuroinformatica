import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.express as px

# Funciones para generar datos aleatorios de modelos neuronales
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

# Configuración del menú de navegación Streamlit_optionmenu
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Modelo Hindmarsh-Rose","Modelo Hindmarsh-Rose Modificado", "Modelo Hindmarsh-Rose Caótico","Modelo Izhikevich", "Modelo Rulkov", "Miguel Angel Calderón"],
        icons=["house", "cloud", "activity", "activity", "triangle", "person"],
        menu_icon="cast",
        default_index=0,
    )

# Página de inicio
if selected == "Home":
    st.header('Aplicación de Simulación de Neuronas')
    st.write("Bienvenido a la aplicación de simulación de neuronas. Use el menú de la izquierda para navegar.")

    # Crear un diseño de fila
    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)

    with c1:
        t_values, x_values = generate_hindmarsh_rose_data()
        fig = px.line(x=t_values, y=x_values, labels={'x':'Tiempo', 'y':'x'})
        fig.update_layout(title='Modelo Hindmarsh-Rose')
        st.plotly_chart(fig)
           
    with c2:
        t_values, x_values = generate_rulkov_data()
        fig = px.line(x=t_values, y=x_values, labels={'x':'Tiempo', 'y':'x'})
        fig.update_layout(title='Modelo Rulkov')
        st.plotly_chart(fig)

    with c3:
        t_values, x_values = generate_hodgkin_huxley_data()
        fig = px.line(x=t_values, y=x_values, labels={'x':'Tiempo', 'y':'x'})
        fig.update_layout(title='Modelo Hodgkin-Huxley')
        st.plotly_chart(fig)

    with c4:
        t_values, x_values = generate_izhikevich_data()
        fig = px.line(x=t_values, y=x_values, labels={'x':'Tiempo', 'y':'x'})
        fig.update_layout(title='Modelo Izhikevich')
        st.plotly_chart(fig)

# Página del modelo Hindmarsh-Rose
if selected == "Modelo Hindmarsh-Rose":
    st.header('Simulación del Modelo Hindmarsh-Rose')
    # Importar y ejecutar el código del modelo Hindmarsh-Rose
    # Reemplazar con la ruta de tu archivo .py
    exec(open("neu.py").read())
    
    # Página del modelo Hindmarsh-Rose modificado
if selected == "Modelo Hindmarsh-Rose Modificado":
    st.header('Simulación del Modelo Hindmarsh-Rose (Modificado)')
    exec(open("neu_mod.py").read())

# Página del modelo Hindmarsh-Rose Caótico
if selected == "Modelo Hindmarsh-Rose Caótico":
    st.header('Simulación del Modelo Hindmarsh-Rose Caótico')
    # Importar y ejecutar el código del modelo Hindmarsh-Rose Caótico
    # Reemplazar con la ruta de tu archivo .py
    exec(open("neu2.py").read())

# Página del modelo Izhikevich
if selected == "Modelo Izhikevich":
    st.header('Simulación del Modelo Izhikevich')
    # Importar y ejecutar el código del modelo Izhikevich
    exec(open("neu3.py").read())

# Página del modelo Rulkov
if selected == "Modelo Rulkov":
    st.header('Simulación del Modelo Rulkov')
    # Anexo al vodigo el código del modelo Rulkov
    exec(open("neu4.py").read())

# Página de contacto
if selected == "Miguel Angel Calderón":
    st.header("Miguel Angel Calderón")
    st.write("Data Scientist.")
