import streamlit as st
import numpy as np
import plotly.express as px

def simulate_rulkov(mu, sigma, alpha, t_max=280, dt=1):
    """
    Simula la dinámica de una neurona usando el modelo Rulkov.
    """
    # Condiciones iniciales
    x = -1.958753
    y = -3.983966
    t = 0

    x_values = []
    t_values = []

    # Simulación
    while t < t_max:
        if x <= 0:
            x1 = alpha / (1 - x) + y
            y1 = y - mu * (x + 1 - sigma)
        elif 0 < x < (alpha + y):
            x1 = alpha + y
            y1 = y - mu * (x + 1 - sigma)
        elif x >= (alpha + y):
            x1 = -1
            y1 = y - mu * (x + 1 - sigma)
        
        x = x1
        y = y1
        t += dt
        
        x_values.append(x)
        t_values.append(t)

    return t_values, x_values

# Configuración de Streamlit
st.title('Simulación del Modelo de Rulkov')

# Sidebar para parámetros del modelo
mu = st.sidebar.slider('mu', min_value=0.0001, max_value=0.01, value=0.001, step=0.0001)
sigma = st.sidebar.slider('sigma', min_value=-1.0, max_value=1.0, value=-0.1, step=0.1)
alpha = st.sidebar.slider('alpha', min_value=1.0, max_value=10.0, value=6.0, step=0.1)
t_max = st.sidebar.number_input('Tiempo máximo de simulación', value=710, step=10)
dt = st.sidebar.number_input('Delta t', value=1, step=1)

# Simulación automática
t_values, x_values = simulate_rulkov(mu, sigma, alpha, t_max, dt)

# Crear y mostrar la gráfica con Plotly
fig = px.line(x=t_values, y=x_values, labels={'x':'Tiempo', 'y':'x'})
fig.update_layout(title='Dinámica de la Neurona (Modelo Rulkov)')

st.plotly_chart(fig)
