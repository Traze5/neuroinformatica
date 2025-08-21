import streamlit as st
import numpy as np
import plotly.express as px

def simulate_izhikevich(a, b, c, d, I, t_max=280, dt=0.001):
    """
    Simula la dinámica de una neurona usando el modelo Izhikevich.
    """
    # Condiciones iniciales
    v = -68.324165
    u = 0.346447
    t = 0

    v_values = []
    t_values = []

    # Simulación
    while t < t_max:
        if v >= 30:
            v = c
            u += d
        v1 = v + dt * (0.04 * v**2 + 5 * v + 140 - u + I)
        u1 = u + dt * a * (b * v - u)
        v = v1
        u = u1
        t += dt
        
        v_values.append(v)
        t_values.append(t)

    return t_values, v_values

# Configuración de Streamlit
st.title('Simulación del Modelo Izhikevich')

# Sidebar para parámetros del modelo
a = st.sidebar.slider('a', min_value=0.01, max_value=0.2, value=0.02, step=0.01)
b = st.sidebar.slider('b', min_value=0.1, max_value=0.3, value=0.2, step=0.01)
c = st.sidebar.slider('c', min_value=-80, max_value=-40, value=-50, step=1)
d = st.sidebar.slider('d', min_value=1, max_value=10, value=2, step=1)
I = st.sidebar.slider('I (corriente de entrada)', min_value=5, max_value=15, value=10, step=1)
t_max = st.sidebar.number_input('Tiempo máximo de simulación (ms)', value=280, step=10)
dt = st.sidebar.number_input('Delta t', value=0.001, format="%.4f")

# Simulación automática
t_values, v_values = simulate_izhikevich(a, b, c, d, I, t_max, dt)

# Crear y mostrar la gráfica con Plotly
fig = px.line(x=t_values, y=v_values, labels={'x':'Tiempo (ms)', 'y':'v (mV)'})
fig.update_layout(title='Dinámica de la Neurona (Modelo Izhikevich)')

st.plotly_chart(fig)
