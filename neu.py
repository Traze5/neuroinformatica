import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def simulate_neuron(a, b, c, d, e, r, s, x0, t_max=15000, dt=0.01):
    """
    Simula la dinamica de una neurona usando el modelo Hindmarsh-Rose.
    """
    # Condiciones iniciales
    x = 0.1
    y = 0.0
    z = 0.0
    t = 0

    x_values = []
    t_values = []

    # Simulacion
    while t < t_max:
        dx = y - a * x**3 + b * x**2 - z + e
        dy = c - d * x**2 - y
        dz = r * (s * (x - x0) - z)
        
        x += dx * dt
        y += dy * dt
        z += dz * dt
        t += dt
        
        x_values.append(x)
        t_values.append(t)

    return t_values, x_values

# Configuracion de Streamlit
st.title('Simulacion del Modelo Hindmarsh-Rose')

# Sidebar para parametros del modelo
a = st.sidebar.slider('a', min_value=0.0, max_value=5.0, value=1.0, step=0.1)
b = st.sidebar.slider('b', min_value=0.0, max_value=5.0, value=3.0, step=0.1)
c = st.sidebar.slider('c', min_value=0.0, max_value=5.0, value=1.0, step=0.1)
d = st.sidebar.slider('d', min_value=0.0, max_value=10.0, value=5.0, step=0.1)
e = st.sidebar.slider('e', min_value=0.0, max_value=5.0, value=3.0, step=0.1)  # Corriente de entrada
r = st.sidebar.slider('r', min_value=0.0, max_value=0.01, value=0.0021, step=0.0001)
s = st.sidebar.slider('s', min_value=0.0, max_value=5.0, value=4.0, step=0.1)
x0 = st.sidebar.slider('x0', min_value=-2.0, max_value=2.0, value=-1.6, step=0.1)
t_max = st.sidebar.number_input('Tiempo maximo de simulacion', value=15000, step=1000)
dt = st.sidebar.number_input('Delta t', value=0.01, format="%.3f")

# Agregar sliders para los límites de x en la barra lateral
xlim_lower = st.sidebar.slider('Límite inferior del eje X', min_value=0, max_value=t_max, value=1000, step=1000)
xlim_upper = st.sidebar.slider('Límite superior del eje X', min_value=0, max_value=t_max, value=t_max, step=1000)

# Validar que el límite inferior sea menor que el límite superior
if xlim_lower >= xlim_upper:
    st.sidebar.error('El límite inferior debe ser menor que el límite superior')


# Simulacion automatica
t_values, x_values = simulate_neuron(a, b, c, d, e, r, s, x0, t_max, dt)

# Crear y mostrar la grafica
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(xlim_lower, xlim_upper)
ax.plot(t_values, x_values)
ax.set_xlabel('Tiempo')
ax.set_ylabel('x')
ax.set_title('Dinámica de la Neurona (Modelo Hindmarsh-Rose)')
st.pyplot(fig)

