import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Simulación del modelo Hindmarsh-Rose modificado (3 variables)
def simulate_modified(a, b, c, d, e, r, s, nu, x0, t_max=15000, dt=0.01):
    x, y, z = 0.1, 0.0, 0.0
    t = 0.0
    t_vals, x_vals = [], []
    while t < t_max:
        dx = y - a*x**3 + b*x**2 - z + e
        dy = c - d*x**2 - y
        dz = r * (-nu*z + s*(x - x0))
        x += dx*dt
        y += dy*dt
        z += dz*dt
        t += dt
        t_vals.append(t)
        x_vals.append(x)
    return t_vals, x_vals

# Interfaz Streamlit para el modificado
st.title('Hindmarsh-Rose Modificado')
# Parámetros ajustables
col1, col2 = st.sidebar.columns(2)
with col1:
    a = st.slider('a', 0.0, 5.0, 1.0, 0.1)
    b = st.slider('b', 0.0, 5.0, 3.0, 0.1)
    c = st.slider('c', 0.0, 5.0, 1.0, 0.1)
    d = st.slider('d', 0.0, 10.0, 5.0, 0.1)
with col2:
    e = st.slider('e', 0.0, 5.0, 3.281, 0.01)
    r = st.slider('mu (r)', 0.0, 0.01, 0.0021, 0.0001)
    s = st.slider('S', 0.0, 5.0, 1.0, 0.1)
    nu = st.slider('nu', 0.0, 1.0, 0.1, 0.01)
x0 = st.slider('x0 (offset)', -2.0, 2.0, -1.6, 0.1)

# Tiempo y resolución
t_max = st.number_input('Tiempo máximo', 1000, 50000, 15000, 1000)
dt = st.number_input('dt', 0.001, 0.1, 0.01, 0.001)

# Límites de visualización
xlim_lower = st.slider('Límite inferior eje x', 0, t_max, 0, 1000)
xlim_upper = st.slider('Límite superior eje x', 0, t_max, t_max, 1000)
if xlim_lower >= xlim_upper:
    st.sidebar.error('El límite inferior debe ser menor al superior')

# Ejecutar simulación
t_vals, x_vals = simulate_modified(a, b, c, d, e, r, s, nu, x0, t_max, dt)

# Dibujar gráfica
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(t_vals, x_vals, color='orange')
ax.set_xlim(xlim_lower, xlim_upper)
ax.set_xlabel('Tiempo')
ax.set_ylabel('x')
ax.set_title('Dinámica del HR Modificado')
st.pyplot(fig)
