import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Definir el modelo basado en mapas con interacción sináptica
def map_model(xn, yn, params):
    a, l, r, rn = params
    if xn <= 0:
        fa = a / (1 - xn)
    elif 0 < xn < a + yn + rn:
        fa = a + yn + rn
    else:
        fa = -1
    
    xn1 = fa
    yn1 = yn - l * (xn + 1) + l * r + l * rn
    return xn1, yn1

def simulate_map_model(a, l, r, rn, steps=1000):
    # Condiciones iniciales
    xn = -0.5
    yn = 0.5

    # Arrays para almacenar los resultados
    x_values = []
    y_values = []

    # Simulación
    for _ in range(steps):
        xn, yn = map_model(xn, yn, (a, l, r, rn))
        x_values.append(xn)
        y_values.append(yn)

    return np.array(x_values), np.array(y_values)

# Crear la interfaz de Streamlit
st.set_page_config(page_title="Simulación Neuronal Basada en Mapas", layout="wide")

st.title('Simulación de Modelos Neuronales Basados en Mapas')
st.markdown("""
Esta aplicación permite simular y visualizar los patrones de actividad neuronal utilizando un modelo basado en mapas. Ajuste los parámetros en la barra lateral para observar cómo cambian los patrones de actividad en tiempo real.
""")

st.sidebar.header('Parámetros del Modelo')
st.sidebar.markdown("Ajuste los parámetros del modelo y haga clic en 'Simular' para actualizar la gráfica.")

# Crear sliders para ajustar los parámetros
a = st.sidebar.slider('a', 3.0, 4.0, 3.65, 0.01)
l = st.sidebar.slider('l', 0.0001, 0.001, 0.0005, 0.0001)
r = st.sidebar.slider('r', 0.0, 1.0, 0.1, 0.01)
rn = st.sidebar.slider('rn', 0.0, 1.0, 0.1, 0.01)

if st.sidebar.button('Simular'):
    x_values, y_values = simulate_map_model(a, l, r, rn)
    
    # Graficar resultados
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    ax[0].plot(x_values, label='x_n')
    ax[0].set_title('Actividad Neuronal x(t)')
    ax[0].set_xlabel('Tiempo (pasos)')
    ax[0].set_ylabel('x(t)')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(y_values, label='y_n', color='red')
    ax[1].set_title('Actividad Neuronal y(t)')
    ax[1].set_xlabel('Tiempo (pasos)')
    ax[1].set_ylabel('y(t)')
    ax[1].legend()
    ax[1].grid(True)

    st.pyplot(fig)
else:
    st.write("Ajuste los parámetros en la barra lateral y haga clic en 'Simular' para ver los resultados.")

st.sidebar.markdown("""
---

""")
