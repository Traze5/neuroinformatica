#librerias de codigo
import streamlit as st
import pandas as pd
import scipy.io
import io
import mne
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.signal import butter, filtfilt
import matplotlib.patches as patches

#librerias estilo
import base64
from PIL import Image

# Abre la imagen y la convierte en base64
with open("Img/fondo2.png", "rb") as img_file:
    b64_string = base64.b64encode(img_file.read()).decode()

# CSS en una cadena de texto
style = f"""
<style>
.stApp {{
    background-image: url("data:image/png;base64,{b64_string}");
    background-size: cover;
}}
</style>
"""

# # Añade el estilo a la aplicación Streamlit
# st.markdown(style, unsafe_allow_html=True)

#Sessiones
# Inicializa una variable en session_state para rastrear si los gráficos han sido generados
if 'graficos_generados' not in st.session_state:
    st.session_state['graficos_generados'] = False


# Título de la aplicación
# Crea tres columnas
col1, col2, col3 = st.columns([1,1,2])

# Coloca tu imagen en la tercera columna
with col3:
    st.image('Img/logo-uam.png', width=100)
with col1:
    st.markdown('### Biodispositivos')
    st.markdown('###### Miguel Angel Calderón')

st.markdown('### Detección en Señales')
# Widget para cargar archivos de datos EEG
uploaded_file = st.file_uploader("Agrega el archivo del sujeto a Analizar", type=["mat"])

if uploaded_file is not None:
    # Utilizar un buffer de memoria para leer el archivo .mat
    file_buffer = io.BytesIO(uploaded_file.read())

    # Cargar los datos EEG
    subject_data = scipy.io.loadmat(file_buffer)
    data = subject_data['data']  # Ajusta esto según la estructura de tus datos

    # Hacer algo con 'data', por ejemplo, mostrar su forma
    st.write('Data shape:', data.shape)
# Widget para cargar el archivo Freq_Phase.mat
# Inicializar 'freqs' y 'phases' fuera del botón para asegurarse de que estén definidos
freqs, phases = None, None

uploaded_freq_phase_file = st.file_uploader("Agrega el archivo Freq_Phase.mat", type=["mat"])

if uploaded_freq_phase_file is not None:
    # Cargar frecuencias y fases
    freq_phase_data = scipy.io.loadmat(uploaded_freq_phase_file)
    freqs = freq_phase_data['freqs'][0]
    phases = freq_phase_data['phases'][0]

    # Mostrar frecuencias y fases
    #st.write("Frecuencias:", freqs)
    #st.write("Fases:", phases)

# Cargar archivo de canales .loc
uploaded_channel_file = st.file_uploader("Agrega el archivo de canales (.loc)", type=["loc"])

# Inicializar selected_indices
selected_indices = []
if uploaded_channel_file is not None:
    # Leyendo el archivo .loc con pandas
    channel_data = pd.read_csv(uploaded_channel_file, sep="\t", header=None)

    # Ajustar los índices del DataFrame para que comiencen desde 1
    channel_data.index = range(1, len(channel_data) + 1)
    
    channel_names = channel_data[3]

    # Seleccionar múltiples canales (hasta 10)
    selected_channels = st.multiselect("Selecciona los canales", channel_names, default=None)

    # Buscar y mostrar los índices de los canales seleccionados
    selected_indices = [channel_data.index[channel_data[3] == channel].tolist()[0] for channel in selected_channels if channel in channel_data[3].values]

#if selected_indices:
#    st.write("Índices de los canales seleccionados:", selected_indices)

# if selected_indices:
#     # Leer las coordenadas de los electrodos del archivo .loc y convertirlas a un diccionario
#     electrode_coords = pd.read_csv(uploaded_channel_file, sep="\t", header=None, index_col=0)
#     electrode_coords.columns = ['theta', 'radius', 'name']
#     # Convertir de polar a coordenadas cartesianas
#     x = electrode_coords['radius'] * np.cos(np.deg2rad(electrode_coords['theta']))
#     y = electrode_coords['radius'] * np.sin(np.deg2rad(electrode_coords['theta']))
#     z = np.zeros(len(x))  # Altura de los electrodos, asumimos un plano
#     ch_pos = dict(zip(electrode_coords['name'], np.vstack([x, y, z]).T))

#     # Crear un objeto de Info con los electrodos seleccionados
#     selected_ch_names = electrode_coords.loc[selected_indices, 'name'].tolist()
#     info = mne.create_info(selected_ch_names, sfreq=1000, ch_types='eeg')

#     # Crear un Montage personalizado
#     montage = mne.channels.make_dig_montage(ch_pos=ch_pos)
#     info.set_montage(montage)

    # Visualizar los electrodos seleccionados
    fig, ax = plt.subplots()
    mne.viz.plot_sensors(info, axes=ax, show_names=True)
    st.pyplot(fig)

# Verificar si los tres archivos necesarios han sido cargados
if uploaded_file is not None and uploaded_freq_phase_file is not None and uploaded_channel_file is not None:
    # Agregar texto explicativo con la fórmula matemática
    st.markdown("""
    ### Análisis de la Dispersión de la Amplitud Promedio

    Este análisis visualiza la dispersión de la amplitud promedio en los datos de frecuencia para los electrodos seleccionados. La amplitud promedio para cada frecuencia se calcula de la siguiente manera:
    """)

    # Mostrar la fórmula matemática y su explicación dentro de un entorno 'align*'
    st.latex(r"""
    \begin{align*}
    A_i &= \frac{1}{N} \sum_{j=1}^{N} \left| D_{ij} \right| \\
    \text{Donde } & A_i \text{ es la amplitud promedio para la frecuencia } i, \\
    & D_{ij} \text{ es la amplitud en la frecuencia } i \text{ y el tiempo } j, \\
    & \text{y } N \text{ es el número total de puntos de tiempo.}
    \end{align*}
    """)


    # Botón para visualizar los gráficos
    if st.button('Observar la dispersión de la amplitud promedio de las señales'):
        # Verificar que todos los archivos necesarios han sido cargados
        if uploaded_file is not None and freqs is not None and uploaded_channel_file is not None:
            # Asegurarse de que se hayan seleccionado electrodos       
            if selected_indices:
                for index in selected_indices:
                    # Obtener las amplitudes promedio para el canal seleccionado
                    amp = np.abs(np.squeeze(data[index, 0, :, :])).mean(axis=1)

                    # Crear gráfico de dispersión
                    fig, ax = plt.subplots(figsize=(15, 6))
                    ax.scatter(freqs, amp, s=15, label=channel_data[3][index])

                    # Identificar y resaltar los tres puntos más altos
                    top_indices = np.argsort(amp)[-3:]
                    ax.scatter(freqs[top_indices], amp[top_indices], color='red', s=30)

                    # Agregar etiquetas a cada punto
                    for i, txt in enumerate(range(1, len(freqs) + 1)):
                        ax.text(freqs[i], amp[i], str(txt), fontsize=9, ha='center', va='bottom')

                    plt.style.use('grayscale')
                    plt.title(f'Frecuencia vs Amplitud para {channel_data[3][index]}')
                    plt.xlabel('Frecuencia [Hz]')
                    plt.ylabel('Amplitud Promedio (uV)')
                    plt.legend()
                    plt.grid(True)
                    st.pyplot(fig)  # Mostrar el gráfico en Streamlit

                st.session_state['graficos_generados'] = True  # Indicar que los gráficos se han generado
            else:
                st.warning('Por favor, selecciona al menos un canal.')
        else:
            st.error('Asegúrate de haber cargado todos los archivos necesarios.')
        
# Mostrar el widget para el punto de interés solo si los gráficos han sido generados
if st.session_state['graficos_generados']:
    punto_interes = st.number_input("Ingresa el número del punto de interés", min_value=1, max_value=len(freqs), step=1, key='punto_interes')
     # Mostrar la imagen (reemplaza 'ruta/a/tu/imagen.jpg' con la ruta de tu imagen)
    st.image('datos/tecladobase.png', caption='Configuración de Experimento Inicial')
    # Mostrar la frecuencia y fase correspondientes al punto de interés automáticamente
    if 1 <= punto_interes <= len(freqs):
        indice_punto_interes = punto_interes - 1
        target_frequency = round(freqs[indice_punto_interes], 2)
        target_phase = round(phases[indice_punto_interes], 4)
        st.write(f'Frecuencia correspondiente al punto {punto_interes}: {target_frequency} Hz')
        st.write(f'Fase correspondiente al punto {punto_interes}: {target_phase} radianes')
            # Añade esta línea para almacenar la frecuencia objetivo en session_state
        st.session_state['target_frequency'] = target_frequency
        st.session_state['target_phase'] = target_phase
    
        # Botón para procesar y mostrar los gráficos
if st.session_state['graficos_generados'] and selected_indices:
    if st.button('Mostrar Datos de EEG para Frecuencia y Fase Seleccionadas'):
        # Encontrar el índice de la frecuencia y fase más cercanas
        closest_freq_idx = np.argmin(np.abs(freqs - target_frequency))
        closest_phase_idx = np.argmin(np.abs(phases - target_phase))

        # Extraer el fragmento de EEG en la frecuencia y fase seleccionadas
        eeg_fragment = data[:, :, closest_freq_idx, closest_phase_idx]

        # Transponer los datos para que la dimensión de tiempo esté en el eje correcto
        eeg_fragment = eeg_fragment.T        
        # Almacenar eeg_fragment en session_state
        st.session_state['eeg_fragment'] = eeg_fragment

        # Obtener el eje temporal
        time_points = np.arange(eeg_fragment.shape[0]) / 250.0  # Escala de tiempo
        # Almacenar eeg_fragment en session_state
        st.session_state['time_points'] = time_points
        
        # Graficar las señales crudas para los canales seleccionados
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(12, 6))
        for index in selected_indices:
            ax.plot(time_points, eeg_fragment[:, index], label=f'Canal {channel_data[3][index]}')

        ax.grid(False)
        ax.set_xlabel('Tiempo [s]')
        ax.set_ylabel('Amplitud [uV]')
        ax.set_title(f'Datos Crudos de señales en {target_frequency} Hz y fase {target_phase}π')
        ax.legend()
        st.pyplot(fig)  # Mostrar el gráfico en Streamlit

# Botón para realizar el análisis FFT y mostrar los resultados
if st.session_state['graficos_generados'] and 'eeg_fragment' in st.session_state and selected_indices:
    if st.button('Realizar análisis FFT'):
        eeg_fragment = st.session_state['eeg_fragment']

        fig, ax = plt.subplots(figsize=(12, 6))

        for index in selected_indices:
            # Aplicar la Transformada Rápida de Fourier (FFT) a las señales
            fft_signal = np.fft.fft(eeg_fragment[:, index])
            frecuencias = np.fft.fftfreq(len(fft_signal), d=(1.0 / 250))
            espectro_amplitud = np.abs(fft_signal)

            # Graficar el espectro de amplitud
            ax.plot(frecuencias, espectro_amplitud, label=f'Canal {channel_data[3][index]}')

            # Marcar y etiquetar el punto correspondiente
            target_frequency_idx = np.argmin(np.abs(frecuencias - st.session_state['target_frequency']))
            ax.scatter(frecuencias[target_frequency_idx], espectro_amplitud[target_frequency_idx], color='red', marker='o')
            ax.text(frecuencias[target_frequency_idx], espectro_amplitud[target_frequency_idx], f'   ({math.floor(frecuencias[target_frequency_idx]*10)/10:.2f} Hz)', color='red')
            
        ax.set_xlim(2, 80)
        ax.set_ylim(0, 6000)
        ax.set_xlabel('Frecuencia [Hz]')
        ax.set_ylabel('Amplitud uV')
        ax.set_title('Espectro de Amplitud')
        ax.legend()
        st.pyplot(fig)  # Mostrar el gráfico en Streamlit

# Botón para aplicar el filtro pasa banda y visualizar
if st.session_state['graficos_generados'] and 'eeg_fragment' in st.session_state and 'target_frequency' in st.session_state and selected_indices:
    if st.button('Aplicar Filtro Pasa Banda y Visualizar'):
        # Utilizar la frecuencia seleccionada por el usuario como frecuencia central
        time_points = st.session_state['time_points']
        frecuencia_corte = st.session_state['target_frequency']
        ancho_banda = 0.2  # Hz (ancho de banda del filtro)

        # Calcular las frecuencias de corte normalizadas
        frecuencia_corte_normalizada = frecuencia_corte / (250 / 2)
        ancho_banda_normalizado = ancho_banda / (250 / 2)

        # Diseño del filtro Butterworth pasa banda
        b, a = butter(N=4, Wn=[frecuencia_corte_normalizada - ancho_banda_normalizado / 2,
                               frecuencia_corte_normalizada + ancho_banda_normalizado / 2], btype='band')

        # Crear una figura y un eje para la visualización
        fig, ax = plt.subplots(figsize=(12, 6))

        # Inicializar un diccionario para almacenar los datos filtrados
        st.session_state['eeg_fragment_filtrado'] = {}

        for index in selected_indices:
            # Aplicar el filtro
            eeg_fragment_filtrado = filtfilt(b, a, st.session_state['eeg_fragment'][:, index])

            # Almacenar los datos filtrados en session_state
            st.session_state['eeg_fragment_filtrado'][index] = eeg_fragment_filtrado

            # Graficar señales originales y filtradas en el mismo eje
            ax.plot(time_points, st.session_state['eeg_fragment'][:, index], label=f'Canal {channel_data[3][index]} (Original)', alpha=0.7)
            ax.plot(time_points, eeg_fragment_filtrado, label=f'Canal {channel_data[3][index]} (Filtrado)', linestyle='--')

        ax.set_xlabel('Tiempo [s]')
        ax.set_ylabel('Amplitud [uV]')
        ax.set_title(f'Datos Crudos vs Filtros para Canales Seleccionados en {frecuencia_corte} Hz')
        ax.legend()
        ax.grid(True)

        # Mostrar el gráfico en Streamlit
        st.pyplot(fig)

# Botón para realizar la FFT en las señales filtradas y visualizar los resultados
if st.session_state['graficos_generados'] and 'eeg_fragment_filtrado' in st.session_state and selected_indices:
    if st.button('Realizar FFT en Señales Filtradas y Mostrar Espectro de Amplitud'):
        fig, ax = plt.subplots(figsize=(12, 6))

        for index in selected_indices:
            # Verificar si el canal filtrado está en session_state
            if index in st.session_state['eeg_fragment_filtrado']:
                # Realizar la FFT en la señal filtrada
                fft_signal_filtrado = np.fft.fft(st.session_state['eeg_fragment_filtrado'][index])
                frecuencias_filtrado = np.fft.fftfreq(len(fft_signal_filtrado), d=(1.0 / 250))
                espectro_amplitud_filtrado = np.abs(fft_signal_filtrado)

                # Encontrar la frecuencia del punto máximo
                frecuencia_max = frecuencias_filtrado[np.argmax(espectro_amplitud_filtrado)]

                # Graficar el espectro de amplitud
                ax.plot(frecuencias_filtrado, espectro_amplitud_filtrado, label=f'Canal {channel_data[3][index]} (Filtrado)')

                # Marcar el punto máximo
                ax.scatter(frecuencia_max, np.max(espectro_amplitud_filtrado), color='black', marker='.', label=f'Max {channel_data[3][index]} ({math.floor(frecuencia_max * 10) / 10:.1f} Hz)')

            else:
                st.error(f'No se encontraron datos filtrados para el canal {index}. Asegúrese de haber aplicado el filtro primero.')

        ax.set_xlim(0, 40)
        ax.set_ylim(0, 800)
        ax.set_xlabel('Frecuencia [Hz]')
        ax.set_ylabel('Amplitud')
        ax.set_title('Espectro de Amplitud para Canales Filtrados')
        ax.legend()

        # Mostrar el gráfico en Streamlit
        st.pyplot(fig)


key_map = {
    (round(8.0, 2), round(0.0, 4)): '.',
    (round(9.0, 2), round(1.5708, 4)): 'c',
    (round(10.0, 2), round(3.1416, 4)): 'h',
    (round(11.0, 2), round(4.7124, 4)): 'm',
    (round(12.0, 2), round(0.0, 4)): 'r',
    (round(13.0, 2), round(1.5708, 4)): 'w',
    (round(14.0, 2), round(3.1416, 4)): '1',
    (round(15.0, 2), round(4.7124, 4)): '6',
    (round(8.2, 2), round(1.5708, 4)): ',',
    (round(9.2, 2), round(3.1416, 4)): 'd',
    (round(10.2, 2), round(4.7124, 4)): '',
    (round(11.2, 2), round(0.0, 4)): 'n',
    (round(12.2, 2), round(1.5708, 4)): 's',
    (round(13.2, 2), round(3.1416, 4)): 'x',
    (round(14.2, 2), round(4.7124, 4)): '2',
    (round(15.2, 2), round(0.0, 4)): '7',
    (round(8.4, 2), round(3.1416, 4)): '<',
    (round(9.4, 2), round(4.7124, 4)): 'e',
    (round(10.4, 2), round(0.0, 4)): 'j',
    (round(11.4, 2), round(1.5708, 4)): 'o',
    (round(12.4, 2), round(3.1416, 4)): 't',
    (round(13.4, 2), round(4.7124, 4)): 'y',
    (round(14.4, 2), round(0.0, 4)): '3',
    (round(15.4, 2), round(1.5708, 4)): '8',
    (round(8.6, 2), round(4.7124, 4)): 'a',
    (round(9.6, 2), round(0.0, 4)): 'f',
    (round(10.6, 2), round(1.5708, 4)): 'k',
    (round(11.6, 2), round(3.1416, 4)): 'p',
    (round(12.6, 2), round(4.7124, 4)): 'u',
    (round(13.6, 2), round(0.0, 4)): 'z',
    (round(14.6, 2), round(1.5708, 4)): '4',
    (round(15.6, 2), round(3.1416, 4)): '9',
    (round(8.8, 2), round(0.0, 4)): 'b',
    (round(9.8, 2), round(1.5708, 4)): 'g',
    (round(10.8, 2), round(3.1416, 4)): 'l',
    (round(11.8, 2), round(4.7124, 4)): 'q',
    (round(12.8, 2), round(0.0, 4)): 'v',
    (round(13.8, 2), round(1.5708, 4)): '0',
    (round(14.8, 2), round(3.1416, 4)): '5',
    (round(15.8, 2), round(4.7124, 4)): '_',
}

# key_map = {
#     (8.0, 0.0): '.',
#     (9.0, 1.57079633): 'c',
#     (10.0, 3.14159265): 'h',
#     (11.0, 4.71238898): 'm',
#     (12.0, 0.0): 'r',
#     (13.0, 1.57079633): 'w',
#     (14.0, 3.14159265): '1',
#     (15.0, 4.71238898): '6',
#     (8.2, 1.57079633): ',',
#     (9.2, 3.14159265): 'd',
#     (10.2, 4.71238898): '',
#     (11.2, 0.0): 'n',
#     (12.2, 1.57079633): 's',
#     (13.2, 3.14159265): 'x',
#     (14.2, 4.71238898): '2',
#     (15.2, 0.0): '7',
#     (8.4, 3.14159265): '<',
#     (9.4, 4.71238898): 'e',
#     (10.4, 0.0): 'j',
#     (11.4, 1.57079633): 'o',
#     (12.4, 3.14159265): 't',
#     (13.4, 4.71238898): 'y',
#     (14.4, 0.0): '3',
#     (15.4, 1.57079633): '8',
#     (8.6, 4.71238898): 'a',
#     (9.6, 0.0): 'f',
#     (10.6, 1.57079633): 'k',
#     (11.6, 3.14159265): 'p',
#     (12.6, 4.71238898): 'u',
#     (13.6, 0.0): 'z',
#     (14.6, 1.57079633): '4',
#     (15.6, 3.14159265): '9',
#     (8.8, 0.0): 'b',
#     (9.8, 1.57079633): 'g',
#     (10.8, 3.14159265): 'l',
#     (11.8, 4.71238898): 'q',
#     (12.8, 0.0): 'v',
#     (13.8, 1.57079633): '0',
#     (14.8, 3.14159265): '5',
#     (15.8, 4.71238898): '_',
# }

def draw_keyboard(key_map=None, selected_freq=None, selected_phase=None):
    fig, ax = plt.subplots(figsize=(16, 6))  # Tamaño para ajustar el teclado

    key_layout = [
        "1234567890<",
        " qwertyuiop ",
        "  asdfghjkl  ",
        "   zxcvbnm.   ",
        "     _,    ",
    ]

    y_offset = 0
    for i, row in enumerate(key_layout):
        x_offset = (14 - len(row)) / 2 if i == 4 else (10 - len(row.strip())) / 2
        for key in row.strip():
            width = 6 if key == '_' else 1
            rect = patches.Rectangle((x_offset, y_offset), width, 1, linewidth=1, edgecolor='black', facecolor='none')
            ax.add_patch(rect)
            plt.text(x_offset + width/2, y_offset + 0.5, key, va='center', ha='center', fontsize=10)
            x_offset += width
        y_offset -= 1

    # Resaltar la tecla seleccionada si se proporcionan frecuencia y fase
    if key_map and selected_freq is not None and selected_phase is not None:
        selected_key = key_map.get((selected_freq, selected_phase))
        if selected_key:
            for y, row in enumerate(key_layout):
                x = (14 - len(row)) / 2 if y == 4 else (10 - len(row.strip())) / 2
                if selected_key in row.strip():
                    x += row.strip().index(selected_key)
                    width = 6 if selected_key == '_' else 1
                    rect = patches.Rectangle((x, -y * 1), width, 1, linewidth=2, edgecolor='red', facecolor='yellow', alpha=0.5)
                    ax.add_patch(rect)
                    break

    ax.set_xlim(-1, 11)
    ax.set_ylim(-5, 1)
    ax.axis('off')
    st.pyplot(fig)  # Mostrar el gráfico en Streamlit


# Botón para mostrar el punto de interés
if st.session_state.get('graficos_generados', False) and 'target_frequency' in st.session_state and 'target_phase' in st.session_state:
    if st.button('Mostrar Punto de Interés'):
        # Llamar a draw_keyboard con los valores almacenados
        draw_keyboard(key_map, st.session_state['target_frequency'], st.session_state['target_phase'])



      