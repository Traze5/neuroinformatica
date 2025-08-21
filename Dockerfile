# ---- Imagen base ligera con Python 3.11 ----
FROM python:3.11-slim

# Evitar prompts interactivos y mejorar logs
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ---- Dependencias de sistema mínimas (para compilar algunas wheels) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---- Directorio de trabajo y usuario no-root ----
ENV APP_HOME=/app
WORKDIR $APP_HOME
RUN useradd -m -u 1000 appuser

# ---- Instalar dependencias Python primero (mejor cache) ----
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---- Copiar el resto del código ----
COPY . $APP_HOME
RUN chown -R appuser:appuser $APP_HOME
USER appuser

# ---- Config Streamlit para contenedores ----
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    PORT=8501

EXPOSE 8501

# Si tu entrypoint es main.py, cambia "streamlit_app.py" por "main.py"
CMD ["bash","-lc","streamlit run main.py --server.port ${PORT} --server.address 0.0.0.0"]
