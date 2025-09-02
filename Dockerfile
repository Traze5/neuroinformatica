FROM python:3.11-slim

WORKDIR /app

# Dependencias del sistema (mínimas, suelen bastar para numpy/pandas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
ENV PORT=8000

# Ejecuta Streamlit
CMD ["python","-m","streamlit","run","main.py","--server.port","8000","--server.address","0.0.0.0"]
