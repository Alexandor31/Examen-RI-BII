FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    STREAMLIT_SERVER_PORT=7860 \
    AUTO_BUILD_INDEX=true

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p /app/data && chmod +x /app/scripts/build_index.py

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD curl --fail http://localhost:7860/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=7860"]
