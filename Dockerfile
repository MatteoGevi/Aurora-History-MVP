# ─────────────────────────────────────────────
# Aurora — Dockerfile for Railway deployment
# ─────────────────────────────────────────────

FROM python:3.12-slim

# System dependencies
# poppler-utils: PDF rendering (pymupdf)
# libgomp1:      OpenMP runtime for spacy
# curl:          Poetry installer
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    poppler-utils \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Poetry — no virtualenvs inside the image
ENV POETRY_HOME=/opt/poetry \
    POETRY_VERSION=1.8.3 \
    POETRY_VIRTUALENVS_CREATE=false \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry

WORKDIR /app

COPY pyproject.toml poetry.lock* ./
RUN poetry lock && poetry install --no-root --only main

# Download SpaCy model required by the ingestion pipeline's text splitter
RUN python -m spacy download en_core_web_sm

# Copy application source
COPY app/    ./app/
COPY src/    ./src/
COPY ingest/ ./ingest/
COPY config/ ./config/

# Railway injects $PORT at runtime
EXPOSE 8501

CMD streamlit run app/app.py \
    --server.port=${PORT:-8501} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=true
