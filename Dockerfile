# ─────────────────────────────────────────────
# Aurora — Dockerfile for Railway deployment
# Uses Poetry with native [tool.poetry.dependencies] format
# ─────────────────────────────────────────────

FROM python:3.12-slim

# System dependencies
# poppler-utils: PDF rendering (pymupdf)
# libgomp1:      OpenMP runtime for sentence-transformers
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

# Install ONLY production dependencies — excludes [dev] group (jupyter, ipykernel)
COPY pyproject.toml poetry.lock* ./
# Install CPU-only PyTorch before Poetry runs.
# sentence-transformers depends on torch; without this, pip pulls the full
# CUDA build (~2 GB of nvidia-* wheels) which makes the image too large to push.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

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