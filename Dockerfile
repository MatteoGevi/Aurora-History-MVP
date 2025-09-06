# Dockerfile
FROM python:3.12-slim

# 1) System deps (add more if your pipeline needs them)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# 2) Poetry (no venvs inside the image)
ENV POETRY_HOME=/opt/poetry \
    POETRY_VERSION=1.8.3 \
    POETRY_VIRTUALENVS_CREATE=false \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
RUN curl -sSL https://install.python-poetry.org | python3 - && ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry

WORKDIR /app

# 3) Copy lockfiles and install deps first (better image caching)
COPY pyproject.toml poetry.lock* /app/
RUN poetry install --no-root --only main

# 4) Copy your source
COPY src /app/src
# (Optional) if you want data samples in the image (usually not for prod):
# COPY data /app/data

# 5) Expose port for FastAPI (push model). For pull model we'll override CMD.
EXPOSE 8000

# 6) Default command: run FastAPI worker (push model)
# Ensure src/worker.py defines `app = FastAPI()`
CMD ["python", "-m", "uvicorn", "src.worker:app", "--host", "0.0.0.0", "--port", "8000"]