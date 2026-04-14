FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=UTC

# System deps for numerical libs (lightgbm, scipy, hmmlearn)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Directories for persistent volumes on Railway
RUN mkdir -p /app/data/raw /app/data/processed /app/artifacts /app/logs

ENV RUN_MODE=shadow \
    POLL_SECONDS=30 \
    LOG_LEVEL=INFO

# Default command uses the unified entrypoint, Railway can override via CMD.
CMD ["python", "-m", "quant.entrypoint"]
