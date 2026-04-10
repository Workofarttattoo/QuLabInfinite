# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved.
# QuLabInfinite Production Docker Image

FROM python:3.11-slim AS base

LABEL maintainer="jhendrickscole@aios.is"
LABEL description="QuLabInfinite — Infinite Scientific Simulation Platform"
LABEL version="1.0.0"

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir . 2>/dev/null || \
    pip install --no-cache-dir \
    numpy scipy pydantic fastapi "uvicorn[standard]" pyyaml python-dotenv

# Copy application code
COPY qulab/ qulab/
COPY config.yaml* ./

# Create non-root user
RUN useradd --create-home qulab
USER qulab

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

EXPOSE 8000

CMD ["uvicorn", "qulab.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
