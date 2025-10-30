# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
# QuLab AI Production Docker Image

FROM python:3.11-slim

LABEL maintainer="joshua@corporationoflight.com"
LABEL description="QuLabInfinite with QuLab AI Model Scaffold - Production Ready"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    pydantic \
    psutil \
    pint \
    jcamp \
    selfies \
    ase \
    biopython

# Copy application code
COPY . .

# Create logs directory
RUN mkdir -p /app/logs

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run API server
CMD ["python", "-m", "uvicorn", "api.production_api:app", "--host", "0.0.0.0", "--port", "8000"]
