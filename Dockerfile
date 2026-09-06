# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Add local user to PATH
ENV PATH=/root/.local/bin:$PATH

# Create non-root user
RUN groupadd -r panda && useradd -r -g panda panda \
    && mkdir -p /app/data /app/logs \
    && chown -R panda:panda /app

# Copy application code
COPY --chown=panda:panda *.py ./
COPY --chown=panda:panda docs/ ./docs/

# Switch to non-root user
USER panda

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OLLAMA_HOST=http://ollama:11434 \
    LLM_PROVIDER=ollama \
    MODEL_NAME=llama3:latest \
    LOG_LEVEL=INFO \
    LOG_FILE=/app/logs/agent.log

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]