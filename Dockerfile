FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    install -m 0755 /root/.local/bin/uv /usr/local/bin/uv && \
    install -m 0755 /root/.local/bin/uvx /usr/local/bin/uvx

# Copy project files
COPY pyproject.toml uv.lock ./
COPY models.py __init__.py openenv.yaml ./
COPY server/ server/
COPY inference.py README.md requirements.txt ./

# Sync dependencies using uv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --no-editable 2>&1 && \
    uv sync --no-editable 2>&1

# Set environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"
# Disable web interface (avoids gradio dependency)
ENV ENABLE_WEB_INTERFACE=false

# Health check — use port 7860, matching HF Spaces app_port
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
