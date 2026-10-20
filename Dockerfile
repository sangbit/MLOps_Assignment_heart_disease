# Container image for the Heart Disease Risk API.
#
# Build:  docker build -t heart-disease-api:latest .
# Run:    docker run -d -p 8000:8000 --name heart-api heart-disease-api:latest
# Test:   curl http://localhost:8000/health

# ---- Base image: the same Python version used for training (reproducibility) ----
FROM python:3.13-slim

# Python behaviour inside containers: no .pyc files, logs printed immediately
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ---- Dependencies first: this layer is cached and only rebuilt when requirements change ----
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

# ---- Application code and the trained model ----
COPY src/ ./src/
COPY models/ ./models/

# ---- Security: run as an unprivileged user, not root ----
RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Docker checks this every 30 s; the container shows as "healthy" once /health answers
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

# --no-access-log: the app writes its own structured JSON request logs (see src/monitoring.py)
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
