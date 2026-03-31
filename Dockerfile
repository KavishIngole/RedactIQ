FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY configs/ configs/
COPY redactiq/ redactiq/
COPY scripts/ scripts/
COPY tests/ tests/

# Install Python dependencies
RUN pip install --no-cache-dir -e ".[dev]"

# Expose ports: API (8000) and Gradio UI (7860)
EXPOSE 8000 7860

# Default command: run the API server
CMD ["python", "-m", "redactiq.serving.api"]
