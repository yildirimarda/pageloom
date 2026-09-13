# Docker container that runs identically on CPU or CUDA.
# The same Python codebase is used regardless of runtime backend.
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies needed for Pillow and diffusers
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv and dependencies
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev

# Copy source and install in editable mode
COPY src/ ./src/
RUN uv pip install -e . --no-deps

# Persist Hugging Face model weights across container restarts.
# Mount a host volume to avoid re-downloading weights on every run:
#   docker run ... -v ~/.cache/huggingface:/root/.cache/huggingface pageloom
ENV HF_HOME=/root/.cache/huggingface
RUN mkdir -p /root/.cache/huggingface
VOLUME ["/root/.cache/huggingface"]

# The same code runs on CPU or CUDA with no behavioral differences
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}

ENTRYPOINT ["python", "-m", "pageloom.pipeline"]
