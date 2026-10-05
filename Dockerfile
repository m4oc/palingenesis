# syntax=docker/dockerfile:1.7
# CUDA 12.9 libraries are supplied by the locked PyTorch wheels.
FROM ghcr.io/astral-sh/uv:0.11.6 AS uv
FROM python:3.12-slim-bookworm AS runtime
# Inductor/Triton compile host launchers at runtime, so a C/C++ compiler is needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd --gid 10001 trainer && \
    useradd --uid 10001 --gid 10001 --create-home trainer && \
    mkdir -p /workspace /cache /outputs && \
    chown trainer:trainer /workspace /cache /outputs
# Install directly in the runtime stage to avoid duplicating multi-GB CUDA layers.
# Bind mounts keep sources and the installer out of the resulting image.
WORKDIR /app
RUN --mount=type=bind,from=uv,source=/uv,target=/usr/local/bin/uv \
    --mount=type=bind,source=.,target=/source \
    UV_PROJECT_ENVIRONMENT=/app/.venv UV_PYTHON_DOWNLOADS=never \
    uv sync --project /source --locked --no-dev --extra train --no-editable --no-cache
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/cache/huggingface TORCH_HOME=/cache/torch \
    TORCHINDUCTOR_CACHE_DIR=/cache/inductor TRITON_CACHE_DIR=/cache/triton \
    XDG_CACHE_HOME=/cache
WORKDIR /workspace
USER 10001:10001
ENTRYPOINT ["pgs"]
CMD ["--help"]
