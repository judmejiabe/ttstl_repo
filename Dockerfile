# Use micromamba for fast, reliable cross-language (Python + R) environment creation
FROM mambaorg/micromamba:1.5.1

# Set working directory
WORKDIR /app

# Copy the environment file(s)
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/env.yaml
COPY --chown=$MAMBA_USER:$MAMBA_USER environment_relaxed.yml /tmp/env_relaxed.yaml

# Install dependencies with simplified failover
RUN \
    echo "⬇️  Attempting exact installation from environment.yml..." && \
    if micromamba install -y -n base -f /tmp/env.yaml; then \
    echo "✅ Exact installation succeeded!"; \
    else \
    echo "⚠️  Exact installation failed (likely due to platform-specific build strings)." && \
    echo "⬇️  Attempting installation from environment_relaxed.yml..." && \
    if micromamba install -y -n base -f /tmp/env_relaxed.yaml; then \
    echo "✅ Relaxed installation succeeded!"; \
    else \
    echo "❌ All installation attempts failed." && exit 1; \
    fi; \
    fi && \
    micromamba clean --all --yes

# Activate the environment for all subsequent commands
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Copy source code
COPY --chown=$MAMBA_USER:$MAMBA_USER . .

# Install the local package in editable mode
RUN pip install --no-deps -e .

# Set environment variables
ENV PYTHONPATH=/app/src
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json

# Expose Jupyter port
EXPOSE 8888

# Default command: Launch Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]