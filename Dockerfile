# Use micromamba for fast, reliable cross-language (Python + R) environment creation
FROM mambaorg/micromamba:1.5.1

# Set working directory
WORKDIR /app

# Copy the environment file
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/env.yaml

# Install dependencies defined in environment.yml
# This handles Python 3.9, R 4.3.3, PyTorch, and R packages (rugarch, etc.) together
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes

# Activate the environment for all subsequent commands
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Copy source code
COPY --chown=$MAMBA_USER:$MAMBA_USER . .

# Install the local package in editable mode
# We use --no-deps because micromamba already handled dependencies
RUN pip install --no-deps -e .

# Set environment variables
ENV PYTHONPATH=/app/src
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json

# Expose Jupyter port
EXPOSE 8888

# Default command: Launch Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]