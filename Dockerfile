FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install R and R packages
RUN apt-get update && apt-get install -y \
    r-base \
    r-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Install R packages
RUN R -e "install.packages(c('rugarch', 'xts', 'zoo'), repos='https://cran.rstudio.com/')"

# Copy requirements first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install the package
RUN pip install -e .

# Set environment variables
ENV PYTHONPATH=/app/src
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json

# Expose port for Jupyter
EXPOSE 8888

# Default command
CMD ["python", "-m", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
