# Use the official NVIDIA CUDA image with Python as the base image
FROM nvidia/cuda:11.4.3-base-ubuntu20.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Add labels for metadata
LABEL maintainer="Your Name <your.email@example.com>"
LABEL version="1.0"
LABEL description="Flask API for S3BERT sentence similarity analysis"

# Set timezone
RUN ln -sf /usr/share/zoneinfo/UTC /etc/localtime && \
    echo UTC > /etc/timezone

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3.8 \
    python3.8-dev \
    python3.8-venv \
    python3.8-distutils \
    python3-pip \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set up Python alias
RUN ln -sf /usr/bin/python3.8 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download the pre-trained S3BERT model
RUN mkdir -p s3bert_all-MiniLM-L12-v2 && \
    curl -L -o s3bert_model.tar.gz https://www.cl.uni-heidelberg.de/~opitz/data/s3bert_all-MiniLM-L12-v2.tar.gz && \
    tar -xzf s3bert_model.tar.gz -C s3bert_all-MiniLM-L12-v2 && \
    rm s3bert_model.tar.gz

# Copy the application code
COPY . .

# Expose port
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]