# Use the official NVIDIA CUDA image with Python as the base image
FROM cuda:12.9.1-cudnn-devel-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    GEMINI_API_KEY=AIzaSyCVE2jmdMR_7hI4niOesXK5iyO-tnEXThQ

# Add labels for metadata
LABEL maintainer="Your Name <your.email@example.com>"
LABEL version="1.0"
LABEL description="Flask API for S3BERT sentence similarity analysis"

# Set timezone
RUN ln -sf /usr/share/zoneinfo/UTC /etc/localtime && \
    echo UTC > /etc/timezone

# Add deadsnakes PPA for Python 3.12
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update

# Install system dependencies
RUN apt-get install -y --no-install-recommends \
    build-essential \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3.12-distutils \
    python3-pip \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set up Python alias
RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download the pre-trained S3BERT model
RUN curl -L -o s3bert_model.tar.gz https://www.cl.uni-heidelberg.de/~opitz/data/s3bert_all-mpnet-base-v2.tar.gz && \
    tar -xzf s3bert_model.tar.gz -C . && \
    rm s3bert_model.tar.gz

# Copy the application code
COPY . .

# Expose port
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]
