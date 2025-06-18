# Use the official Python 3.12 image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GEMINI_API_KEY=AIzaSyCVE2jmdMR_7hI4niOesXK5iyO-tnEXThQ

# Add labels for metadata
LABEL maintainer="Your Name <your.email@example.com>"
LABEL version="1.0"
LABEL description="Flask API for S3BERT sentence similarity analysis"

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install -U setuptools
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
