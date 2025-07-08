# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Set working directory
WORKDIR /app

# Install PyTorch first
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch>=2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
RUN pip install --no-cache-dir \
    transformers>=4.36.0 \
    datasets>=2.14.0 \
    accelerate>=0.20.0 \
    peft>=0.7.0 \
    trl>=0.7.0 \
    bitsandbytes>=0.41.0 \
    scipy>=1.10.0 \
    scikit-learn>=1.3.0 \
    tqdm>=4.65.0 \
    pandas>=1.5.0 \
    numpy>=1.24.0 \
    sentence-transformers>=2.2.0 \
    wandb>=0.15.0 \
    tensorboard>=2.13.0

# Copy application files
COPY . .

# Create directories
RUN mkdir -p /app/outputs /app/logs

# Set permissions
RUN chmod +x *.py *.sh

# Keep container running
CMD ["tail", "-f", "/dev/null"] 