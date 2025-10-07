FROM nvcr.io/nvidia/cuda:12.6.0-runtime-ubuntu24.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Update system packages and install Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python command
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /workspace

# Copy requirements file first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt --break-system-packages

# Default command (you can override this)
CMD ["/bin/bash"]