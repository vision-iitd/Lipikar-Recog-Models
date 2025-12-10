
# Use NVIDIA CUDA base image with Python
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv (modern Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --default-timeout=100 --retries=10 -r requirements.txt


# Copy project files
COPY . /app/

# Install your specific strhub dependencies if needed
# RUN uv pip install --system -e ./strhub

# Create directories for model and data
RUN mkdir -p /model /data

# Set default entrypoint
ENTRYPOINT ["python3", "infer.py"]

# Default command (can be overridden)
CMD ["--help"]


