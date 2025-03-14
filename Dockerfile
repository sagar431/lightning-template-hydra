# Use PyTorch base image with CUDA support as specified in project requirements
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager for faster dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p checkpoints data/dog_breed/dataset

# Install Python dependencies using uv for better performance
RUN uv pip install -e .

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Expose port
EXPOSE 8080

# Default command (can be overridden by docker-compose)
CMD ["bash", "-c", "cd src && uvicorn app:app --host 0.0.0.0 --port 8080 --reload"]
