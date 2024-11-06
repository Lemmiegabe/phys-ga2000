# Use Python 3.9.13 as the base image
FROM python:3.9.13

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
RUN pip install pandas matplotlib scipy "jax[cpu]"

# Set working directory
WORKDIR /Dockerfile

# Create an output directory in the container
RUN mkdir -p /Dockerfile/image

# Copy all files from current directory to /Dockerfile in container
COPY . /Dockerfile

# Default command to run the Python script
CMD ["python", "Problem-1.py"]
