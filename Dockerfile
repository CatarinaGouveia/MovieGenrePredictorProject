# Use a Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (and system libraries for LightGBM)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    libomp-dev \
    gfortran \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create models directory
RUN mkdir -p models

# Copy models directory
COPY models /app/models/

# Copy the rest of the application
COPY . .

# Expose ports for Streamlit
EXPOSE 8000

# Create a script to run the service
RUN echo '#!/bin/bash' > /app/start.sh && \
    echo 'streamlit run app.py --server.port 8000 --server.address 0.0.0.0' >> /app/start.sh && \
    chmod +x /app/start.sh

# Set the correct PATH
ENV PATH="/usr/local/bin:${PATH}"

# Command to run the application
CMD ["./start.sh"] 