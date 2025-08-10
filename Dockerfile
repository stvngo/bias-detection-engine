# Use Python 3.11 slim image for smaller size and security
# Force amd64 architecture for Google Cloud compatibility
FROM --platform=linux/amd64 python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create __init__.py files for proper Python package structure
RUN touch app/__init__.py && \
    touch app/core/__init__.py && \
    touch app/api/__init__.py && \
    touch app/models/__init__.py && \
    touch app/utils/__init__.py && \
    touch config/__init__.py

# Create data directories
RUN mkdir -p data/raw data/processed

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
