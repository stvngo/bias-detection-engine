# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Create requirements.txt with all dependencies
RUN echo "fastapi==0.104.1" > requirements.txt && \
    echo "uvicorn[standard]==0.24.0" >> requirements.txt && \
    echo "pydantic==2.5.0" >> requirements.txt && \
    echo "pydantic-settings==2.1.0" >> requirements.txt && \
    echo "anthropic==0.7.8" >> requirements.txt && \
    echo "openai==1.3.7" >> requirements.txt && \
    echo "sentence-transformers==2.2.2" >> requirements.txt && \
    echo "scikit-learn==1.3.2" >> requirements.txt && \
    echo "redis==5.0.1" >> requirements.txt && \
    echo "aiohttp==3.9.1" >> requirements.txt && \
    echo "httpx==0.25.2" >> requirements.txt && \
    echo "python-dotenv==1.0.0" >> requirements.txt && \
    echo "structlog==23.2.0" >> requirements.txt && \
    echo "rich==13.7.0" >> requirements.txt && \
    echo "typer==0.9.0" >> requirements.txt && \
    echo "feedparser==6.0.10" >> requirements.txt

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

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
