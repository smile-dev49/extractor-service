FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Copy mapping files
COPY mappings/ ./mappings/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV EXTRACTION_SERVICE_PORT=8091
ENV EXTRACTION_SERVICE_HOST=0.0.0.0
ENV MAPPINGS_DIR=/app/mappings
ENV MAX_FILE_SIZE_MB=50
ENV PYTHONPATH=/app:/app/app

# Expose port
EXPOSE 8091

# Keep working directory as /app (app module is here)
# The app will change to mappings directory automatically
WORKDIR /app

# Run the application from /app, Python will find app.main because PYTHONPATH=/app
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8091"]

