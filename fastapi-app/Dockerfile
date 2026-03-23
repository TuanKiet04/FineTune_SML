# Dockerfile cho FastAPI News Chatbot
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ./app ./app

# Create non-root user for security
RUN useradd -m -u 1000 fastapi && chown -R fastapi:fastapi /app
USER fastapi

EXPOSE 8000

# Health check
# Sử dụng wget/curl tiêu chuẩn để kiểm tra endpoint /health
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget -q -O /dev/null http://localhost:8000/health || exit 1

# Run FastAPI with uvicorn
# Xóa "--reload"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]