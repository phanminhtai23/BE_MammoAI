# Base image với Python 3.11
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Cài đặt system dependencies cần thiết cho AI/ML
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    wget \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements trước để tận dụng Docker cache
COPY requirements.txt .

# Cài đặt Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ application code
COPY . .

# Tạo các thư mục cần thiết
RUN mkdir -p /app/model_cache \
    && mkdir -p /app/logs \
    && mkdir -p /app/temp

# Set environment variables
ENV MODEL_CACHE_DIR=/app/model_cache
ENV MODEL_CACHE_PERSISTENT=true
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port (sử dụng PORT từ config hoặc default 8000)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=60s --timeout=60s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]