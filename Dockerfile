# Base image với Python 3.10
FROM python:3.10.18-slim-bullseye

# Set working directory
WORKDIR /app

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

# Start command
CMD ["python", "-u", "main.py"]