# Base image với Python 3.10
FROM python:3.10.18-slim-bullseye

# Set working directory
WORKDIR /app

# Copy requirements trước để tận dụng Docker cache
COPY requirements.txt .

# Cài đặt Python dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy toàn bộ application code
COPY . .


# Start command
CMD ["python", "-u", "main.py"]