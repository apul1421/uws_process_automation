# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for OCR and others
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    poppler-utils tesseract-ocr libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app/

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port 8000 (Django default, but Railway will override with $PORT)
EXPOSE 8000

# Final Run Command — Fixes $PORT issue
CMD ["sh", "-c", "gunicorn origin_underwriter.wsgi:application --bind 0.0.0.0:$PORT"]