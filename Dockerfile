# Use official Python image
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    poppler-utils tesseract-ocr libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY . /app/

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose default Django port
EXPOSE 8000

# Here we detect $PORT or default to 8000 if not set
CMD ["sh", "-c", "gunicorn origin_underwriter.wsgi:application --bind 0.0.0.0:${PORT:-8000}"]