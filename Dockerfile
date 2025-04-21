# Step 1: Use official Python image
FROM python:3.11-slim

# Step 2: Set working directory
WORKDIR /app

# Step 3: Copy all project files to /app folder inside container
COPY . /app/

# Step 4: Install latest pip
RUN pip install --upgrade pip

# Step 5: Install all required packages
RUN pip install -r requirements.txt

# Step 6: Run the Django app with Gunicorn
CMD ["gunicorn", "origin_underwriter.wsgi:application", "--bind", "0.0.0.0:8000"]