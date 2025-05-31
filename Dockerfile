# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install system dependencies for LightGBM, scikit-learn, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Set environment variables (if needed)
ENV PYTHONUNBUFFERED=1

# Command to run the API (change to your actual entrypoint)
CMD ["python", "main.py"]
