FROM tensorflow/tensorflow:2.10.0-gpu

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose API port
EXPOSE 8000

# Default command
CMD ["python", "serve.py", "--model_path", "./checkpoints/best_model.h5", "--host", "0.0.0.0", "--port", "8000"]







