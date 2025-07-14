# Use a lightweight Python image
FROM python:3.10-slim

# Set working dir
WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your FastAPI app code
COPY app/ ./app

# Copy the small‐index fixtures into /data so the service finds them at runtime
RUN mkdir -p /data
COPY tests/fixtures/data_small/ivf_flat_small.index /data/ivf_flat_1024.index
COPY tests/fixtures/data_small/id2meta_small.json    /data/id2meta.json
COPY tests/fixtures/data_small/img_embs_small.npy     /data/img_embs.npy

# Expose the FastAPI port
EXPOSE 8000

# Launch Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
