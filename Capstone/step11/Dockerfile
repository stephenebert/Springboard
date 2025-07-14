# 1. Base image
FROM python:3.10-slim

# 2. Create app directory
WORKDIR /app

# 3. Make sure pip is up to date
RUN pip install --no-cache-dir --upgrade pip

# 4. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your application code
COPY app/ ./app

# copy the small‐fixture files into /data and rename them
#     to match the production defaults
COPY tests/fixtures/data_small/ivf_flat_small.index  /data/ivf_flat_1024.index
COPY tests/fixtures/data_small/id2meta_small.json   /data/id2meta_1024.json

# 6. Expose port & launch
EXPOSE 8000
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000","--log-level","info"]
