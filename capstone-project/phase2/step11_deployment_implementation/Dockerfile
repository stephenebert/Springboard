FROM python:3.10-slim

# 1) Set workdir
WORKDIR /app

# 2) Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 3) Copy code
COPY app/ ./app

# 4) Copy the small‐index fixtures into /data
#    Adjust paths if kept them somewhere else in repo
COPY tests/fixtures/data_small/ivf_flat_small.index  /data/ivf_flat_1024.index
COPY tests/fixtures/data_small/id2meta_small.json     /data/id2meta.json

# 5) Expose port and launch
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
