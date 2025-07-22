# Text-to-Image Retrieval Service

A **production-ready**, cross-modal retrieval system that enables fast, accurate image search using text queries through semantic similarity matching.

## Overview

This system combines state-of-the-art neural search with a user-friendly interface to deliver:

- **High-Performance ANN Search**: FastAPI service with FAISS IVF-Flat indexing for sub-second query response
- **Intuitive Web Interface**: Gradio-powered UI for seamless text-to-image search
- **Production-Ready Architecture**: Comprehensive testing, monitoring, and deployment automation

## Key Features

### **Efficient ANN Search API**
- FastAPI service with FAISS IVF-Flat index loaded at startup
- High-throughput, low-latency cosine similarity search
- RESTful endpoints with comprehensive health monitoring
- Handles 512-dimensional embeddings with configurable result limits

### **User-Friendly Interface**
- Gradio-powered web UI for intuitive text queries
- Responsive image gallery with thumbnail previews
- Adjustable result count via interactive slider
- One-click deployment to Hugging Face Spaces

### **Robust Development & Deployment**
- Comprehensive pytest suite with fixtures and mini-index testing
- Docker containerization with multi-stage builds
- Automated CI/CD pipeline integration
- Production deployments on Render.com and Hugging Face Spaces

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Gradio UI     │───▶│   FastAPI       │───▶│   FAISS Index   │
│   (Frontend)    │    │   (Backend)     │    │   (Vector DB)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Prerequisites

- Python 3.8+
- Docker (optional, for containerized deployment)
- Git

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/Capstone.git
cd Capstone/step12
```

### 2. Environment Setup

Create a `.env` file or set environment variables:

```bash
# Required paths
export FAISS_INDEX_PATH=/data/ivf_flat_1024.index
export META_PATH=/data/id2meta.json

# Optional configuration
export NPROBE=16
export PORT=8000
```

### 3. Installation & Deployment

#### Option A: Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Option B: Docker Deployment

```bash
# Build the container
docker build -t capstone-retrieval-api .

# Run with environment variables
docker run \
  -e FAISS_INDEX_PATH=/data/ivf_flat_1024.index \
  -e META_PATH=/data/id2meta.json \
  -p 8000:8000 \
  capstone-retrieval-api
```

## API Reference

### Health Check Endpoint

**GET** `/health`

Returns service status and index information:

```bash
curl https://capstone-retrieval-api.onrender.com/health
```

**Response:**
```json
{
  "status": "ok",
  "index_dim": 512,
  "nprobe": 16,
  "index_size": 1000
}
```

### Search Endpoint

**POST** `/search`

Performs semantic similarity search:

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "query_vec": [0.0, 0.1, 0.2, ...], 
    "k": 3
  }' \
  https://capstone-retrieval-api.onrender.com/search
```

**Parameters:**
- `query_vec`: Array of 512 float values (embedding vector)
- `k`: Number of results to return (1-50)

**Response:**
```json
{
  "results": [
    {
      "id": "image_001",
      "score": 0.95,
      "metadata": {
        "url": "https://example.com/image1.jpg",
        "caption": "A beautiful sunset over mountains"
      }
    }
  ]
}
```

## Testing

### Run Complete Test Suite

```bash
# Using Docker Compose
docker-compose -f docker-compose.test.yml up --build --exit-code-from tests

# Local testing
pytest tests/ -v
```

### Test Coverage

Our test suite covers:
- API endpoint validation
- FAISS index loading and querying
- Error handling and edge cases
- Performance benchmarks
- End-to-end integration tests

## 🌐 Production Deployment

### Render.com Deployment

1. **Create Web Service**
   - Navigate to [Render Dashboard](https://dashboard.render.com)
   - Click **"New +" → "Web Service"**
   - Connect your GitHub repository

2. **Service Configuration**
   ```yaml
   Name: capstone-retrieval-api
   Environment: Docker
   Branch: main
   Dockerfile Path: ./Dockerfile
   Port: 8000
   ```

3. **Environment Variables**
   ```bash
   FAISS_INDEX_PATH=/data/ivf_flat_1024.index
   META_PATH=/data/id2meta.json
   NPROBE=16
   ```

4. **Deploy & Verify**
   - Click **"Create Web Service"**
   - Monitor deployment logs
   - Test health endpoint: `curl https://your-service.onrender.com/health`

### Hugging Face Spaces (Gradio Frontend)

1. **Setup Space**
   - Create new Space on [Hugging Face](https://huggingface.co/spaces)
   - Choose Gradio as the SDK

2. **Configuration**
   ```bash
   # In Space Settings → Variables
   API_URL=https://capstone-retrieval-api.onrender.com
   ```

3. **Deploy Files**
   ```bash
   # Upload to your HF Space
   app.py
   requirements.txt
   ```

4. **Access Demo**
   - Live demo: `https://huggingface.co/spaces/<your-user>/retrieval-demo`

## Project Structure

```
step12/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py       # Pydantic models
│   └── routers/
│       ├── __init__.py
│       └── search.py        # Search endpoints
├── tests/
│   ├── __init__.py
│   ├── test_api.py         # API tests
│   └── fixtures/           # Test data
├── images/                 # Documentation screenshots
├── Dockerfile              # Container configuration
├── docker-compose.test.yml # Test environment
├── requirements.txt        # Python dependencies
├── app.py                 # Gradio frontend
└── README.md              # This file
```

## Development

### Code Quality Standards

- **Type Hints**: All functions include comprehensive type annotations
- **Documentation**: Detailed docstrings following Google style
- **Testing**: Minimum 90% code coverage required
- **Linting**: Pre-commit hooks with black, flake8, and mypy

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## Troubleshooting

### Common Issues

**Q: Service fails to start with "Index not found" error**
```bash
# Verify your data paths
ls -la /data/ivf_flat_1024.index
ls -la /data/id2meta.json

# Check environment variables
echo $FAISS_INDEX_PATH
echo $META_PATH
```

**Q: Search returns empty results**
```bash
# Verify index dimensions match embedding size
curl https://your-service.onrender.com/health

# Check query vector format (must be exactly 512 dimensions)
```

**Q: High latency on search requests**
```bash
# Adjust NPROBE value (higher = more accurate, slower)
export NPROBE=32
```

## Performance Metrics

- **Search Latency**: < 100ms for typical queries
- **Throughput**: 1000+ queries/second
- **Index Size**: Supports millions of vectors
- **Memory Usage**: ~2GB for 1M vectors

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
- [FastAPI](https://fastapi.tiangolo.com/) for the robust API framework
- [Gradio](https://gradio.app/) for the intuitive web interface
- [Render](https://render.com/) and [Hugging Face](https://huggingface.co/) for hosting platforms

---

## Screenshots

### Deployment Dashboard
![Render Deployment Logs](images/Screenshot%202025-07-14%20004518.png)

### Search Interface
![Retrieval Results](images/Screenshot%202025-07-14%20004105.png)

### File Structure
![HF Space Files](images/Screenshot%202025-07-14%20004348.png)

