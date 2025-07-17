# Step 9: Deployment Plan

This step outlines how we deploy our **image-to-text retrieval service** at production scale, using an automated offline pipeline, containerized inference API, and real-time monitoring. The service supports sub-150ms response times and automated retraining via CI/CD and AWS infrastructure.

---

## Executive Summary

We serve a **text-to-image-to-caption** retrieval model (ViT-B-32 + FAISS IVF-Flat) behind a **FastAPI service** deployed via **AWS ECS Fargate**.  

Key components:
- CI/CD pipeline: GitHub Actions → Docker → Amazon ECR → ECS Fargate
- Nightly batch jobs: Recompute embeddings + FAISS index using AWS Batch
- Monitoring: Prometheus + CloudWatch
- Cost-efficient: ~$5.82/month for full pipeline

---

## Deployment Options Considered

| Option | Pros | Cons | Verdict |
|--------|------|------|--------|
| **Lambda + API Gateway** | Zero-ops, cheap idle | 512MB RAM, no GPU, 5s cold starts | ❌ Too slow |
| **SageMaker Endpoints** | Managed A/B testing | Expensive, vendor lock-in | ❌ Overkill |
| **Kubernetes (GKE)** | Full control | High ops burden | ❌ Too heavy |
| **ECS Fargate** | Pay-per-second, IAM-native, auto-scale | Slightly slower scale-out | ✅ **Chosen** |
| **On-prem GPU** | No cloud spend | High CapEx, ops overhead | ❌ Not practical |

---

## Chosen Architecture

A fully managed pipeline from S3 → Embeddings → Retrieval API:

1. **Offline Batch Job** (AWS Batch)
   - Pull raw images & captions from S3
   - Encode with ViT-B-32
   - Write FAISS index + embeddings back to S3

2. **CI/CD (GitHub Actions)**
   - On push to `main`: build Docker image with FastAPI + FAISS
   - Push to Amazon ECR
   - Trigger ECS deployment (Fargate)

3. **Online Serving**
   - Requests → ALB → API Gateway → FastAPI container
   - FastAPI loads IVF index in RAM and performs ANN search
   - Results (IDs) → DynamoDB metadata → JSON response

4. **Monitoring & Scaling**
   - Prometheus sidecar scrapes latency, QPS, FAISS metrics
   - CloudWatch dashboards + alerts
   - Fargate auto-scales on CPU / memory / p95 latency

---

## Repository Structure

```
step9_deployment/
├── .github/workflows/deploy.yml   # CI/CD GitHub Actions
├── inference.py                   # FastAPI server with FAISS ANN
├── Dockerfile
├── terraform/                     # (optional) Infra-as-code
└── README.md
```

---

## Serving Logic

FastAPI startup:
- Load FAISS IVF index (`ivf_flat.index`)
- Preload into RAM (~2 GB)

Request flow:
1. Input (text or image) → Tokenize → Embed (CLIP)
2. `faiss.search(k)` returns top IDs + distances
3. Lookup metadata in DynamoDB → return JSON payload

---

## CI/CD Pipeline

`.github/workflows/deploy.yml`:
- Trigger: push to `main`
- Actions:
  - Checkout code
  - Build Docker image
  - Push to ECR
  - Trigger ECS (or EKS) rolling deploy

---

## Monitoring & Observability

| Metric                     | Tool        |
|---------------------------|-------------|
| p95 Latency, QPS          | Prometheus  |
| Error Rate (4xx/5xx)      | CloudWatch  |
| Embedding drift (cosine)  | Custom Job  |
| Canary recall             | Scheduled tests |
| GPU memory (if enabled)   | Prometheus  |

SLOs:
- Latency (p95): ≤ 150 ms
- Availability: ≥ 99.5%

---

## Retraining & Rollback

- **Trigger:** Weekly schedule or drift > 10% (3 days)
- **Pipeline:**
  - Retrain embeddings → New FAISS index
  - Blue/Green deploy via ECS task sets
  - Rollback: toggle Terraform `deploy_version` or use old ECR tag
- **Version control:** ECR tags + deploy metadata

---

## Estimated Monthly Cost

| Item                    | Rate            | Est. Cost |
|-------------------------|------------------|-----------|
| ECS Fargate (40 hrs/mo) | $0.04/vCPU-hr    | $1.76     |
| S3 Storage (20 GB)      | $0.023/GB        | $0.46     |
| AWS Batch GPU (4 hrs)   | $0.526/hr        | $2.10     |
| CloudWatch logs/metrics | Approx.          | $1.50     |
| **Total**               |                  | **$5.82** |

---

## Post-Deployment Care

- SLO: p95 latency ≤ 150 ms, uptime ≥ 99.5%
- Monitor: latency, recall canary, 4xx/5xx, drift, GPU usage
- Retrain: weekly or on drift
- Rollout: blue/green ECS task sets
- Rollback: revert ECR tag in Terraform
- Docs: runbook, on-call schedule, AWS cost dashboard

---

## 🏁 Conclusion

Our deployment plan offers a cost-effective, scalable, and maintainable solution for real-time multimodal retrieval. With CI/CD, nightly embeddings, full observability, and recovery plans, the system is robust for production workloads and research iterations alike.

