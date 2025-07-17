# Step 10: Deployment Architecture

This README outlines the production-grade deployment architecture for our cross-modal retrieval system, following the AWS-native strategy decided in Step 9. It includes system design, component breakdown, CI/CD pipelines, monitoring, tools, and estimated costs.

---

## 1. Introduction
We deploy our FastAPI + FAISS retrieval service using:
- **AWS ECS Fargate** (inference)
- **Amazon S3** (embeddings + assets)
- **GitHub Actions + ECR** (CI/CD)

Step 10 converts the prototype into a production-ready pipeline, with monitoring, automated rollbacks, and scalable inference.

---

## 2. Architecture Diagram

[Architecture Diagram (Mermaid)](https://www.mermaidchart.com/app/projects/96c849f6-3985-443f-bd40-d3ad458b833d/diagrams/13031856-5897-437b-862a-53f5bb049522/version/v0.1/edit)

This diagram illustrates:
- S3 ingestion to Fargate inference service  
- CI/CD pipeline from GitHub → ECR → ECS  
- Monitoring with Prometheus & CloudWatch

---

## 3. Major Components & Data Flow

### 3.1 Data Ingestion & ETL
- **Source**: Raw images + captions land in S3 ingest bucket  
- **Batch Job**: Daily AWS Batch runs `scale_pipeline_hdf5.py`  
- **Output**: Writes `embeddings_full.h5` to `s3://.../embeddings/`

### 3.2 CI/CD & Container Registry
- **GitHub**: Commits to `main` trigger GitHub Actions  
- **Docker Image**: Bundles FastAPI app + FAISS index logic  
- **Push**: Image pushed to ECR (Elastic Container Registry)

### 3.3 Inference Service
- **ECS Fargate**: Pulls image, deploys container behind ALB  
- **API Gateway**: Handles auth, throttling, request routing  
- **FAISS IVF**: Loads IVF index from S3 or EFS on startup

---

## 4. Monitoring & Metrics

- **Prometheus** (sidecar): collects `latency`, `QPS`, `error rate`  
- **CloudWatch**: logs, aggregates, and alerts  
- **Alarms**:
  - p95 latency > 150ms  
  - recall drift below threshold (canary)  
  - container crash or unhealthy status  

---

## 5. Model Lifecycle

### Training & Evaluation
- **Offline**: Precomputed embeddings, IVF index via batch

### Retraining Cadence
- **Scheduled**: Weekly job refreshes embeddings + index  
- **Conditional**: Canary triggers rebuild if recall < 80%

### Artifact Management
- **Embeddings**: `s3://.../embeddings/v2025-07-01/`  
- **Images**: Tagged `:v2025-07-01` in ECR

### Deployment
- **Promoted**: GitHub Action tags image `prod` and updates ECS  
- **Rollback**: Terraform-managed variable to point to last good tag

---

## 6. Tools & Technologies

| Layer             | Service / Tool              | Purpose                                 |
|------------------|-----------------------------|-----------------------------------------|
| Compute          | AWS Fargate                 | Serverless inference                    |
| Storage          | Amazon S3                   | Embeddings, assets, versioned backups   |
| CI/CD            | GitHub Actions + ECR        | Build/test/push images                  |
| Batch ETL        | AWS Batch / Step Functions  | Offline embedding + index job           |
| API Gateway      | ALB + API Gateway           | Routing + Throttling                    |
| Monitoring       | Prometheus + CloudWatch     | Logs, metrics, canary alarms            |
| Infra as Code    | Terraform                   | Automated rollouts + rollback control   |

---

## 7. Implementation Effort & Cost Estimate

- **Setup**: 4–8 hours (Terraform, CI/CD, dashboards)
- **Daily Runtime Cost**:
  - Fargate CPU/RAM: ~$2/day under moderate load  
  - S3 storage: ~$0.50/month  
  - Optional GPU batch jobs (extra)

---

## 8. Scalability & Fault Tolerance

- **Autoscaling**: ECS adjusts task count via CPU/memory/latency  
- **Multi-AZ**: ALB routes across availability zones  
- **Graceful Failure**: ALB + container health checks + retry logic  
- **Backfill**: Batch job reprocesses late/corrected data  
- **Versioning**: S3 prefixes + ECR tags = safe rollbacks

---

This architecture is production-ready: scalable, cost-effective, monitorable, and easy to roll back when needed.
