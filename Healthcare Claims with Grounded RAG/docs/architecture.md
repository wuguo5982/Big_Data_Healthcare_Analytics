# Architecture

```text
Raw Claims + CMS-style Policy + Reference Tables
        |
        v
Schema Validation + Feature Engineering
        |
        v
Supervised Risk Model + Isolation Forest Anomaly Model
        |
        v
Agentic FWA Workflow
        |-- Provider profile lookup
        |-- Procedure benchmark lookup
        |-- Diagnosis policy lookup
        |-- Rule checks
        |-- ML risk scoring
        |-- Anomaly detection
        |-- RAG policy retrieval
        |-- Evidence-grounded explanation
        |-- Hallucination check
        |-- Safety guardrails
        v
FastAPI + Streamlit
        |
        v
Docker
        |
        v
GitHub Actions / CodeBuild
        |
        v
AWS ECR → ECS Fargate → CloudWatch
```
