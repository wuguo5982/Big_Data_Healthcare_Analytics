# AWS Deployment Guide

This project supports container-based deployment to AWS through GitHub Actions.

## Recommended Deployment Options

### Option 1: AWS App Runner
Best for quickly deploying the Streamlit app from the ECR image.

### Option 2: Amazon ECS Fargate
Best for more production-style container orchestration.

---

## Required GitHub Secrets

Add these secrets in GitHub:

```text
AWS_ROLE_TO_ASSUME
AWS_ACCOUNT_ID
```

Recommended approach:
- create an IAM Role for GitHub Actions OIDC
- grant ECR push permissions
- optionally grant App Runner or ECS deployment permissions

---

## Required Runtime Secret

Do not commit your OpenAI API key.

Use:

```text
OPENAI_API_KEY
```

Store it in:
- AWS App Runner secret/environment variable, or
- AWS Secrets Manager for ECS Fargate

---

## CI/CD Flow

```text
GitHub Push
  ↓
Install dependencies
  ↓
Generate synthetic CMS-style data
  ↓
Build RAG index
  ↓
Run import tests
  ↓
Validate data
  ↓
Compile app files
  ↓
Run pytest
  ↓
Build Docker image
  ↓
Push image to Amazon ECR
  ↓
Deploy with App Runner or ECS
```
