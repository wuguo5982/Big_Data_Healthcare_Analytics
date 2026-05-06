# AWS Deployment Guide

This project includes CI/CD support for GitHub Actions and AWS.

## Recommended AWS Deployment Options

### Option 1: AWS App Runner
Best for simple deployment of the Streamlit app from an ECR image.

### Option 2: Amazon ECS Fargate
Best for more production-oriented container deployment.

### Option 3: EC2
Simple but more manual operational overhead.

---

## GitHub Actions AWS Secrets

Add these GitHub repository secrets:

```text
AWS_ROLE_TO_ASSUME
AWS_ACCOUNT_ID
```

Recommended AWS region is configured in:

```text
.github/workflows/ci-cd.yml
```

Default:

```text
us-east-1
```

---

## OpenAI API Key

Do not commit your OpenAI API key.

For AWS deployment, store it in:

- AWS Secrets Manager, or
- App Runner environment secret, or
- ECS task secret

Required variable:

```text
OPENAI_API_KEY
```

---

## CI/CD Flow

```text
GitHub Push
  ↓
Install dependencies
  ↓
Run import tests
  ↓
Validate data
  ↓
Compile Streamlit app
  ↓
Run pytest
  ↓
Build Docker image
  ↓
Push Docker image to Amazon ECR
  ↓
Deploy using App Runner or ECS Fargate
```
