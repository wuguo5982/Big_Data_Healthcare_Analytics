# Hallucination Reduction Strategy for the Agentic FWA Assistant

This project is intentionally designed as a compliance decision-support system, not an autonomous fraud adjudication system. In healthcare FWA analytics, hallucination risk is especially important because an unsupported statement can create legal, compliance, provider-reputation, or patient-care risk.

## Core Design Principle

The agent should only produce an evidence-grounded risk summary. It should never state that a provider committed fraud, never make a final legal determination, and never invent policy details that are not present in retrieved evidence, deterministic rules, or structured reference tables.

## Methods Implemented in This Project

### 1. Retrieval-Augmented Generation Grounding

The workflow retrieves CMS-style policy evidence, procedure benchmarks, and diagnosis-policy mappings before creating the final explanation. The explanation displays the retrieved evidence chunks and similarity scores so reviewers can inspect the source context.

Implemented in:

```text
src/rag_engine.py
src/agentic_fwa_workflow.py
```

### 2. Template-Based Explanation Instead of Free-Form Generation

For the local GitHub demo, the final explanation is template-based. This is safer than free-form LLM generation because the output is assembled from known claim fields, deterministic rule flags, ML/anomaly outputs, and retrieved evidence.

Production upgrade:

- Use a foundation model only inside a strict prompt template.
- Require the model to cite retrieved chunk IDs for every factual statement.
- Reject output that contains unsupported claims.

### 3. Retrieval-Quality Gate

The agent checks whether enough retrieved chunks meet a minimum similarity score. If retrieval is weak, the agent marks the case for abstention or human review rather than confidently answering with weak evidence.

Implemented in:

```text
src/tools.py -> HallucinationReductionTool
```

Key config values:

```text
MIN_RETRIEVAL_SCORE
MIN_EVIDENCE_CHUNKS
```

### 4. Claim-Fact Support Check

The agent checks whether key facts such as provider ID, procedure code, and diagnosis code are supported by either retrieved evidence or structured reference context. This helps avoid explanations that drift away from the actual claim.

### 5. Unsupported High-Risk Assertion Detection

The agent scans for risky phrases such as:

```text
intentional fraud
confirmed fraud
definitely fraud
criminal
guilty
final determination
```

If these phrases appear without support, the workflow flags them and recommends abstention/human review.

### 6. Safety Guardrails

The safety guardrail blocks legally risky language. The assistant is allowed to identify risk indicators but not allowed to make final fraud accusations.

Implemented in:

```text
src/tools.py -> SafetyGuardrailTool
```

### 7. Human-in-the-Loop Escalation

The agent escalates to human review when any of the following is true:

- ML risk is medium or high.
- The claim is anomalous.
- The claim is above the procedure benchmark expected maximum.
- Citation coverage is weak.
- Hallucination-reduction checks recommend abstention.
- Safety guardrails detect risky language.

This is appropriate for healthcare compliance because AI should prioritize review, not replace reviewer judgment.

## Production-Grade Upgrades

For a real enterprise deployment, extend this project with:

1. **RAGAS or TruLens evaluation** for faithfulness, context precision, context recall, and answer relevance.
2. **LLM-as-judge evaluation** using a golden evaluation set with known expected answers.
3. **Amazon Bedrock Guardrails** for policy-based filtering, contextual grounding checks, and unsafe-output blocking.
4. **Bedrock Knowledge Bases or OpenSearch Serverless** for scalable retrieval with metadata filters by policy type, date, state, payer, and procedure code.
5. **Citation-required prompting** so every generated claim includes a source chunk ID.
6. **Abstention policy** requiring the model to answer: "Insufficient evidence; escalate to human review" when retrieval is weak.
7. **Audit logging** to S3/CloudWatch with immutable retention for compliance review.
8. **CI/CD evaluation gate** that blocks deployment if faithfulness, citation coverage, or F1 score falls below the threshold.

## Interview Talking Point

A strong answer for interviews:

> I reduce hallucination using a layered approach: retrieval grounding, deterministic rules, structured reference checks, template-based response generation, citation coverage scoring, retrieval-quality gates, unsupported-assertion detection, safety guardrails, abstention logic, and human-in-the-loop escalation. The system does not make final fraud determinations; it prioritizes claims for compliance review with transparent evidence.
