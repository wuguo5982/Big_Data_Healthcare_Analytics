
# Improved Healthcare FWA with Grounded RAG

## Production Enterprise Integration

This project extends the original Stage1 healthcare FWA workflow into a more advanced enterprise-ready GenAI architecture.

### Added Enterprise Features
- Grounded RAG retrieval
- Retrieval reranking
- Hallucination scoring
- Validation layer
- LLM-as-judge evaluation
- Small-LLM workflow
- PEFT/QLoRA-ready fine-tuning
- CI/CD-ready architecture

---

## Recommended Small Models

To reduce local GPU/compute requirements:

- GPT-4o-mini
- Phi-3-mini
- TinyLlama

These models are suitable for:
- low-cost inference
- lightweight fine-tuning
- local experimentation
- PEFT/QLoRA workflows

---

## Final Enterprise Workflow

User Query
→ Retriever
→ Reranker
→ Grounded CMS Context
→ Small Fine-Tuned LLM
→ Validation Layer
→ Hallucination Detection
→ LLM-as-Judge Evaluation
→ Final Grounded Response

---

## Hallucination Reduction Strategy

This project reduces hallucination using:

1. Grounded retrieval
2. Evidence verification
3. Validation checks
4. Retrieval reranking
5. Confidence scoring
6. LLM-as-judge evaluation
7. Abstain/escalation logic

---

## Suggested Fine-Tuning (Updating...)

Use:
- PEFT
- QLoRA
- 4-bit quantization

to fine-tune small healthcare-oriented models with limited local compute capacity.

