# Document Analytics, RAG, Evaluation, and Hallucination Reduction

## Document analytics

The Streamlit app summarizes the RAG corpus with:

- source inventory
- approximate token counts
- unique terms
- top terms
- clinical-note samples
- structured claim summaries

## RAG resources

The RAG index includes:

- CMS-style FWA policy text
- procedure benchmark table
- diagnosis-policy mapping
- audit rules JSON
- 1,000 synthetic clinical notes

## Retrieval evaluation

Representative queries are tested for:

- top retrieval score
- average top-4 retrieval score
- number of returned evidence chunks
- retrieval gate pass/fail

## Validation

The validation report checks:

- required claim schema
- claim row count
- clinical-note row count
- one note per claim
- orphan notes
- unknown procedure codes
- unknown diagnosis codes
- null counts
- positive claim amounts

## Hallucination reduction

The agent uses:

1. retrieval-quality gates
2. structured context grounding
3. unsupported assertion detection
4. abstention when evidence is weak
5. safe reviewer language
6. human-review escalation

In production, extend this with:

- RAGAS faithfulness and answer relevance
- LLM-as-judge using a golden evaluation set
- source citation enforcement
- model invocation logging
- model/prompt versioning
- CI/CD quality gates before deployment
