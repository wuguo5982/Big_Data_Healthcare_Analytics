# Blood Cancer Research Assistant — CrewAI

A multi-agent **educational research prototype** project. It is designed for literature review and structured summaries of leukemia, lymphoma, myeloma, and related blood cancers.

## Architecture

**Sequential workflow (8 agents / 8 tasks):** PubMed publications → clinical guidelines → symptoms → diagnosis → laboratory tests → treatment evidence → independent evidence audit → final report.

The code retains the source project's `crew.py`, `main.py`, `config/agents.yaml`, `config/tasks.yaml`, and `tools/` conventions. Research agents have bounded iterations and execution times; the crew uses `memory=False` to avoid stale embedding conflicts. A shared global search-call budget is **not yet implemented**.

## Setup (Windows CMD)

```cmd
cd blood_cancer_crewai
copy .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY`. `SERPER_API_KEY` is optional for general web searches; PubMed searches work without it. Do **not** commit `.env`.

```cmd
set SSL_CERT_FILE=
set SSL_CERT_DIR=
set UV_SYSTEM_CERTS=true
uv lock --refresh --system-certs
uv sync --system-certs
uv run --no-sync blood_cancer --topic "acute myeloid leukemia"
```

Alternative topic: `--topic "multiple myeloma" --question "Review diagnostic criteria and treatment evidence"`.

## Output

`output/blood_cancer_report.md` — source-linked research synthesis with evidence audit and limitations.

## Sources and limitations

- PubMed is queried using NCBI E-utilities (up to five papers per tool call). Abstracts are not full-text evidence, and publication retrieval does not establish current standard of care.
- General web research requires a configured Serper key. A guideline not retrieved live must be labeled **unverified**.
- This prototype has **not** been end-to-end executed against live services here. Source quality, tool budgets, citation accuracy, and medical correctness need validation before use.
- **No patient identifiers or protected health information (PHI)** should be entered. No EHR integration, clinical-grade privacy controls, or HIPAA compliance is provided.
- **Not a medical device or clinical decision system.** It cannot diagnose cancer, interpret personal lab results reliably, recommend individualized treatment, or replace a hematologist/oncologist. Severe symptoms warrant prompt professional evaluation.

## Acknowledgment

Architecture inspired by the user's Stock Picker CrewAI example, based on educational materials by Ed Donner.
