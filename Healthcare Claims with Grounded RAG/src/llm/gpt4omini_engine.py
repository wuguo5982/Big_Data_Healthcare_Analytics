
"""
Production-ready GPT-4o-mini reasoning engine for grounded healthcare FWA analysis.

Design:
- Local RAG retrieves evidence.
- GPT-4o-mini is used only for grounded reasoning.
- If the API key is missing, a safe local fallback keeps the demo usable.
- The model is instructed to abstain when evidence is insufficient.
"""

from __future__ import annotations

import os
from typing import Optional

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


SYSTEM_PROMPT = """
You are a grounded healthcare Fraud, Waste, and Abuse (FWA) analysis assistant.

Strict rules:
1. Use ONLY the retrieved CMS/policy/audit context provided by the application.
2. Do NOT invent CMS rules, diagnosis meanings, clinical facts, or fraud conclusions.
3. Do NOT say a claim is definitely fraud. Say it may require review if evidence supports that.
4. If evidence is weak or missing, clearly state that evidence is insufficient and recommend manual review.
5. This is AI-assisted review prioritization, not a final fraud determination or claims denial decision.
6. Organize the answer using these headings:
   - Evidence Used
   - Risk Reasoning
   - Hallucination / Evidence Limitation
   - Recommended Next Step
"""


def is_openai_ready() -> bool:
    """Return True when OpenAI client and API key are available."""
    return bool(os.getenv("OPENAI_API_KEY")) and OpenAI is not None


def generate_grounded_response(
    query: str,
    context: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.1,
    max_tokens: int = 700,
    fallback_when_no_key: bool = True,
) -> str:
    """Generate a grounded response with GPT-4o-mini.

    Parameters
    ----------
    query:
        User question about healthcare FWA.
    context:
        Retrieved and reranked policy/audit evidence.
    model:
        Default is GPT-4o-mini for low-cost production-style inference.
    temperature:
        Low temperature improves consistency and reduces hallucination risk.
    max_tokens:
        Keeps response concise/readable for the Streamlit UI.
    fallback_when_no_key:
        If True, return a deterministic local fallback when API key is unavailable.
    """
    if not query.strip():
        return "Please enter a healthcare FWA question."

    if not context.strip():
        return (
            "Evidence is insufficient because no retrieved context was provided. "
            "Manual review is recommended before making any FWA interpretation."
        )

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key or OpenAI is None:
        if fallback_when_no_key:
            return (
                "Evidence Used\n"
                f"- Retrieved context was available, but GPT-4o-mini was not called because OPENAI_API_KEY is missing.\n\n"
                "Risk Reasoning\n"
                "- The retrieved context should be reviewed for abnormal billing, diagnosis-procedure inconsistency, duplicate billing, excessive utilization, or provider outlier behavior.\n\n"
                "Hallucination / Evidence Limitation\n"
                "- This is a local fallback response and does not use live GPT-4o-mini reasoning.\n"
                "- Add OPENAI_API_KEY to enable the production LLM workflow.\n\n"
                "Recommended Next Step\n"
                "- Treat this as review prioritization only and escalate suspicious cases for manual audit review."
            )
        raise RuntimeError("OPENAI_API_KEY is missing or the openai package is unavailable.")

    client = OpenAI(api_key=api_key)

    prompt = f"""
Healthcare FWA user question:
{query}

Retrieved and reranked evidence:
{context}

Generate a concise, grounded answer. Use only the evidence above.
"""

    try:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or "No response returned from GPT-4o-mini."
    except Exception as exc:
        return (
            "GPT-4o-mini API call failed.\n\n"
            f"Error: {exc}\n\n"
            "Fallback recommendation: verify OPENAI_API_KEY, network access, and OpenAI package version."
        )
