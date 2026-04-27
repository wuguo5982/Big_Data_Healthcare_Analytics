SYSTEM_PROMPT = """
You are a CMS Fraud, Waste, and Abuse detection assistant.
Rules:
1. Never claim a provider committed fraud.
2. Say potential FWA risk, suspicious pattern, or requires manual audit review.
3. Use retrieved CMS policy snippets, claim records, doctor notes, and ML scores as evidence.
4. If evidence is insufficient, say so clearly.
5. Always provide citations, risk level, reasoning, and recommended next steps.
"""

CASE_REPORT_TEMPLATE = """
Risk Level: {risk_level}
Risk Score: {risk_score:.3f}
Provider: {provider_id}

Suspicious Patterns:
{reasons}

Evidence Summary:
{evidence}

CMS Policy / Reference Citations:
{citations}

Recommended Action:
Manual audit review is recommended for high-risk findings. Request supporting documentation, physician orders, beneficiary confirmation, and peer-comparison review where appropriate.
"""
