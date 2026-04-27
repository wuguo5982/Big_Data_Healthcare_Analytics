from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from src.rag_pipeline import LocalPolicyRAG, format_citations
from src.fraud_models import score_providers, reasons_for_provider
from src.feature_engineering import add_claim_flags
from src.notes_nlp import analyze_claim_notes
from src.guardrails import enforce_fwa_language, validate_grounding
from src.prompts import CASE_REPORT_TEMPLATE

@dataclass
class AgentResponse:
    answer: str
    citations: str
    confidence: float
    grounding_status: str

class QueryRouterAgent:
    def route(self, query: str) -> str:
        q = query.lower()
        if any(x in q for x in ["provider", "npi", "top", "highest", "suspicious"]): return "provider_risk"
        if any(x in q for x in ["claim", "duplicate", "doctor note", "medical necessity"]): return "claim_review"
        if any(x in q for x in ["policy", "fraud", "waste", "abuse", "definition"]): return "policy_rag"
        return "general"

class PolicyRAGAgent:
    def __init__(self):
        self.rag = LocalPolicyRAG().load_or_build()
    def retrieve(self, query: str, k: int=5):
        chunks = self.rag.retrieve(query, k=k)
        return chunks, format_citations(chunks)

class ClaimsAnalyticsAgent:
    def __init__(self, claims: pd.DataFrame, providers: pd.DataFrame, notes: pd.DataFrame):
        self.claims = add_claim_flags(claims)
        self.providers = providers
        self.notes = notes
        self.provider_scores = score_providers(claims, providers)
    def top_providers(self, n:int=10):
        rows=[]
        for _, r in self.provider_scores.head(n).iterrows():
            risk_score = round(float(r["risk_score"]), 3)
            total_claims = int(r["total_claims"])
            total_paid = round(float(r["total_paid"]), 2)
            reasons = ", ".join(reasons_for_provider(r))
            rows.append({
                "provider_id": r["provider_id"],
                "risk_level": str(r["risk_level"]),
                "risk_score": risk_score,
                "total_claims": total_claims,
                "total_paid": total_paid,
                "reasons": reasons,
                "score": risk_score,
                "claims": total_claims,
                "paid_amount": total_paid,
            })
        return rows
    def provider_report(self, provider_id: str):
        p = self.provider_scores[self.provider_scores["provider_id"].astype(str).str.upper()==provider_id.upper()]
        if p.empty: return None
        row = p.iloc[0]
        subset = self.claims[self.claims["provider_id"]==row["provider_id"]].sort_values("paid_amount", ascending=False).head(8)
        evidence = []
        for _, c in subset.iterrows():
            flags=[]
            for col,label in [("duplicate_claim_flag","duplicate"),("dme_dx_mismatch_flag","DME mismatch"),("high_complexity_low_dx_flag","upcoding"),("high_paid_flag","high paid")]:
                if int(c.get(col,0))==1: flags.append(label)
            evidence.append(f"Claim {c['claim_id']}: code {c['cpt_hcpcs_code']}, dx {c['icd10_code']}, paid ${c['paid_amount']}, flags={flags or ['none']}")
        return row, evidence
    def claim_review(self, claim_id: str):
        c = self.claims[self.claims["claim_id"].astype(str).str.upper()==claim_id.upper()]
        if c.empty: return None
        c = c.iloc[0]
        note = analyze_claim_notes(c["claim_id"], self.notes)
        flags = [name for col,name in [("duplicate_claim_flag","duplicate claim"),("dme_dx_mismatch_flag","medical necessity mismatch"),("high_complexity_low_dx_flag","possible upcoding"),("high_paid_flag","high paid amount")] if int(c.get(col,0))==1]
        return c.to_dict(), note, flags

class ExplanationAgent:
    def write_provider_answer(self, row, evidence, citations):
        reasons = "\n".join([f"- {r}" for r in reasons_for_provider(row)])
        evidence_txt = "\n".join([f"- {e}" for e in evidence])
        ans = CASE_REPORT_TEMPLATE.format(
            risk_level=row["risk_level"], risk_score=float(row["risk_score"]), provider_id=row["provider_id"],
            reasons=reasons, evidence=evidence_txt, citations=citations
        )
        return enforce_fwa_language(ans)
    def write_claim_answer(self, claim, note, flags, citations):
        flag_txt = ", ".join(flags) if flags else "No strong claim-level rule flag; review context."
        ans = f"""
Claim Review: {claim['claim_id']}
Risk Indicators: {flag_txt}
Claim Details: provider {claim['provider_id']}, beneficiary {claim['beneficiary_id']}, code {claim['cpt_hcpcs_code']}, diagnosis {claim['icd10_code']}, paid ${claim['paid_amount']}.
Doctor Note Assessment: {note['summary']} Support score: {note['support_score']}.
Note Excerpt: {note.get('note_excerpt','')}
Citations: {citations}
Recommended Action: Review medical necessity documentation, compare against policy, and verify duplicate or high-cost billing patterns.
"""
        return enforce_fwa_language(ans)


class CMSFWAOrchestrator:
    def __init__(self, claims: pd.DataFrame, providers: pd.DataFrame, notes: pd.DataFrame):
        self.router = QueryRouterAgent()
        self.policy = PolicyRAGAgent()
        self.claims_agent = ClaimsAnalyticsAgent(claims, providers, notes)
        self.explainer = ExplanationAgent()

    def answer(self, query: str) -> AgentResponse:
        route = self.router.route(query)
        chunks, citations = self.policy.retrieve(query, k=5)

        if route == "provider_risk":
            import re
            m = re.search(r"P\d{4}", query.upper())
            if m:
                report = self.claims_agent.provider_report(m.group(0))
                if report:
                    row, evidence = report
                    ans = self.explainer.write_provider_answer(row, evidence, citations)
                else:
                    ans = f"Provider {m.group(0)} was not found in the current dataset. Citations: {citations}"
            else:
                top = self.claims_agent.top_providers(10)
                lines = ["Top high-risk providers for potential FWA review:"]
                for r in top:
                    lines.append(
                        f"- {r['provider_id']}: {r['risk_level']} risk, "
                        f"score={r['risk_score']}, claims={r['total_claims']}, "
                        f"paid=${r['total_paid']}, reasons={r['reasons']}"
                    )
                lines.append(f"Citations: {citations}")
                ans = enforce_fwa_language("\n".join(lines))

        elif route == "claim_review":
            import re
            m = re.search(r"C(?:DUP)?\d{5,7}", query.upper())
            if m:
                review = self.claims_agent.claim_review(m.group(0))
                if review:
                    claim, note, flags = review
                    ans = self.explainer.write_claim_answer(claim, note, flags, citations)
                else:
                    ans = f"Claim {m.group(0)} was not found in the current dataset. Citations: {citations}"
            else:
                ans = "Please include a claim ID such as C0000001 or CDUP00001 for claim-level review."

        else:
            context = "\n\n".join([f"[{c.source} p.{c.page}] {c.text}" for c in chunks])
            ans = (
                "Based on retrieved CMS policy/reference material, here is the grounded answer:\n\n"
                f"{context[:2200]}\n\n"
                f"Citations: {citations}"
            )
            ans = enforce_fwa_language(ans)

        ok, status = validate_grounding(ans, citations)
        confidence = 0.85 if ok else 0.55
        return AgentResponse(ans, citations, confidence, status)
