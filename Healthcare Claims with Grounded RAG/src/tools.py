import json
import re
import pandas as pd
from typing import Dict, Any, List

from src.risk_model import FWARiskModel
from src.anomaly_model import FWAAnomalyModel
from src.config import (
    RISK_MODEL_PATH, ANOMALY_MODEL_PATH,
    PROVIDER_PROFILE_PATH, PROCEDURE_BENCHMARK_PATH,
    DIAGNOSIS_POLICY_PATH, AUDIT_RULES_PATH, GROUNDING_THRESHOLD,
    MIN_RETRIEVAL_SCORE, MIN_EVIDENCE_CHUNKS,
)


class ClaimValidationTool:
    """Validate the minimum claim schema before the agent runs downstream tools."""

    def run(self, claim: Dict[str, Any]):
        required = [
            "provider_id", "procedure_code", "diagnosis_code", "claim_amount",
            "patient_age", "num_prior_claims", "days_since_last_claim",
            "is_high_risk_provider",
        ]
        missing = [col for col in required if col not in claim]
        return {"valid": len(missing) == 0, "missing_fields": missing}


class ProviderProfileTool:
    """
    Looks up provider-level synthetic profile.

    Annotation:
    Provider context helps the agent reason beyond a single claim. It prevents
    the workflow from overreacting to a single row when historical provider
    information is available.
    """
    def __init__(self):
        self.df = pd.read_csv(PROVIDER_PROFILE_PATH)

    def run(self, provider_id: str):
        row = self.df[self.df["provider_id"] == provider_id]
        if row.empty:
            return {"provider_profile_found": False}
        result = row.iloc[0].to_dict()
        result["provider_profile_found"] = True
        return result


class ProcedureBenchmarkTool:
    """
    Checks procedure-code benchmark context.

    Annotation:
    Benchmark tables add deterministic context that supports RAG grounding and
    helps reduce hallucination because the agent can cite expected amount ranges
    rather than inventing clinical or billing thresholds.
    """
    def __init__(self):
        self.df = pd.read_csv(PROCEDURE_BENCHMARK_PATH)

    def run(self, procedure_code: str, claim_amount: float):
        row = self.df[self.df["procedure_code"].astype(str) == str(procedure_code)]
        if row.empty:
            return {"procedure_benchmark_found": False}
        record = row.iloc[0].to_dict()
        record["procedure_benchmark_found"] = True
        record["above_expected_max"] = bool(claim_amount > record["expected_max_amount"])
        return record


class DiagnosisPolicyTool:
    """Map diagnosis code to synthetic policy/medical-necessity context."""

    def __init__(self):
        self.df = pd.read_csv(DIAGNOSIS_POLICY_PATH)

    def run(self, diagnosis_code: str):
        row = self.df[self.df["diagnosis_code"].astype(str) == str(diagnosis_code)]
        if row.empty:
            return {"diagnosis_policy_found": False}
        record = row.iloc[0].to_dict()
        record["diagnosis_policy_found"] = True
        return record


class RuleBasedFWATool:
    def __init__(self):
        with open(AUDIT_RULES_PATH, "r", encoding="utf-8") as f:
            self.audit_rules = json.load(f)

    def run(self, claim: Dict[str, Any]) -> List[str]:
        """
        Deterministic business-rule layer.

        Annotation:
        This is important in regulated domains because it gives transparent logic.
        ML scores alone are often not enough for compliance reviewers. Deterministic
        rules also reduce hallucination because explanations can reference explicit,
        reproducible checks.
        """
        flags = []

        if claim["claim_amount"] > 2000:
            flags.append("Unusually high claim amount.")

        if claim["days_since_last_claim"] <= 7:
            flags.append("Repeated claim submitted within a short time window.")

        if claim["num_prior_claims"] >= 10:
            flags.append("High prior claim frequency.")

        if claim["is_high_risk_provider"] == 1:
            flags.append("Provider has high-risk billing history.")

        if claim["procedure_code"] in ["99214", "99215"] and claim["claim_amount"] > 1500:
            flags.append("Possible upcoding pattern involving high-complexity evaluation code.")

        return flags


class MLRiskScoringTool:
    """Load the trained supervised model and score a normalized claim feature row."""

    def __init__(self):
        self.model = FWARiskModel.load(RISK_MODEL_PATH)

    def run(self, claim: Dict[str, Any]):
        return self.model.predict(pd.DataFrame([claim]))


class AnomalyDetectionTool:
    """Load the trained unsupervised anomaly model and identify unusual claims."""

    def __init__(self):
        self.model = FWAAnomalyModel.load(ANOMALY_MODEL_PATH)

    def run(self, claim: Dict[str, Any]):
        return self.model.predict(pd.DataFrame([claim]))


class EvidenceGroundingTool:
    def run(self, explanation: str, evidence_chunks: List[Dict[str, Any]]):
        """
        Simple hallucination/evidence support check.

        Annotation:
        This checks whether important terms in the explanation are supported by
        retrieved evidence. In production, upgrade with RAGAS, Bedrock evaluation,
        groundedness metrics, or human review.
        """
        evidence_text = " ".join([x["text"].lower() for x in evidence_chunks])
        explanation_lower = explanation.lower()

        key_terms = [
            "upcoding", "duplicate billing", "high claim", "repeated claims",
            "provider history", "medically unnecessary", "fraud", "waste", "abuse",
            "medical necessity", "human review", "high-complexity"
        ]

        supported_terms = [term for term in key_terms if term in explanation_lower and term in evidence_text]
        unsupported_terms = [term for term in key_terms if term in explanation_lower and term not in evidence_text]

        citation_coverage = len(supported_terms) / max(1, len(supported_terms) + len(unsupported_terms))
        return {
            "supported_terms": supported_terms,
            "unsupported_terms": unsupported_terms,
            "citation_coverage": round(citation_coverage, 3),
            "hallucination_risk": "High" if citation_coverage < GROUNDING_THRESHOLD else "Low",
        }


class HallucinationReductionTool:
    """
    Advanced practical guardrail layer for reducing hallucination in RAG agents.

    This tool combines several production-style checks:
    1. Retrieval-quality gate: require enough retrieved chunks above a minimum score.
    2. Claim-evidence consistency: verify that key claim facts appear in the evidence,
       reference tables, rules, or structured context before the assistant states them.
    3. Unsupported-assertion detection: identify high-risk phrases that should not be
       stated unless evidence supports them.
    4. Abstention policy: when evidence is weak, force the final output to say that
       human review and stronger documentation are required instead of inventing facts.
    """

    HIGH_RISK_ASSERTIONS = [
        "intentional fraud",
        "definitely fraud",
        "criminal",
        "guilty",
        "illegal provider",
        "confirmed fraud",
        "final determination",
        "medically unnecessary",
    ]

    def _normalize(self, value: Any) -> str:
        return re.sub(r"\s+", " ", str(value).lower()).strip()

    def run(
        self,
        claim: Dict[str, Any],
        explanation: str,
        evidence_chunks: List[Dict[str, Any]],
        rule_flags: List[str],
        provider_profile: Dict[str, Any],
        procedure_context: Dict[str, Any],
        diagnosis_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        evidence_text = self._normalize(" ".join(chunk.get("text", "") for chunk in evidence_chunks))
        structured_context_text = self._normalize(
            " ".join(rule_flags)
            + " " + json.dumps(provider_profile, default=str)
            + " " + json.dumps(procedure_context, default=str)
            + " " + json.dumps(diagnosis_context, default=str)
        )
        explanation_lower = self._normalize(explanation)

        strong_chunks = [
            chunk for chunk in evidence_chunks
            if float(chunk.get("score", 0.0)) >= MIN_RETRIEVAL_SCORE
        ]
        retrieval_gate_passed = len(strong_chunks) >= MIN_EVIDENCE_CHUNKS

        required_facts = {
            "provider_id": claim.get("provider_id"),
            "procedure_code": claim.get("procedure_code"),
            "diagnosis_code": claim.get("diagnosis_code"),
        }
        fact_support = {}
        for name, value in required_facts.items():
            token = self._normalize(value)
            fact_support[name] = bool(token and (token in evidence_text or token in structured_context_text))

        unsupported_high_risk_assertions = [
            phrase for phrase in self.HIGH_RISK_ASSERTIONS
            if phrase in explanation_lower and phrase not in evidence_text and phrase not in structured_context_text
        ]

        abstain_required = (
            not retrieval_gate_passed
            or not all(fact_support.values())
            or len(unsupported_high_risk_assertions) > 0
        )

        mitigation_actions = [
            "Use extractive/template-based explanation instead of free-form generation.",
            "Show retrieved evidence chunks and similarity scores to the reviewer.",
            "Require human review when evidence coverage is weak or assertions are unsupported.",
            "Avoid final fraud accusations; describe only risk signals and review rationale.",
            "Log grounding, safety, and abstention checks for auditability.",
        ]

        return {
            "retrieval_gate_passed": retrieval_gate_passed,
            "strong_evidence_chunks": len(strong_chunks),
            "minimum_required_chunks": MIN_EVIDENCE_CHUNKS,
            "minimum_retrieval_score": MIN_RETRIEVAL_SCORE,
            "claim_fact_support": fact_support,
            "unsupported_high_risk_assertions": unsupported_high_risk_assertions,
            "abstain_required": abstain_required,
            "recommended_response_style": (
                "Abstain / require stronger evidence and human review"
                if abstain_required else "Proceed with evidence-grounded risk summary"
            ),
            "mitigation_actions": mitigation_actions,
        }


class SafetyGuardrailTool:
    def run(self, explanation: str):
        """
        Safety guardrail to prevent legally risky language.

        Annotation:
        The system supports compliance review. It should not accuse a provider or
        present a final legal/fraud determination.
        """
        banned_phrases = ["definitely fraud", "criminal", "guilty", "illegal provider", "final determination"]
        violations = [phrase for phrase in banned_phrases if phrase in explanation.lower()]

        return {
            "safe": len(violations) == 0,
            "violations": violations,
            "disclaimer": (
                "This is an AI-generated compliance analytics output. "
                "It is not a final fraud determination and requires human review."
            ),
        }
