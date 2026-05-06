"""Agentic workflow for healthcare claims FWA detection.

This module is intentionally verbose and highly annotated because it is designed
for a portfolio/interview project. The goal is to show how a production-style
FWA assistant can combine deterministic checks, ML models, anomaly detection,
RAG evidence retrieval, safety checks, and human-review escalation.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.tools import (
    AnomalyDetectionTool,
    ClaimValidationTool,
    DiagnosisPolicyTool,
    EvidenceGroundingTool,
    HallucinationReductionTool,
    MLRiskScoringTool,
    ProcedureBenchmarkTool,
    ProviderProfileTool,
    RuleBasedFWATool,
    SafetyGuardrailTool,
)


class AgenticFWAWorkflow:
    """
    Integrated agentic workflow for healthcare FWA detection.

    Important distinction:
    This is not a simple chatbot. It is an agentic decision-support workflow:

        validation
        -> reference lookup
        -> deterministic rules
        -> supervised ML risk scoring
        -> unsupervised anomaly detection
        -> RAG policy retrieval
        -> evidence-grounded explanation
        -> grounding evaluation
        -> safety guardrail
        -> human-review decision

    Design principle:
    The agent never makes a final fraud accusation. It produces a risk analysis
    and recommends human compliance review when risk, anomaly, grounding, or
    safety signals indicate uncertainty or potential concern.
    """

    def __init__(self, rag_engine):
        self.rag = rag_engine
        self.validator = ClaimValidationTool()
        self.provider_tool = ProviderProfileTool()
        self.procedure_tool = ProcedureBenchmarkTool()
        self.diagnosis_tool = DiagnosisPolicyTool()
        self.rule_tool = RuleBasedFWATool()
        self.risk_tool = MLRiskScoringTool()
        self.anomaly_tool = AnomalyDetectionTool()
        self.grounding_tool = EvidenceGroundingTool()
        self.hallucination_tool = HallucinationReductionTool()
        self.safety_tool = SafetyGuardrailTool()

    def plan(self) -> List[str]:
        """Return the high-level agent plan shown in the UI/API output."""
        return [
            "Validate claim fields and stop early if required information is missing.",
            "Look up provider profile, procedure benchmark, and diagnosis policy context.",
            "Run deterministic FWA audit rules for transparent compliance signals.",
            "Run supervised ML fraud risk scoring for probability-based prioritization.",
            "Run unsupervised anomaly detection to catch unusual billing patterns.",
            "Retrieve CMS-style policy evidence using RAG to ground the explanation.",
            "Generate a structured, evidence-grounded explanation for reviewers.",
            "Evaluate citation coverage and hallucination risk.",
            "Apply retrieval-quality gates, claim-fact support checks, and abstention rules to reduce hallucination.",
            "Apply safety guardrails to avoid final accusations or legally risky language.",
            "Escalate risky, anomalous, weakly grounded, or unsafe cases to human review.",
        ]

    @staticmethod
    def _trace(step: int, name: str, purpose: str, method: str, output: Any) -> Dict[str, Any]:
        """Build a standardized trace object for Streamlit and API consumers."""
        return {
            "step": step,
            "name": name,
            "purpose": purpose,
            "method": method,
            "output": output,
        }

    def build_policy_query(
        self,
        claim: Dict[str, Any],
        rule_flags: List[str],
        risk_result: Dict[str, Any],
        anomaly_result: Dict[str, Any],
        provider_profile: Dict[str, Any],
        procedure_context: Dict[str, Any],
        diagnosis_context: Dict[str, Any],
    ) -> str:
        """
        Build a rich RAG query from claim facts and model/tool outputs.

        Why this matters:
        RAG quality depends heavily on query construction. A plain query such as
        "is this fraud?" is weak. This query includes procedure, diagnosis,
        provider, benchmark, rule flags, risk score, and anomaly signal so the
        retriever can find more relevant policy evidence.
        """
        return (
            f"CMS fraud waste abuse policy for claim amount {claim['claim_amount']}, "
            f"procedure code {claim['procedure_code']}, diagnosis {claim['diagnosis_code']}, "
            f"provider specialty {provider_profile.get('provider_specialty')}, "
            f"procedure risk note {procedure_context.get('risk_note')}, "
            f"diagnosis review note {diagnosis_context.get('review_note')}, "
            f"upcoding duplicate billing repeated claims high-risk provider "
            f"rule flags {rule_flags}, ML risk {risk_result}, anomaly {anomaly_result}"
        )

    def generate_explanation(
        self,
        claim: Dict[str, Any],
        rule_flags: List[str],
        risk_result: Dict[str, Any],
        anomaly_result: Dict[str, Any],
        provider_profile: Dict[str, Any],
        procedure_context: Dict[str, Any],
        diagnosis_context: Dict[str, Any],
        evidence: List[Dict[str, Any]],
    ) -> str:
        """
        Generate a reviewer-friendly explanation.

        Annotation:
        In a production system, this section could be generated by a foundation
        model using a strict prompt template. For this portfolio project, it is
        template-based to reduce hallucination risk and keep the output fully
        reproducible during interviews and GitHub demos.
        """
        evidence_summary = "\n".join([f"- {item['text']}" for item in evidence])
        rule_summary = "\n".join([f"- {flag}" for flag in rule_flags]) if rule_flags else "- No major rule-based flags detected."

        explanation = f"""
FWA Risk Assessment Summary

Claim Information:
- Provider ID: {claim['provider_id']}
- Procedure Code: {claim['procedure_code']}
- Diagnosis Code: {claim['diagnosis_code']}
- Claim Amount: ${claim['claim_amount']}
- Prior Claims: {claim['num_prior_claims']}
- Days Since Last Claim: {claim['days_since_last_claim']}
- High-Risk Provider Indicator: {claim['is_high_risk_provider']}

Provider Profile Context:
- Specialty: {provider_profile.get('provider_specialty', 'Unknown')}
- Provider Synthetic Risk Score: {provider_profile.get('synthetic_provider_risk_score', 'Unknown')}
- Provider High-Complexity Rate: {provider_profile.get('high_complexity_rate', 'Unknown')}

Procedure Benchmark Context:
- Description: {procedure_context.get('description', 'Unknown')}
- Expected Max Amount: {procedure_context.get('expected_max_amount', 'Unknown')}
- Above Expected Max: {procedure_context.get('above_expected_max', 'Unknown')}
- Risk Note: {procedure_context.get('risk_note', 'Unknown')}

Diagnosis Policy Context:
- Diagnosis Description: {diagnosis_context.get('diagnosis_description', 'Unknown')}
- Review Note: {diagnosis_context.get('review_note', 'Unknown')}

Rule-Based Findings:
{rule_summary}

ML Risk Scoring:
- Risk Level: {risk_result['risk_level']}
- Risk Score: {risk_result['risk_score']}

Anomaly Detection:
- Is Anomaly: {anomaly_result['is_anomaly']}
- Anomaly Score: {anomaly_result['anomaly_score']}

Retrieved CMS-Style Policy Evidence:
{evidence_summary}

Reasoning:
This claim may require compliance review because the system detected a combination of structured risk factors, rule-based FWA indicators, provider-level context, procedure benchmark context, anomaly signals, and policy-grounded evidence. The analysis focuses on possible FWA patterns such as high claim amount, repeated claims, provider billing risk, medical necessity concerns, duplicate billing, and possible upcoding. This output should be reviewed by a human compliance specialist before any action is taken.
"""
        return explanation.strip()

    def reflect(
        self,
        risk_result: Dict[str, Any],
        anomaly_result: Dict[str, Any],
        grounding_result: Dict[str, Any],
        hallucination_result: Dict[str, Any],
        safety_result: Dict[str, Any],
        procedure_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Summarize the agent's self-check before final escalation decision."""
        issues: List[str] = []

        if risk_result["risk_level"] == "High":
            issues.append("High ML fraud risk score.")

        if anomaly_result["is_anomaly"]:
            issues.append("Claim is anomalous compared with learned billing patterns.")

        if procedure_context.get("above_expected_max") is True:
            issues.append("Claim amount is above the procedure benchmark expected maximum.")

        if grounding_result["hallucination_risk"] == "High":
            issues.append("Low evidence grounding; explanation may require stronger citations.")

        if hallucination_result.get("abstain_required"):
            issues.append("Advanced hallucination checks recommend abstention or stronger evidence before action.")

        if not safety_result["safe"]:
            issues.append("Safety guardrail violation detected.")

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "interpretation": (
                "No major escalation signal detected."
                if len(issues) == 0
                else "One or more risk/quality/safety signals support human review."
            ),
        }

    def run(self, claim: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the full FWA analysis workflow for one claim."""
        agent_plan = self.plan()
        trace: List[Dict[str, Any]] = []

        validation = self.validator.run(claim)
        trace.append(
            self._trace(
                1,
                "Claim validation",
                "Ensure required fields exist before running models or rules.",
                "Schema-style validation using required claim columns.",
                validation,
            )
        )
        if not validation["valid"]:
            return {
                "error": "Invalid claim input.",
                "missing_fields": validation["missing_fields"],
                "agent_plan": agent_plan,
                "agent_trace": trace,
            }

        # Exclude non-feature identifiers and labels before model inference.
        model_claim = {key: value for key, value in claim.items() if key not in ["claim_id", "patient_id", "label"]}

        provider_profile = self.provider_tool.run(model_claim["provider_id"])
        procedure_context = self.procedure_tool.run(model_claim["procedure_code"], model_claim["claim_amount"])
        diagnosis_context = self.diagnosis_tool.run(model_claim["diagnosis_code"])
        trace.append(
            self._trace(
                2,
                "Reference context lookup",
                "Add business context beyond the raw claim row.",
                "Lookup provider profile, procedure benchmark, and diagnosis policy mapping tables.",
                {
                    "provider_profile": provider_profile,
                    "procedure_context": procedure_context,
                    "diagnosis_context": diagnosis_context,
                },
            )
        )

        rule_flags = self.rule_tool.run(model_claim)
        trace.append(
            self._trace(
                3,
                "Deterministic FWA rules",
                "Produce transparent, explainable compliance flags.",
                "Apply threshold and pattern rules such as high amount, repeated claim, and possible upcoding.",
                {"rule_flags": rule_flags},
            )
        )

        risk_result = self.risk_tool.run(model_claim)
        anomaly_result = self.anomaly_tool.run(model_claim)
        trace.append(
            self._trace(
                4,
                "ML and anomaly scoring",
                "Prioritize claims by learned risk and unusual billing behavior.",
                "Run XGBoost supervised classifier and Isolation Forest anomaly detector.",
                {"ml_risk": risk_result, "anomaly_detection": anomaly_result},
            )
        )

        query = self.build_policy_query(
            model_claim,
            rule_flags,
            risk_result,
            anomaly_result,
            provider_profile,
            procedure_context,
            diagnosis_context,
        )
        evidence = self.rag.retrieve(query, top_k=4)
        trace.append(
            self._trace(
                5,
                "RAG evidence retrieval",
                "Ground the analysis in CMS-style policy and reference evidence.",
                "Build a claim-aware retrieval query and fetch top policy/reference chunks from the TF-IDF RAG index.",
                {"retrieval_query": query, "retrieved_evidence": evidence},
            )
        )

        explanation = self.generate_explanation(
            model_claim,
            rule_flags,
            risk_result,
            anomaly_result,
            provider_profile,
            procedure_context,
            diagnosis_context,
            evidence,
        )

        grounding_result = self.grounding_tool.run(explanation, evidence)
        hallucination_result = self.hallucination_tool.run(
            model_claim,
            explanation,
            evidence,
            rule_flags,
            provider_profile,
            procedure_context,
            diagnosis_context,
        )
        safety_result = self.safety_tool.run(explanation)
        reflection = self.reflect(risk_result, anomaly_result, grounding_result, hallucination_result, safety_result, procedure_context)
        trace.append(
            self._trace(
                6,
                "Grounding, hallucination reduction, safety, and reflection",
                "Check whether the final explanation is supported, evidence-gated, and safe for compliance use.",
                "Evaluate citation coverage, retrieval quality, claim-fact support, abstention triggers, banned phrases, and escalation signals before returning the final output.",
                {
                    "grounding_evaluation": grounding_result,
                    "hallucination_reduction": hallucination_result,
                    "safety_evaluation": safety_result,
                    "reflection": reflection,
                },
            )
        )

        human_review_required = (
            risk_result["risk_level"] in ["Medium", "High"]
            or anomaly_result["is_anomaly"]
            or grounding_result["hallucination_risk"] == "High"
            or hallucination_result.get("abstain_required")
            or not safety_result["safe"]
            or procedure_context.get("above_expected_max") is True
        )
        trace.append(
            self._trace(
                7,
                "Human-review escalation decision",
                "Avoid automated final decisions in a regulated healthcare workflow.",
                "Escalate if risk, anomaly, grounding, safety, or benchmark signals require reviewer attention.",
                {"human_review_required": human_review_required},
            )
        )

        return {
            "claim_id": claim.get("claim_id"),
            "agent_plan": agent_plan,
            "agent_trace": trace,
            "provider_profile": provider_profile,
            "procedure_context": procedure_context,
            "diagnosis_context": diagnosis_context,
            "rule_flags": rule_flags,
            "ml_risk": risk_result,
            "anomaly_detection": anomaly_result,
            "retrieval_query": query,
            "retrieved_evidence": evidence,
            "explanation": explanation,
            "grounding_evaluation": grounding_result,
            "hallucination_reduction": hallucination_result,
            "safety_evaluation": safety_result,
            "reflection": reflection,
            "human_review_required": human_review_required,
            "disclaimer": safety_result["disclaimer"],
        }
