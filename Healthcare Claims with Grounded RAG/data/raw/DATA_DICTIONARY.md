# Data Dictionary

Synthetic data for portfolio/demo only. No PHI.

## claims.csv
1,000 synthetic claim rows with `claim_id`, provider/patient IDs, procedure and diagnosis codes, claim amount, patient age, utilization features, high-risk provider indicator, and binary `label`.

## clinical_notes.csv
1,000 synthetic unstructured notes linked to claim IDs for document analytics and RAG retrieval demos.

## cms_fwa_policy.txt
CMS-style synthetic policy knowledge base with FWA review principles, upcoding, duplicate billing, DME, ambulance, unclassified drug, human-review escalation, and grounding guidance.

## provider_profiles.csv
80 synthetic providers with specialty, region, utilization, peer percentile, and synthetic provider risk score.

## procedure_code_benchmarks.csv
Procedure descriptions and expected synthetic min/max amounts used for deterministic validation.

## diagnosis_policy_mapping.csv
Diagnosis descriptions and review notes used for diagnosis-procedure consistency checks.

## audit_rules.json
Transparent deterministic rules used by the agent before ML/RAG reasoning.
