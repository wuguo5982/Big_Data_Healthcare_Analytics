from typing import TypedDict, Any, Dict, List


class WorkflowState(TypedDict, total=False):
    """Shared state populated incrementally by the multi-agent workflow.

    total=False makes all keys optional at the typing level. This avoids
    typing.NotRequired and is compatible with Python 3.10 environments.
    """

    patient_id: str
    query: str
    plan: List[str]

    patient_rows: Dict[str, Any]
    patient_summary: Dict[str, Any]
    clinical_summary: Dict[str, Any]
    claims_summary: Dict[str, Any]
    risk_summary: Dict[str, Any]
    evidence_summary: Dict[str, Any]

    report_data: Dict[str, Any]
    safety: Dict[str, Any]
    final_report: str
