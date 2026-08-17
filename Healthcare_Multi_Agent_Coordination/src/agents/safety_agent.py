# safety_agent.py
# Safety Agent checks generated reports for unsafe medical language, missing evidence references, and decision-support disclaimers. 
# It flags issues and can repair reports to include necessary disclaimers and evidence references.    

import re

class SafetyAgent:
    name = "safety_agent"

    UNSAFE_PATTERNS = [
        r"\bi diagnose\b",
        r"\byou have\b.*\bdisease\b",
        r"\bstart taking\b",
        r"\bstop taking\b",
        r"\bi prescribe\b",
    ]

    def run(self, report: str, evidence_summary: dict) -> dict:
        issues = []
        lower = report.lower()

        for pattern in self.UNSAFE_PATTERNS:
            if re.search(pattern, lower):
                issues.append(f"unsafe medical-language pattern: {pattern}")

        evidence_ids = [
            item["guideline_id"]
            for item in evidence_summary.get("evidence", [])
            if "guideline_id" in item
        ]
        if evidence_ids and not any(eid in report for eid in evidence_ids):
            issues.append("report does not reference retrieved evidence IDs")

        if "decision support" not in lower:
            issues.append("missing decision-support limitation")

        return {
            "passed": len(issues) == 0,
            "issues": issues,
        }

    def repair(self, report: str, evidence_summary: dict) -> str:
        ids = ", ".join(
            item.get("guideline_id", "")
            for item in evidence_summary.get("evidence", [])
        )
        return (
            report.strip()
            + f" Evidence references: {ids}. "
            + "This output is decision support only and is not a diagnosis or prescription."
        )
