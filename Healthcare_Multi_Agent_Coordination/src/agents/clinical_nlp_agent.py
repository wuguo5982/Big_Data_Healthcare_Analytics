# Clinical NLP Agent handles extraction of clinical conditions and flags from unstructured text notes. 
# It uses a combination of rule-based matching and fuzzy string matching to identify relevant information.

from rapidfuzz import fuzz

class ClinicalNLPAgent:
    name = "clinical_nlp_agent"

    CONDITION_LEXICON = {
        "heart failure": ["heart failure", "congestive heart failure", "CHF"],
        "COPD": ["COPD", "chronic obstructive pulmonary disease", "copd"],
        "CKD": ["CKD", "chronic kidney disease", "kidney dz", "renal impairment"],
        "diabetes": ["diabetes", "type 2 diabetes", "DM2"],
    }

    def _fuzzy_present(self, text: str, variants: list[str], threshold: int = 78) -> bool:
        text_lower = text.lower()
        if any(v.lower() in text_lower for v in variants):
            return True

        # Fuzzy-match short text windows against expected clinical terms.
        words = text_lower.replace("/", " ").replace(";", " ").split()
        for width in (2, 3, 4, 5):
            for i in range(max(1, len(words) - width + 1)):
                window = " ".join(words[i:i+width])
                if any(fuzz.partial_ratio(window, v.lower()) >= threshold for v in variants):
                    return True
        return False

    def run(self, rows: dict) -> dict:
        text = " ".join(n["note_text"] for n in rows["notes"])
        entities = [
            condition
            for condition, variants in self.CONDITION_LEXICON.items()
            if self._fuzzy_present(text, variants)
        ]

        flags = []
        if "shortness of breath" in text.lower() or " sob " in f" {text.lower()} ":
            flags.append("shortness of breath")
        if "multiple recent ed visits" in text.lower():
            flags.append("high acute-care utilization")
        if "several recent hospital admissions" in text.lower():
            flags.append("recurrent admissions")

        return {
            "detected_conditions": entities,
            "clinical_flags": flags,
            "note_excerpt": text[:450],
            "method": "rule + fuzzy matching demo; production can add clinical transformer/LLM extraction",
        }
