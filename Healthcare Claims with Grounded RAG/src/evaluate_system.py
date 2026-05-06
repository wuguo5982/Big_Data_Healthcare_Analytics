import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.config import CLAIMS_PATH, GOLDEN_EVAL_PATH
from src.rag_engine import FWARAGEngine
from src.agentic_fwa_workflow import AgenticFWAWorkflow


def evaluate_agentic_system(use_golden_set: bool = False):
    """
    Evaluate predictive and agentic/RAG behavior.

    Annotation:
    For LLM/RAG systems, evaluation should include ML metrics plus grounding,
    hallucination risk, and human-review escalation rate.
    """
    df = pd.read_csv(GOLDEN_EVAL_PATH if use_golden_set else CLAIMS_PATH)

    rag = FWARAGEngine.load()
    agent = AgenticFWAWorkflow(rag)

    y_true, y_pred = [], []
    hallucination_risks, citation_coverages, human_review_flags = [], [], []

    for _, row in df.iterrows():
        claim = row.to_dict()
        result = agent.run(claim)
        predicted_risk = result["ml_risk"]["risk_level"] in ["Medium", "High"]

        if "label" in row:
            y_true.append(int(row["label"]))
            y_pred.append(int(predicted_risk))

        hallucination_risks.append(result["grounding_evaluation"]["hallucination_risk"])
        citation_coverages.append(result["grounding_evaluation"]["citation_coverage"])
        human_review_flags.append(result["human_review_required"])

    metrics = {
        "classification_accuracy": accuracy_score(y_true, y_pred) if y_true else None,
        "classification_precision": precision_score(y_true, y_pred, zero_division=0) if y_true else None,
        "classification_recall": recall_score(y_true, y_pred, zero_division=0) if y_true else None,
        "classification_f1": f1_score(y_true, y_pred, zero_division=0) if y_true else None,
        "avg_citation_coverage": sum(citation_coverages) / len(citation_coverages),
        "hallucination_high_risk_rate": hallucination_risks.count("High") / len(hallucination_risks),
        "human_review_rate": sum(human_review_flags) / len(human_review_flags),
    }

    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    evaluate_agentic_system(use_golden_set=False)
