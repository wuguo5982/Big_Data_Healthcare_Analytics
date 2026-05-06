import re
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import POLICY_PATH, DIAGNOSIS_POLICY_PATH, PROCEDURE_BENCHMARK_PATH, RAG_INDEX_PATH, MODEL_DIR, CLINICAL_NOTES_PATH, AUDIT_RULES_PATH


class FWARAGEngine:
    """
    Lightweight RAG retriever using TF-IDF.

    Annotation:
    This local implementation is portable for GitHub. In production AWS, replace
    it with Amazon Bedrock Knowledge Bases, Amazon OpenSearch Serverless, or
    Aurora PostgreSQL with pgvector.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.chunks = []
        self.matrix = None

    def chunk_text(self, text):
        # Difficulty point:
        # Chunking controls retrieval quality. Bad chunks can cause weak grounding
        # and increase hallucination risk.
        return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    def build(self):
        policy_text = Path(POLICY_PATH).read_text(encoding="utf-8")

        # Integrate structured reference tables, audit rules, and synthetic
        # clinical notes into the RAG knowledge base. This supports both policy
        # grounding and document analytics demos in Streamlit.
        diagnosis_text = Path(DIAGNOSIS_POLICY_PATH).read_text(encoding="utf-8")
        procedure_text = Path(PROCEDURE_BENCHMARK_PATH).read_text(encoding="utf-8")
        audit_rules_text = Path(AUDIT_RULES_PATH).read_text(encoding="utf-8")

        notes_text = ""
        if Path(CLINICAL_NOTES_PATH).exists():
            import pandas as pd
            notes_df = pd.read_csv(CLINICAL_NOTES_PATH)
            notes_text = "\n\n".join(notes_df["note_text"].astype(str).tolist())

        combined_text = (
            policy_text
            + "\n\nPROCEDURE CODE BENCHMARK TABLE\n"
            + procedure_text
            + "\n\nDIAGNOSIS POLICY MAPPING TABLE\n"
            + diagnosis_text
            + "\n\nAUDIT RULES JSON\n"
            + audit_rules_text
            + "\n\nSYNTHETIC CLINICAL NOTES FOR DOCUMENT ANALYTICS\n"
            + notes_text
        )

        self.chunks = self.chunk_text(combined_text)
        self.matrix = self.vectorizer.fit_transform(self.chunks)
        return self

    def retrieve(self, query, top_k=4):
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).flatten()
        top_indices = scores.argsort()[::-1][:top_k]

        return [
            {"score": round(float(scores[idx]), 3), "text": self.chunks[idx]}
            for idx in top_indices
        ]

    def save(self, path=RAG_INDEX_PATH):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"vectorizer": self.vectorizer, "chunks": self.chunks, "matrix": self.matrix},
            path,
        )

    @classmethod
    def load(cls, path=RAG_INDEX_PATH):
        obj = cls()
        artifact = joblib.load(path)
        obj.vectorizer = artifact["vectorizer"]
        obj.chunks = artifact["chunks"]
        obj.matrix = artifact["matrix"]
        return obj
