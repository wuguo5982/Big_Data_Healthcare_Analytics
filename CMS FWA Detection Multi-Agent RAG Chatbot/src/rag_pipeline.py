from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import joblib
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .config import POLICY_DIR, VECTOR_DIR

@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int
    score: float

class LocalPolicyRAG:
    """Lightweight local RAG over CMS policy PDFs using TF-IDF.
    This is intentionally dependency-light and runs without an API key.
    You can replace it with LangChain + FAISS/Pinecone/OpenSearch in production.
    """
    def __init__(self, policy_dir: Path = POLICY_DIR, vector_dir: Path = VECTOR_DIR):
        self.policy_dir = Path(policy_dir)
        self.vector_dir = Path(vector_dir)
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=20000)
        self.chunks = []
        self.matrix = None

    def _read_pdf(self, path: Path):
        reader = PdfReader(str(path))
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            yield i, text

    def _chunk(self, text: str, size: int = 900, overlap: int = 120):
        text = re.sub(r"\s+", " ", text).strip()
        if not text: return []
        chunks=[]; start=0
        while start < len(text):
            chunks.append(text[start:start+size])
            start += size - overlap
        return chunks

    def build(self):
        self.chunks=[]
        for pdf in sorted(self.policy_dir.glob("*.pdf")):
            for page, text in self._read_pdf(pdf):
                for chunk in self._chunk(text):
                    self.chunks.append({"text": chunk, "source": pdf.name, "page": page})
        if not self.chunks:
            raise FileNotFoundError(f"No PDFs found in {self.policy_dir}")
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])
        joblib.dump({"vectorizer": self.vectorizer, "chunks": self.chunks, "matrix": self.matrix}, self.vector_dir / "policy_tfidf.joblib")
        return self

    def load_or_build(self):
        path = self.vector_dir / "policy_tfidf.joblib"
        if path.exists():
            obj = joblib.load(path)
            self.vectorizer = obj["vectorizer"]; self.chunks = obj["chunks"]; self.matrix = obj["matrix"]
            return self
        return self.build()

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        if self.matrix is None: self.load_or_build()
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.matrix).ravel()
        idx = sims.argsort()[::-1][:k]
        return [RetrievedChunk(self.chunks[i]["text"], self.chunks[i]["source"], self.chunks[i]["page"], float(sims[i])) for i in idx if sims[i] > 0]

def format_citations(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No policy citations retrieved."
    seen=[]
    for c in chunks:
        label=f"{c.source}, page {c.page}"
        if label not in seen: seen.append(label)
    return "; ".join(seen[:5])
