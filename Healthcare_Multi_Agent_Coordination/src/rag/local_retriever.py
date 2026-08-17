from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class LocalEvidenceRetriever:
    """Small local RAG retriever for a portfolio demo.

    Production replacement:
      - Amazon Bedrock Knowledge Bases
      - enterprise vector store
      - metadata filtering + reranking
    """

    def __init__(self, guidelines_df):
        self.guidelines = guidelines_df.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(self.guidelines["text"])

    def retrieve(self, query: str, k: int = 3):
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix).ravel()
        idx = scores.argsort()[::-1][:k]
        return [
            {
                "guideline_id": self.guidelines.iloc[i]["guideline_id"],
                "topic": self.guidelines.iloc[i]["topic"],
                "text": self.guidelines.iloc[i]["text"],
                "score": round(float(scores[i]), 4),
            }
            for i in idx
        ]
