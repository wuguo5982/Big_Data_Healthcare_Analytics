from src.rag_pipeline import LocalPolicyRAG
if __name__ == "__main__":
    rag = LocalPolicyRAG().build()
    print(f"Built RAG index with {len(rag.chunks)} chunks.")
