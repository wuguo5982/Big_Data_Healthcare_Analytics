def rerank_context(chunks):
    return sorted(chunks, key=lambda x: len(x), reverse=True)
