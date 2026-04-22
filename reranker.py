from sentence_transformers import CrossEncoder

reranker_model = CrossEncoder("BAAI/bge-reranker-base")

def rerank(query, docs, top_k=3):
    if not docs:
        return []

    pairs = [[query, d.page_content] for d in docs]
    scores = reranker_model.predict(pairs)

    ranked_results = []
    for d, score in zip(docs, scores):
        d.metadata["relevance_score"] = float(score) 
        ranked_results.append(d)

    ranked_results.sort(key=lambda x: x.metadata["relevance_score"], reverse=True)

    return ranked_results[:top_k]
