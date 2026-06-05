"""
将初筛的文档用 bge-reranker-base 重新打分，
并用 Sigmoid 将原始 Logits 映射到 [0, 1] 之间，按降序排列
"""
import numpy as np
from sentence_transformers import CrossEncoder

reranker_model = CrossEncoder("BAAI/bge-reranker-base")

def sigmoid(x):
    return 1 / (1 + np.exp(-(x)))

def rerank(query, docs, top_k=None):
    if not docs:
        return []

    pairs = [[query, d.page_content] for d in docs]
    raw_scores = reranker_model.predict(pairs)

    if isinstance(raw_scores, (int, float)):
        raw_scores = [raw_scores]

    ranked_results = []
    for d, raw_scores in zip(docs, raw_scores):
        normalized_score = float(sigmoid(raw_scores))

        d.metadata["relevance_score"] = normalized_score
        ranked_results.append(d)

    ranked_results.sort(key=lambda x: x.metadata["relevance_score"], reverse=True)

    if top_k is not None:
        return ranked_results[:top_k]
    
    return ranked_results[:top_k]
