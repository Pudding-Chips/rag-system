from sentence_transformers import CrossEncoder

# 保持模型加载不变
reranker_model = CrossEncoder("BAAI/bge-reranker-base")

def rerank(query, docs, top_k=3):
    if not docs:
        return []

    # 1. 适配 Document 对象，提取文本进行打分
    # 注意：这里改用 d.page_content
    pairs = [[query, d.page_content] for d in docs]
    scores = reranker_model.predict(pairs)

    # 2. 将分数和对象合并
    ranked_results = []
    for d, score in zip(docs, scores):
        # 核心修改：将分数存入 metadata，方便 api.py 调用
        d.metadata["relevance_score"] = float(score) 
        ranked_results.append(d)

    # 3. 按分数从高到低排序
    ranked_results.sort(key=lambda x: x.metadata["relevance_score"], reverse=True)

    # 4. 返回前 top_k 个结果
    return ranked_results[:top_k]