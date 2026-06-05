"""
复杂文本转化为Dense Vectors
"""
import threading
import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODELS

_model_cache = {}
_model_lock = threading.Lock()

def get_embedding_model(model_name: str) -> SentenceTransformer:
    global _model_cache

    if model_name not in _model_cache:
        with _model_lock:
                if model_name not in _model_cache:
                    if model_name not in EMBEDDING_MODELS:
                        raise ValueError(f"Model Configuration not found: {model_name}")
                    print(f"Loading text vectorization model: {model_name} | {EMBEDDING_MODELS[model_name]}")
                    _model_cache[model_name] = SentenceTransformer(EMBEDDING_MODELS[model_name])
    return _model_cache[model_name]

def embed_texts(texts, model_name: str):
    model = get_embedding_model(model_name)
    embeddings = model.encode(texts)
    if hasattr(embeddings, "tolist"):
        result = embeddings.tolist()
    elif isinstance(embeddings, np.ndarray):
        result = embeddings.tolist()
    else:
        result = list(embeddings)

    return result
