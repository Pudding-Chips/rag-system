# =========================
# embedding.py（统一embedding）
# =========================

from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODELS

_model_cache = {}


def get_embedding_model(model_name):
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(
            EMBEDDING_MODELS[model_name]
        )
    return _model_cache[model_name]


def embed_texts(texts, model_name):
    model = get_embedding_model(model_name)
    return model.encode(texts)