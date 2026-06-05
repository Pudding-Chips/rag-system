import os
DB_HOST = "localhost"
DB_PORT = 8000
RAG_API_URL = "http://localhost:8001"

DEFAULT_MODEL = "text2vec"

EMBEDDING_MODELS = {
    "text2vec": "shibing624/text2vec-base-chinese",
}

COLLECTION_TEMPLATE = "{biz}_v1"
DEFAULT_BIZ = "general"

RERANK_THRESHOLD = 0.50