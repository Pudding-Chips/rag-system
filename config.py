# =========================
# config.py（统一配置）
# =========================

DB_HOST = "localhost"
DB_PORT = 8000

DEFAULT_MODEL = "text2vec"

EMBEDDING_MODELS = {
    "text2vec": "shibing624/text2vec-base-chinese",
    # 以后可以加：
    # "bge": "BAAI/bge-base-zh",
}

COLLECTION_TEMPLATE = "{biz}_{model}_v1"

DEFAULT_BIZ = "cdn"