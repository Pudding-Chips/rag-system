DB_HOST = "localhost"
DB_PORT = 8000

DEFAULT_MODEL = "text2vec"

EMBEDDING_MODELS = {
    "text2vec": "shibing624/text2vec-base-chinese",
}

COLLECTION_TEMPLATE = "{biz}_{model}_v1"

DEFAULT_BIZ = "cdn"
