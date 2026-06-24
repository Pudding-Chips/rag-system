# 极简离线 RAG 系统 🚀

这是一个不接入 LLM 的本地 RAG 系统，支持向量检索 + 语义精排。

## 快速开始
1. **安装依赖**: `pip install -r requirements.txt`
2. **运行chromadb**: `chroma run --host localhost --port 8000`
3. **初始化数据库**: `python fill.py`
4. **启动 API**: `python api.py`

## 功能特性
- **Embedding**: text2vec-base-chinese
- **Reranker**: BAAI/bge-reranker-base
- **Database**: ChromaDB