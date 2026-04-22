# =========================
# llm.py
# =========================

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"


def generate_answer(query, docs):
    context = "\n\n".join([d["content"] for d in docs])

    prompt = f"""
你是技术专家，请基于知识回答问题。如果资料里没有明确提到某项功能，请直接回答：‘根据现有资料，未提及该功能’，不要尝试根据现有资料信息进行推测。如果资料是中文，就用中文回答就好了。

1. 如果参考资料中没有提到相关信息，不要尝试根据现有资料信息进行推测，或者用户输入的是无意义的内容（如数字、乱码、闲聊），请直接回答：“抱歉，在现有文档中未找到相关信息，请尝试更具体的 CDN 问题。”
2. 不要编造资料中不存在的内容。

问题：{query}

知识：
{context}

请给出清晰、专业的回答：
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json().get("response", "")