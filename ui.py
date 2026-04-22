import streamlit as st
import requests
import json

# 设置页面基本信息
st.set_page_config(page_title="腾讯云 CDN 智能助手", layout="wide")

st.title("🚀 CDN 知识库 RAG 助手")
st.markdown("---")

# 侧边栏配置
with st.sidebar:
    st.header("系统设置")
    api_url = st.text_input("API 接口地址", value="http://localhost:8001/rag/query")
    biz_type = st.selectbox("业务场景", ["cdn"])
    model_type = st.selectbox("嵌入模型", ["text2vec"])

# 聊天记录初始化
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史聊天
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 用户输入
if prompt := st.chat_input("问问关于 CDN 的问题，比如：支持 POST 吗？"):
    # 1. 显示用户提问
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 调用你写的 api.py 接口
    with st.chat_message("assistant"):
        with st.spinner("正在检索文档并思考..."):
            try:
                payload = {
                    "query": prompt,
                    "biz": biz_type,
                    "model": model_type
                }
                # 明确指定 UTF-8 编码发送
                response = requests.post(api_url, json=payload, timeout=120)
                response.encoding = 'utf-8' # 确保接收也不乱码
                result = response.json()

                answer = result.get("answer", "未获取到回答")
                sources = result.get("sources", [])

                # 3. 显示回答
                st.markdown(answer)
                
                # 4. 显示参考来源（折叠显示）
                if sources:
                    with st.expander("查看参考来源"):
                        for i, doc in enumerate(sources):
                            # 兼容对象和字符串格式
                            content = doc.get("content", str(doc)) if isinstance(doc, dict) else str(doc)
                            st.write(f"**来源 {i+1}:**")
                            st.code(content, language="text")

                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"调用 API 出错: {e}")