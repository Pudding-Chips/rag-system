# =========================
# api.py
# =========================
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import traceback

# 导入检索和重排序逻辑
from retriever import retrieve
from reranker import rerank

class RAGHandler(BaseHTTPRequestHandler):

    def _send(self, data, code=200):
        self.send_response(code)
        self.send_header("Content-type", "application/json; charset=utf-8")
        self.end_headers()
        response = json.dumps(data, ensure_ascii=False, indent=4)
        self.wfile.write(response.encode('utf-8'))

    def do_POST(self):
        if self.path != "/rag/query":
            self._send({"error": "Not found"}, 404)
            return

        length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(length)
        
        try:
            body_str = raw_body.decode('utf-8')
            data = json.loads(body_str)
            
            query = data.get("query")
            model = data.get("model", "text2vec")
            biz = data.get("biz", "cdn")

            print(f"\n[检索请求] 问题: {query}")

            # 1. 召回
            raw_docs = retrieve(query, model=model, biz=biz)
            
            # 2. 精排
            docs = rerank(query, raw_docs)

            # 3. 优化后的展示逻辑
            result_lines = []
            formatted_sources = []
            
            # 建议阈值设为 0.3。
            # 1.0 代表完全一致，0.3-0.5 代表语义高度相关
            threshold = 0.3 

            for i, d in enumerate(docs):
                score = d.metadata.get("relevance_score", 0.0)
                print(f"DEBUG: 排名 {i+1} 分数: {score:.4f}")

                # 只有超过阈值的结果才会被放入最终回答
                if score >= threshold:
                    answer = d.metadata.get("answer", "无预设回答")
                    result_lines.append(
                        f"### 排名 {len(result_lines) + 1} | 相似度得分: {score:.4f}\n"
                        f"**知识库问题:** {d.page_content}\n"
                        f"**标准回答:** {answer}"
                    )

                # sources 始终保留前几个结果的得分，方便调试查看
                formatted_sources.append({
                    "matched_question": d.page_content,
                    "content": d.metadata.get("answer", ""),
                    "score": round(score, 4)
                })

            # 4. 返回结果判断
            if not result_lines:
                final_answer = "❌ 匹配度低于阈值 (当前阈值: {})，未找到合适内容。".format(threshold)
            else:
                final_answer = "\n\n---\n\n".join(result_lines)

            self._send({
                "query": query,
                "answer": final_answer,
                "sources": formatted_sources
            })

        except Exception as e:
            traceback.print_exc()
            self._send({"error": str(e)}, 500)

def run():
    server = HTTPServer(("", 8001), RAGHandler)
    print("API 服务已启动: http://localhost:8001")
    server.serve_forever()

if __name__ == "__main__":
    run()