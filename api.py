from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import traceback

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

            raw_docs = retrieve(query, model=model, biz=biz)
            
            docs = rerank(query, raw_docs)

            result_lines = []
            formatted_sources = []
            
            threshold = 0.3 

            for i, d in enumerate(docs):
                score = d.metadata.get("relevance_score", 0.0)
                print(f"DEBUG: 排名 {i+1} 分数: {score:.4f}")

                if score >= threshold:
                    answer = d.metadata.get("answer", "无预设回答")
                    result_lines.append(
                        f"### 排名 {len(result_lines) + 1} | 相似度得分: {score:.4f}\n"
                        f"**知识库问题:** {d.page_content}\n"
                        f"**标准回答:** {answer}"
                    )

                formatted_sources.append({
                    "matched_question": d.page_content,
                    "content": d.metadata.get("answer", ""),
                    "score": round(score, 4)
                })

                        # 修改 api.py 中的逻辑
            if not result_lines:
                # 告诉 AI：库里没找到，你可以用你自带的知识（比如搜到的通用获取IP方法）来回答
                self._send({
                    "query": query,
                    "answer": "", 
                    "status": "not_found",
                    "msg": "知识库中无相关规范，请基于通用知识回答。"
                })
            else:
                # 告诉 AI：找到了，请务必参考这个回答
                self._send({
                    "query": query,
                    "answer": final_answer,
                    "status": "success"
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
