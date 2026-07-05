"""
token校验、统一API
"""
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import traceback
import chromadb
import os
from typing import Dict
from retriever import retrieve
from reranker import rerank
from fill import fill_database
from db import get_collection
from config import DEFAULT_MODEL, COLLECTION_TEMPLATE, DEFAULT_BIZ, RERANK_THRESHOLD


class RAGHandler(BaseHTTPRequestHandler):

    def _send(self, data, code=200):
        self.send_response(code)
        self.send_header("Content-type", "application/json; charset=utf-8")
        self.end_headers()
        response = json.dumps(data, ensure_ascii=False, indent=4)
        self.wfile.write(response.encode('utf-8'))

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length", 0))

            if length <= 0:
                return None

            raw_body = self.rfile.read(length)
            return json.loads(raw_body.decode("utf-8"))

        except json.JSONDecodeError:
            return None

    def do_POST(self):
        def check_auth(data: dict, target_chat_id) -> bool:
            if not data or "token" not in data:
                return False
            token = data.get("token")
            if not token:
                return False

            cid = str(target_chat_id).strip()

            GLOBAL_ADMIN_TOKEN = os.getenv("RAG_ADMIN_TOKEN", "SYSTEM_SUPER_ADMIN_TOKEN")
            if token == GLOBAL_ADMIN_TOKEN:
                return True

            tokens_file_path = os.path.join("config", "tokens.json")

            if not os.path.exists(tokens_file_path):
                print(f"[WARNING] Can't find {tokens_file_path}, token validate failed.")
                return False
            
            try:
                with open(tokens_file_path, "r", encoding="utf-8") as f:
                    tenant_tokens =json.load(f)
            except Exception as e:
                print(f"error: Fail to read file: {e}")
                return False

            expected_token = tenant_tokens.get(cid)
            if expected_token and token == expected_token:
                return True

            return False

        if self.path == "/rag/delete":
            try:             
                data = self._read_json()
                if not data:
                    self._send({"error": "Invalid JSON"}, 400)
                    return

                chat_id = data.get("chat_id")
                biz = data.get("biz", DEFAULT_BIZ)

                if not chat_id or not str(chat_id).strip():
                    self._send({"error": "chat_id cannot be empty"}, 400)
                    return

                cid = str(chat_id).strip()

                if not check_auth(data, cid):
                    self._send({"error": "Unauthorized"}, 401)
                    return

                target_coll = get_collection(biz)

                if target_coll:
                    target_coll.delete(where={"chat_id": {"$eq": cid}})
                    self._send({"message": f"Successfully cleared workspace: {cid} in biz: {biz}"})
                else:
                    self._send({"error": "Database connection failed"}, 500)
                    return

            except Exception as e:
                traceback.print_exc()
                self._send({"error": str(e)}, 500)
            return

        if self.path == "/rag/update":
            try:
                data = self._read_json()

                if not data:
                    self._send({"error": "Invalid JSON"}, 400)
                    return

                chat_id = data.get("chat_id")
                if not chat_id or not str(chat_id).strip():
                    self._send({"error": "chat_id cannot be empty"}, 400)
                    return

                cid = str(chat_id).strip()

                request_file = data.get("json_file", "data/data.json")
                file_name = os.path.basename(request_file)
                if not file_name.endswith('.json'):
                    self._send({"error": "Only JSON files are allowed"}, 400)
                    return

                json_file = os.path.join("data", file_name)

                if not check_auth(data, cid):
                    self._send({"error": "Unauthorized"}, 401)
                return

                biz = data.get("biz", DEFAULT_BIZ)
                result = fill_database(
                    workspace_id=cid, 
                    biz=biz, 
                    json_file=json_file, 
                    agent_id=data.get("agent_id", "unknown"), 
                    group_id=data.get("group_id", "")
                )
                self._send({"message": "Update successful", "detail": result})

            except Exception as e:
                traceback.print_exc()
                self._send({"error": str(e)}, 500)
            return

        if self.path == "/rag/query":
            try:
                data = self._read_json()
                if not data:
                    self._send({"error": "Invalid JSON"}, 400)
                    return

                query = data.get("query")
                inbound_context = data.get("inbound_context", {})

                chat_id = data.get("chat_id")
                if isinstance(inbound_context, dict):
                    chat_id = (
                        inbound_context.get("chat_id") or
                        inbound_context.get("user_id") or
                        chat_id
                    )

                if not query or not chat_id:
                    self._send({"error": "query or dynamic chat id cannot be empty"}, 400)
                    return

                cid = str(chat_id).strip()

                if not check_auth(data, cid):
                    self._send({"error": "Unauthorized"}, 401)
                    return

                model = data.get("model", DEFAULT_MODEL)
                biz = data.get("biz", DEFAULT_BIZ)

                print(f"\n [Telegram ID: {cid}] | question: {query[:50]}")

                raw_docs = retrieve(query=query, workspace_id=cid, model=model, biz=biz)
                print(f"retrieve: {len(raw_docs)}")

                docs = rerank(query, raw_docs) if raw_docs else []

                result_lines = []
                formatted_sources = []

                threshold = RERANK_THRESHOLD
                has_knowledge = False

                for i, d in enumerate(docs):
                    score = d.metadata.get("relevance_score", 0.0)
                    print(f"Ranking: {i+1} 分数: {score:.4f} | text: {d.page_content[:15]}...")

                    if score >= threshold:
                        has_knowledge = True
                        answer = d.metadata.get("answer", "No preset answer")
                        result_lines.append(
                            f"### Ranking {len(result_lines) + 1} | Similarity Score: {score:.4f}\n"
                            f"**Base Questions:** {d.page_content}\n"
                            f"**Standard Answer:** {answer}"
                        )

                    formatted_sources.append({
                        "matched_question": d.page_content,
                        "content": d.metadata.get("answer", ""),
                        "score": round(score, 4)
                    })

                context_for_ai = "\n\n---\n\n".join(result_lines) if result_lines else ""

                self._send({
                    "query": query,
                    "answer": context_for_ai,
                    "has_knowledge": has_knowledge,
                    "sources": formatted_sources,
                    "biz": biz
                })
            except Exception as e:
                traceback.print_exc()
                self._send({"error": str(e)}, 500)
            return

        self._send({"error": "Not found"}, 404)

def run():
    target_collection_name = COLLECTION_TEMPLATE.format(biz=DEFAULT_BIZ)
    client = chromadb.HttpClient(host='localhost', port=8000)
    collection = client.get_or_create_collection(name=target_collection_name)

    server = HTTPServer(("0.0.0.0", 8001), RAGHandler)
    server.collection = collection

    print("API Server is running on port 8001: http://localhost:8001")
    print(f"Target Collection: {target_collection_name}")
    server.serve_forever()

if __name__ == "__main__":
    run()
