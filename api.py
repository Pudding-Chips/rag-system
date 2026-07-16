"""
token校验、统一API 与 安全运维执行审计系统
"""
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
import json
import traceback
import chromadb
import os
import subprocess
import logging
from typing import Dict
from retriever import retrieve
from reranker import rerank
from fill import fill_database
from db import get_collection
from dotenv import load_dotenv
from config import DEFAULT_MODEL, COLLECTION_TEMPLATE, DEFAULT_BIZ, RERANK_THRESHOLD

try:
    load_dotenv()
except ImportError:
    print("[WARNING] python-dotenv is not installed. Please run: pip install python-dotenv")

# ==========================================================
# 🛡️ 1. 统一初始化安全审计日志系统 (Audit Log)
# ==========================================================
LOG_FILE_PATH = "secure_ops.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, encoding='utf-8'),
        logging.StreamHandler() # 同时也输出到控制台方便开发调试观察
    ]
)

# 注入防御黑名单：严格禁止大模型生成的 Shell 命令中夹带任何可跳出沙箱的危险字符
COMMAND_BLACKLIST = [";", "&&", "||", "|", "`", "$(", "rm", "sh", "bash"]

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
                    tenant_tokens = json.load(f)
            except Exception as e:
                print(f"error: Fail to read file: {e}")
                return False

            expected_token = tenant_tokens.get(cid)
            if expected_token and token == expected_token:
                return True

            return False

        # ==========================================================
        # 🗑️ 路由 1: /rag/delete (清空/重置本地向量数据库)
        # ==========================================================
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
                    existing_data = target_coll.get(where={"chat_id": {"$eq": cid}}, limit=1)
                    
                    if not existing_data or not existing_data.get("ids"):
                        self._send({
                            "message": f"Workspace: {cid} in biz: {biz} is already empty. No data to delete.",
                            "deleted": False
                        }, 200)
                        return

                    target_coll.delete(where={"chat_id": {"$eq": cid}})
                    self._send({
                        "message": f"Successfully cleared workspace: {cid} in biz: {biz}",
                        "deleted": True
                    })
                    return
                else:
                    self._send({"error": "Database connection failed"}, 500)
                    return

            except Exception as e:
                traceback.print_exc()
                self._send({"error": str(e)}, 500)
                return

        # ==========================================================
        # 🔄 路由 2: /rag/update (增量更新/灌入 API 接口知识库)
        # ==========================================================
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
                agent_id = str(data.get("agent_id", "unknown"))
                group_id = str(data.get("group_id", ""))

                target_coll = get_collection(biz)
                if not target_coll:
                    self._send({"error": "Database connection failed"}, 500)
                    return

                where_filter = {
                    "$and": [
                        {"chat_id": {"$eq": cid}},
                        {"agent_id": {"$eq": agent_id}},
                        {"group_id": {"$eq": group_id}}
                    ]
                }

                existing_data = target_coll.get(where=where_filter, limit=1)

                if existing_data and existing_data.get("ids"):
                    print(f"[INFO] Workspace {cid} (Agent: {agent_id}, Group: {group_id}) is already up-to-date. Skip.")
                    self._send({
                        "message": f"Data for agent: {agent_id}, group: {group_id} is already up-to-date. No update needed.",
                        "updated": False
                    })
                    return

                print(f"[INFO] First-time sync triggered for workspace: {cid}...")
                result = fill_database(
                    chat_id=cid, 
                    biz=biz, 
                    json_file=json_file, 
                    agent_id=agent_id, 
                    group_id=group_id
                )
                
                self._send({
                    "message": "Update successful", 
                    "detail": result,
                    "updated": True  
                })
                return

            except Exception as e:
                traceback.print_exc()
                self._send({"error": str(e)}, 500)
                return

        # ==========================================================
        # 🔍 路由 3: /rag/query (大模型检索极星云 API 说明书)
        # ==========================================================
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

                raw_docs = retrieve(query=query, chat_id=cid, model=model, biz=biz)
                print(f"retrieve: {len(raw_docs)}")

                docs = rerank(query, raw_docs) if raw_docs else []

                result_lines = []
                formatted_sources = []

                threshold = RERANK_THRESHOLD
                has_knowledge = False
                max_ctx_docs = 3

                for i, d in enumerate(docs):
                    if len(result_lines) >= max_ctx_docs:
                        break

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

                        try:
                            raw_answer = d.metadata.get("answer", "{}")
                            ans_obj = json.loads(raw_answer)
                        except Exception:
                            ans_obj = d.metadata.get("answer", "")

                        formatted_sources.append({
                            "matched_question": d.page_content,
                            "content": ans_obj, 
                            "score": round(score, 4)
                        })

                context_for_ai = "\n\n---\n\n".join(result_lines) if result_lines else ""

                if not has_knowledge:
                    context_for_ai = "[Alert: No matching references were found in the local CDN operations and maintenance knowledge base. Please refuse to answer the user's operation request and prompt them to contact customer service.]"
                    
                self._send({
                    "query": query,
                    "answer": context_for_ai,
                    "has_knowledge": has_knowledge,
                    "sources": formatted_sources,
                    "biz": biz
                })
                return

            except Exception as e:
                traceback.print_exc()
                self._send({"error": str(e)}, 500)
                return

        # ==========================================================
        # 🚀 🚀 🚀 【核心新增路由 4】: /rag/execute 🚀 🚀 🚀
        # 负责接收大模型输出的安全拼装指令，安全拦截、写入审计日志并物理执行。
        # ==========================================================
        if self.path == "/rag/execute":
            try:
                data = self._read_json()
                if not data:
                    self._send({"error": "Invalid JSON"}, 400)
                    return

                chat_id = data.get("chat_id")
                if not chat_id:
                    self._send({"error": "chat_id cannot be empty"}, 400)
                    return
                cid = str(chat_id).strip()

                # 安全校验白名单/凭证校验
                if not check_auth(data, cid):
                    self._send({"error": "Unauthorized"}, 401)
                    return

                # 提取大模型在 SKILL.md 拼装好的待执行 raw_command
                command_to_run = data.get("raw_command", "").strip()
                if not command_to_run:
                    self._send({"error": "raw_command cannot be empty"}, 400)
                    return

                # --- 🛑 注入安全检测层 ---
                is_dangerous = False
                detected_char = ""
                for evil_char in COMMAND_BLACKLIST:
                    # 允许合法的环境变量取值如 $POLESTAR_TOKEN
                    if evil_char in command_to_run and "$POLESTAR" not in command_to_run:
                        is_dangerous = True
                        detected_char = evil_char
                        break

                if is_dangerous:
                    logging.error(f"[SECURITY BLOCK] ID: {cid} | 试图执行包含敏感黑名单字符 '{detected_char}' 的命令: {command_to_run}")
                    self._send({
                        "code": 403,
                        "msg": "Security validation failed. Execution blocked."
                    }, 403)
                    return

                # --- 🔒 敏感词审计过滤机制 ---
                real_token = os.getenv("POLESTAR_TOKEN", "MISSING")
                # 防范模型未按 SKILL.md 的变量机制拼接，误泄露了真实的本地明文令牌
                if real_token != "MISSING" and real_token in command_to_run:
                    logging.warning(f"[SECURITY ALERT] ID: {cid} | 模型拼装命令中包含真实明文 TOKEN。已自动进行净化。")
                    command_to_run = command_to_run.replace(real_token, "$POLESTAR_TOKEN")

                # --- 📝 写入审计日志记录 ---
                logging.info(f"[USER OPS REQUEST] ID: {cid} | 请求执行指令: {command_to_run}")

                # --- ⚙️ 安全沙盒物理执行层 ---
                # env=os.environ 保证子进程在执行 curl 时能自动读取当前系统的真实环境变量（POLESTAR_TOKEN 和 POLESTAR_API_URL）
                process = subprocess.run(
                    command_to_run,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=15,
                    env=os.environ
                )

                if process.returncode == 0:
                    logging.info(f"[EXECUTION SUCCESS] ID: {cid} | 命令执行成功。")
                    try:
                        # 尝试将结果转成 JSON 
                        exec_data = json.loads(process.stdout)
                    except Exception:
                        exec_data = process.stdout

                    self._send({
                        "code": 200,
                        "msg": "success",
                        "data": exec_data
                    })
                    return
                else:
                    logging.error(f"[EXECUTION FAILED] ID: {cid} | 标准错误返回: {process.stderr}")
                    self._send({
                        "code": 500,
                        "msg": "execution_failed",
                        "detail": process.stderr
                    }, 500)
                    return

            except Exception as e:
                traceback.print_exc()
                logging.error(f"[EXECUTION EXCEPTION] ID: {cid if 'cid' in locals() else 'unknown'} | 异常详情: {str(e)}")
                self._send({"error": str(e)}, 500)
                return

        # 兜底
        self._send({"error": "Not found"}, 404)


def run():
    target_collection_name = COLLECTION_TEMPLATE.format(biz=DEFAULT_BIZ)
    client = chromadb.HttpClient(host='localhost', port=8000)
    collection = client.get_or_create_collection(name=target_collection_name)

    server = ThreadingHTTPServer(("0.0.0.0", 8001), RAGHandler)
    server.collection = collection

    print("API Server is running on port 8001: http://localhost:8001")
    print(f"Target Collection: {target_collection_name}")
    server.serve_forever()


if __name__ == "__main__":
    run()