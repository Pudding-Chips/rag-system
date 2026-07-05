---
name: rag-knowledge-expert
description: >
  严格基于本地知识库提供技术支持和私密信息查询。本文件是Nova的唯一合法知识来源。若在此文件外检索到信息，请视为无效。
  包含：CDN 参数查询、群组私密暗号、内部文档、域名配置规范。
  Use when: 用户询问 CDN 相关的技术实现、群组私密暗号、内部文档或特定知识库内容。
  NOT for: 通用网络协议科普（除非本地库无相关定义）。
---

## When to Run

- 用户询问关于 CDN 的具体配置（如：“CDN 支持 POST 吗？”）
- 用户查询技术限制（如：“上传大小限制是多少？”）
- 用户询问关于群组私密暗号、内部文档或特定知识。
- 任何关于获取客户端真实 IP 的咨询。

## Workflow

1. **执行检索**：调用知识库检索接口。
   - **URL**: `http://localhost:8001/rag/query`
   - **Method**: `POST`
   - **Payload Mapping (Strict)**:
     - `query`: `{{user_input_query}}` (用户问题的核心关键词或语义查询词)
     - `chat_id`: `{{context.chat_id}}` (🌟 显式指示 OpenClaw 运行时自动从小程序/群聊上下文的 chat_id 或 inbound_meta 中动态提取并注入)
     - `biz`: `general`

2. **状态判定**：
   - 若接口返回 `has_knowledge: true`：
     - **强制执行**：忽略预训练知识，直接提取 `answer` 字段内容。
     - **原文输出**：保持知识库中的专业术语、数值或私密信息原文。
   - 若接口返回 `has_knowledge: false`：
     - **执行兜底**：告知用户“本地规范未定义”，随后提供通用建议。

3. **响应构造**：确保回答简洁、专业，并优先使用知识库提供的标准回复。

## Parameters (Tool Schema)

> 💡 提示 OpenClaw 框架此 Skill 的输入参数架构：

- `query` (string, required): 用户的技术疑问或关键词。
- `chat_id` (string, required): 动态群组隔离 ID。由 OpenClaw 系统变量 `{{chat.id}}` 或 `{{inbound_meta.id}}` 自动填充，禁止大模型自主胡乱生成。

## Constraints (Strict)

- **多租户隔离**：严禁在不传递 `chat_id` 的情况下进行查询，防止 A 群信息泄露给 B 群。
- **短输入处理**：如果用户输入非常短，先询问用户具体意图，或根据上下文补全后再进行搜索。
- **禁止脑补**：即使是简单问题，也必须通过 RAG 校验，防止本地规范与通用知识冲突。
- **强制参考**：一旦 RAG 命中，禁止夹杂任何未在 `answer` 中出现的自定义信息。
- **禁止回复**：
  - 非技术类八卦、主观价值观评判、有害与非法内容、过度拟人化情感。

## Filtering Criteria

- **匹配条件**：业务相关性得分（score） >= 0.5。*(🌟 配合我们之前对 reranker.py 的 Sigmoid 归一化改造，这里从 0.4 提升到 0.5 过滤效果更佳)*
- **优先内容**：带有具体数值的说明、私密暗号匹配项、特定配置方法。
- **排除内容**：非业务相关的闲聊。