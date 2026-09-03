"""Step 5：LLM 生成（OpenAI 兼容 API）。

读取项目根 .env 中的 OPENAI_API_KEY / OPENAI_BASE_URL / RAG_LLM_MODEL。
System prompt 约束：仅依据检索资料作答、注明页码与章节、不得编造。
"""
from __future__ import annotations

import functools
import os
from typing import Optional

from dotenv import load_dotenv

from .config import LLM_CONTEXT_BUDGET, ROOT

load_dotenv(ROOT / ".env")

SYSTEM_PROMPT = """你是《京东集团员工手册》的政策问答助手。规则：
1. 只能依据下方【参考资料】作答，禁止编造手册中不存在的内容。
2. 每个事实点后标注来源，格式：〔第X页·章节路径〕。
3. 若资料不足以回答，明确说明"手册未查到相关内容"。
4. 回答使用简体中文，条理清晰；涉及天数/金额/比例时尽量引用原文。
5. 参考资料按相关度排序，优先采信前面条目。"""


def build_context(chunks: list[dict]) -> str:
    """把检索结果拼成上下文，超过预算则截断（保持块完整）。"""
    parts = []
    used = 0
    for c in chunks:
        text = c["text"]
        if used + len(text) > LLM_CONTEXT_BUDGET:
            break
        parts.append(f"〔第{c['page_start']}页·{c['path'] or '无章节'}〕\n{text}")
        used += len(text)
    return "\n\n".join(parts)


def env_config() -> dict:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    model = os.getenv("RAG_LLM_MODEL", "deepseek-chat").strip()
    return {"api_key": api_key, "base_url": base_url, "model": model}


def api_ready() -> bool:
    cfg = env_config()
    return bool(cfg["api_key"] and cfg["base_url"])


@functools.lru_cache(maxsize=1)
def _client():
    from openai import OpenAI

    cfg = env_config()
    if not api_ready():
        raise RuntimeError(".env 缺少 OPENAI_API_KEY 或 OPENAI_BASE_URL")
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"], timeout=90, max_retries=1)


def generate(question: str, chunks: list[dict]) -> str:
    """生成回答；失败抛出异常（由调用方兜底展示检索片段）。"""
    cfg = env_config()
    client = _client()
    context = build_context(chunks)
    if not context:
        return "（未能检索到相关手册内容，请换个问法。）"
    resp = client.chat.completions.create(
        model=cfg["model"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"问题：{question}\n\n【参考资料】\n{context}"},
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    return (resp.choices[0].message.content or "").strip()
