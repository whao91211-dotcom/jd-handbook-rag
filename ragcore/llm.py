"""Step 5：LLM 生成（OpenAI 兼容 API）。

读取项目根 .env 中的 OPENAI_API_KEY / OPENAI_BASE_URL / RAG_LLM_MODEL。
System prompt 约束：仅依据检索资料作答、注明页码与章节、不得编造。
"""
from __future__ import annotations

import functools
import os
import sys
from typing import Optional

from dotenv import load_dotenv

from .config import (
    LLM_CONTEXT_BUDGET,
    LLM_GEN_ATTEMPTS,
    LLM_MAX_TOKENS,
    LLM_MAX_TOKENS_CAP,
    ROOT,
)

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


def _http_client():
    """显式决定代理策略：默认直连，**不静默继承系统代理**。

    本机实测：Windows 系统代理 `127.0.0.1:7897` 会被 httpx 继承，导致 TLS 握手失败
    （httpcore.ConnectError: EOF occurred in violation of protocol），外部表现是
    所有 LLM 调用抛 APIConnectionError —— 与 API key 无关，极难排查。
    确实需要代理时显式设 `RAG_HTTP_PROXY`。
    """
    try:
        import httpx2 as httpx          # 本环境 openai 依赖的 httpx 分支
    except ImportError:                 # pragma: no cover
        import httpx

    proxy = os.getenv("RAG_HTTP_PROXY", "").strip()
    if proxy:
        return httpx.Client(proxy=proxy, timeout=90)
    return httpx.Client(trust_env=False, timeout=90)


@functools.lru_cache(maxsize=1)
def get_client():
    """OpenAI 兼容客户端（显式代理策略，见 `_http_client`）。"""
    from openai import OpenAI

    cfg = env_config()
    if not api_ready():
        raise RuntimeError(".env 缺少 OPENAI_API_KEY 或 OPENAI_BASE_URL")
    return OpenAI(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        timeout=90,
        max_retries=1,
        http_client=_http_client(),
    )


class EmptyGenerationError(RuntimeError):
    """模型未产出正文（reasoning 吃光预算 / 内容被过滤 / 空响应）。见 GAPS A-043。"""


# 最近一次生成的实际用量，供成本与诊断使用（原先 resp.usage 被直接丢弃）
LAST_USAGE: dict = {}


def last_usage() -> dict:
    """返回最近一次生成的实际 token 用量。"""
    return dict(LAST_USAGE)


def generate(question: str, chunks: list[dict]) -> str:
    """生成回答。

    契约：**失败必须抛异常，绝不返回空串。**
    "成功但内容为空"是一种静默失败——调用方的兜底逻辑通常挂在异常上，
    返回空串会绕过全部兜底，用户只看到一片空白（GAPS A-043）。

    预算策略：首次 LLM_MAX_TOKENS；正文为空则预算翻倍重试，封顶 CAP。
    理由：reasoning token 长度随问题难度变化，**任何固定值都有边界**，
    所以用"宽松起步 + 按需抬高 + 有限次后显式失败"，而不是猜一个够大的数。
    """
    cfg = env_config()
    client = get_client()
    context = build_context(chunks)
    if not context:
        return "（未能检索到相关手册内容，请换一种问法。）"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"问题：{question}\n\n【参考资料】\n{context}"},
    ]
    attempts: list[str] = []
    fallback: Optional[str] = None   # 有内容但被截断时保留，作为全部失败后的兜底
    for i in range(LLM_GEN_ATTEMPTS):
        budget = min(LLM_MAX_TOKENS * (2 ** i), LLM_MAX_TOKENS_CAP)
        try:
            resp = client.chat.completions.create(
                model=cfg["model"], messages=messages,
                temperature=0.2, max_tokens=budget,
            )
        except Exception as exc:
            attempts.append(f"第{i + 1}次(预算{budget})调用异常: {type(exc).__name__}: {exc}")
            continue

        choice = resp.choices[0]
        content = (choice.message.content or "").strip()
        usage = resp.usage
        details = getattr(usage, "completion_tokens_details", None)
        reasoning = getattr(details, "reasoning_tokens", None) if details else None

        LAST_USAGE.clear()
        LAST_USAGE.update({
            "model": cfg["model"], "attempt": i + 1, "max_tokens": budget,
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "reasoning_tokens": reasoning,
            "finish_reason": choice.finish_reason,
            "content_chars": len(content),
        })
        if content and choice.finish_reason != "length":
            return content                       # 只有"有正文且正常收尾"才算成功

        if content:
            # 有正文但被截断：政策答案被切一半可能误导，所以也重试，但先留作兜底
            fallback = content
            attempts.append("第%d次(预算%d)被截断: reasoning_tokens=%s, 正文%d字"
                            % (i + 1, budget, reasoning, len(content)))
            print("[警告] 回答被截断，抬高预算重试 -> %s" % attempts[-1], file=sys.stderr)
        else:
            attempts.append("第%d次(预算%d)正文为空: finish_reason=%s, reasoning_tokens=%s"
                            % (i + 1, budget, choice.finish_reason, reasoning))
            print("[警告] LLM 未产出正文，抬高预算重试 -> %s" % attempts[-1], file=sys.stderr)

    if fallback:
        print("[警告] 预算已到上限，返回最后一次被截断的部分回答", file=sys.stderr)
        return fallback
    raise EmptyGenerationError(
        "模型连续 %d 次未产出可用正文，已放弃（拒绝返回空串）。明细: %s"
        % (LLM_GEN_ATTEMPTS, " | ".join(attempts)))
