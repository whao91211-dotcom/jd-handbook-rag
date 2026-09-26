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
5. 参考资料按相关度排序，优先采信前面条目。
6. 你的职责是解释手册政策；不要代写辞职申请、协议或其他个人文件。
7. 对辞退、离职、赔偿、处分、竞业限制等可能影响个人权益的决定，
   说明手册条款的适用边界，并建议向人力资源部或相关专业人士核实个人情况；
   不替用户作个人法律或职业决定。
8. 对“现在是否生效”“今年是否调整”等时效问题，不能仅凭文档中的生效日期
   推断它仍是最新版本；资料没有当前版本证据时明确说明无法确认。"""


EVIDENCE_PROMPT = """你是员工手册政策问答助手，仅根据提供的参考资料回答。
1. 先判断用户问的是哪种事项。缺少假别、岗位、工龄、地点、日期等必要条件时，给有条件的规则并提出最少的澄清问题，不替用户选择个人结论。
2. 必须先结合事项定义、分类总则、具体条款及例外再回答。单条没有某个字样不等于没有规则：例如假别性质要与该类假期的薪酬福利总则共同理解。
3. 数值示例必须保留资格、年度累计、入职当年折算、转正及属地条件；条件不全时不给个人天数、工资或预算。上限不等于额外额度。
4. 离职按辞职、公司解除、协商解除、依法终止区分，调动按发起方区分。不同条款有张力时分别列明，不自行确定优先级或口头同意的法律效力。
5. 未定义的时间地点边界保持未知，建议向负责部门核实；不要推断21点整等边界。资料未给出的申请时限、软件名称、补考次数、费用、余额及结转规则不能编造。
6. 找不到信息时只说“提供的参考资料未明确”，不能据此宣称整本手册没有规定。已给出的相关规则应先说明，未提供的专项制度不推测其内容。
7. 仅回答与问题直接相关的内容。简体中文，优先使用“结论、规则与条件、仍需确认”三段，避免罗列无关福利和处分。
8. 每个制度事实后复制对应的来源标记，如〔来源1〕。仅引用输入中存在且支持该事实的来源，不编造页码、章节或子条号；多条共同支持时列多个来源。
9. 不代写个人文件、不替用户作法律职业决定；权益事项建议向HR或负责部门核实。生效日期不能证明当前最新版本，未给现行证据时说明无法确认。
"""


def prepare_context(chunks: list[dict], prompt_version: str = "baseline") -> tuple[str, dict]:
    if prompt_version not in ("baseline", "evidence"):
        raise ValueError("prompt_version must be baseline or evidence")
    parts = []
    used = 0
    mapping = {}
    for c in chunks:
        text = c["text"]
        if prompt_version == "baseline":
            if used + len(text) > LLM_CONTEXT_BUDGET:
                break
            parts.append(f"〔第{c['page_start']}页·{c['path'] or '无章节'}〕\n{text}")
            used += len(text)
        else:
            label = f"来源{len(parts)+1}"
            part = f"〔{label}〕 第{c['page_start']}页·{c['path'] or '无章节'}\n{text}"
            cost = len(part) + (2 if parts else 0)
            if used + cost > LLM_CONTEXT_BUDGET:
                continue
            parts.append(part)
            mapping[label] = c['id']
            used += cost
    return "\n\n".join(parts), mapping


def build_context(chunks: list[dict]) -> str:
    """Keep the historical baseline context unchanged."""
    return prepare_context(chunks, "baseline")[0]


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


def generate(question: str, chunks: list[dict], *, prompt_version: str = "baseline", usage_sink: Optional[list[dict]] = None, thinking_mode: str = 'default') -> str:
    """生成回答。

    契约：**失败必须抛异常，绝不返回空串。**
    "成功但内容为空"是一种静默失败——调用方的兜底逻辑通常挂在异常上，
    返回空串会绕过全部兜底，用户只看到一片空白（GAPS A-043）。

    预算策略：首次 LLM_MAX_TOKENS；正文为空则预算翻倍重试，封顶 CAP。
    理由：reasoning token 长度随问题难度变化，**任何固定值都有边界**，
    所以用"宽松起步 + 按需抬高 + 有限次后显式失败"，而不是猜一个够大的数。
    """
    if thinking_mode not in ('default', 'disabled', 'low'):
        raise ValueError('thinking_mode must be default, disabled or low')
    LAST_USAGE.clear()
    cfg = env_config()
    client = get_client()
    context, _ = prepare_context(chunks, prompt_version)
    if not context:
        return "（未能检索到相关手册内容，请换一种问法。）"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT if prompt_version == "baseline" else EVIDENCE_PROMPT},
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
                **({'extra_body': {'thinking': {'type': 'disabled'}}} if thinking_mode == 'disabled' else
                   {'extra_body': {'thinking': {'type': 'enabled'}}, 'reasoning_effort': 'low'} if thinking_mode == 'low' else {}),
            )
        except Exception as exc:
            status = getattr(exc, 'status_code', None)
            if usage_sink is not None:
                usage_sink.append({'attempt': i+1, 'max_tokens': budget, 'error_type': type(exc).__name__, 'status_code': status, 'prompt_tokens': None, 'completion_tokens': None})
            # Larger token budgets cannot fix credentials, balance, invalid parameters or rate limits.
            if status in (400, 401, 402, 403, 404, 429):
                raise
            attempts.append(f"第{i + 1}次(预算{budget})调用异常: {type(exc).__name__}: {exc}")
            continue

        choice = resp.choices[0]
        content = (choice.message.content or "").strip()
        usage = resp.usage
        details = getattr(usage, "completion_tokens_details", None)
        reasoning = getattr(details, "reasoning_tokens", None) if details else None

        request_usage = {
            "model": cfg["model"], "attempt": i + 1, "max_tokens": budget,
            "response_model": getattr(resp, 'model', None), "thinking_mode": thinking_mode,
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "reasoning_tokens": reasoning,
            "finish_reason": choice.finish_reason,
            "content_chars": len(content),
        }
        LAST_USAGE.clear()
        LAST_USAGE.update(request_usage)
        if usage_sink is not None:
            usage_sink.append(dict(request_usage))
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
        if prompt_version == 'evidence':
            raise EmptyGenerationError('证据回答在全部预算内均被截断，拒绝返回不完整政策。')
        print("[警告] 预算已到上限，返回最后一次被截断的部分回答", file=sys.stderr)
        return fallback
    raise EmptyGenerationError(
        "模型连续 %d 次未产出可用正文，已放弃（拒绝返回空串）。明细: %s"
        % (LLM_GEN_ATTEMPTS, " | ".join(attempts)))
