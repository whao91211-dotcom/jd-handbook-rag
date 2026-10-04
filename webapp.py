"""Local-only browser interface for the existing RAG pipeline."""
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Literal
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from ragcore.llm import api_ready, generate, prepare_context

ROOT = Path(__file__).resolve().parent
app = FastAPI(title="员工手册问答", docs_url=None, redoc_url=None)
pipeline_lock = Lock()
PROFILES = {
    'baseline': ('baseline', 'baseline', 'default'),
    'quality': ('guarded', 'evidence', 'low'),
    'fast': ('guarded', 'evidence', 'disabled'),
    'agent': ('guarded', 'evidence', 'low'),
}

def agent_answer(question):
    from ragcore.agent import answer_question
    return answer_question(question)

def retrieve(*args, **kwargs):
    # Heavy embedding runtime loads on the first question, not on every request.
    from ragcore.retriever import retrieve as run
    return run(*args, **kwargs)

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    profile: Literal['baseline', 'quality', 'fast', 'agent'] = 'quality'
    retrieval_only: bool = False

    @field_validator('question')
    @classmethod
    def clean_question(cls, value):
        value = value.strip()
        if not value:
            raise ValueError('请输入问题')
        return value

@app.get('/api/health')
def health():
    return {'status': 'ready', 'api_configured': api_ready(), 'busy': pipeline_lock.locked()}

@app.post('/api/ask')
def ask(request: AskRequest):
    if not pipeline_lock.acquire(blocking=False):
        raise HTTPException(409, '正在处理另一个问题，请稍后再试。')
    try:
        return run_question(request)
    finally:
        pipeline_lock.release()

def run_question(request):
    start = perf_counter()
    if request.profile == 'agent' and not request.retrieval_only and api_ready():
        try:
            result = dict(agent_answer(request.question))
            # Raw context belongs in evaluation artifacts, not the public API.
            result.pop('context', None)
            result.pop('source_map', None)
            return result
        except Exception as exc:
            missing = isinstance(exc, ImportError)
            return {'question': request.question, 'profile': 'agent', 'answer': '', 'sources': [],
                'usage': [], 'trace': [], 'status': 'agent_unavailable' if missing else 'agent_failed',
                'message': ('复杂问题模式需要项目 Agent 环境，请使用 .venv 启动服务。' if missing else
                            '复杂问题代理未能启动，请检查模型配置和服务终端。'),
                'metrics': {'retrieval_seconds': None, 'generation_seconds': None,
                    'total_seconds': perf_counter()-start, 'total_tokens': None, 'attempts': 0}}
    strategy, prompt, thinking = PROFILES[request.profile]
    result = {'question': request.question, 'profile': request.profile,
              'answer': '', 'sources': [], 'usage': [], 'status': 'complete',
              'message': '', 'metrics': {}}
    try:
        chunks = retrieve(request.question, k=8, strategy=strategy)
    except Exception:
        result.update(status='retrieval_failed', message='检索失败，请检查本地索引是否已构建，并查看服务终端。')
        result['metrics'] = {'retrieval_seconds': perf_counter()-start, 'generation_seconds': None,
                             'total_seconds': perf_counter()-start, 'total_tokens': None, 'attempts': 0}
        return result
    retrieved_at = perf_counter()
    context, mapping = prepare_context(chunks, prompt)
    reverse = {value: key for key, value in mapping.items()}
    # Baseline's budget counts body only, and breaks at the first overflow.
    if prompt == 'baseline':
        from ragcore.config import LLM_CONTEXT_BUDGET
        used = 0
        included = []
        for chunk in chunks:
            if used + len(chunk['text']) > LLM_CONTEXT_BUDGET:
                break
            included.append(chunk['id']); used += len(chunk['text'])
    else:
        included = list(mapping.values())
    for c in chunks:
        if request.retrieval_only or c['id'] in included:
            pages = c.get('pages', [c.get('page_start')])
            page = c.get('page_start', pages[0])
            result['sources'].append({'id': c['id'], 'label': reverse.get(c['id'], f"第{page}页·{c['path'] or '无章节'}"),
                'pages': pages, 'path': c['path'], 'text': c['text'],
                'in_context': c['id'] in included})
    generation_seconds = None
    if not chunks:
        result.update(status='no_evidence', message='未检索到相关条款，请换一种问法。')
    elif request.retrieval_only:
        result.update(status='retrieval_only', message='已展示检索证据，本次未调用模型。')
    elif not api_ready():
        result.update(status='not_configured', message='模型接口尚未配置。请在服务端.env填写密钥和地址；已保留检索证据。')
    else:
        generation_start = perf_counter()
        try:
            result['answer'] = generate(request.question, chunks, prompt_version=prompt,
                                       thinking_mode=thinking, usage_sink=result['usage'])
            if result['usage'] and result['usage'][-1].get('finish_reason') == 'length':
                result.update(status='truncated', message='原方案回答被截断，不能作为完整结论。请切换优化方案。')
        except Exception as exc:
            status = getattr(exc, 'status_code', None)
            message = {402:'模型接口余额不足，请补充额度。', 429:'模型接口限流，请稍后再试。',
                       401:'模型认证失败，请检查服务端配置。',403:'模型访问被拒绝，请检查服务端权限。'}.get(status)
            if not message:
                if any(u.get('error_type') for u in result['usage']):
                    message = '模型连接或请求失败，请检查网络及服务端配置。'
                else:
                    message = '模型未返回完整回答，请重试或切换快速方案。'
            result.update(status='generation_failed', message=message+' 已保留检索证据。')
        generation_seconds = perf_counter()-generation_start
    usage = result['usage']
    known = bool(usage) and all(u.get('prompt_tokens') is not None and u.get('completion_tokens') is not None for u in usage)
    result['metrics'] = {'retrieval_seconds': retrieved_at-start, 'generation_seconds': generation_seconds,
        'total_seconds': perf_counter()-start,
        'total_tokens': sum(u['prompt_tokens']+u['completion_tokens'] for u in usage) if known else None,
        'attempts': len(usage)}
    return result

@app.get('/')
def index():
    return FileResponse(ROOT/'web/index.html')

app.mount('/assets', StaticFiles(directory=ROOT/'web'), name='assets')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
