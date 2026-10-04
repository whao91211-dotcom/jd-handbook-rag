import asyncio
from contextlib import asynccontextmanager
import importlib
import importlib.util
import unittest
import time
from types import SimpleNamespace
from unittest.mock import patch


def chunk(i, text='规则'):
    return {'id': i, 'text': text, 'pages': [21], 'page_start': 21, 'path': '休假'}


def call(name, **kwargs):
    return SimpleNamespace(tool_id='call-' + name, tool_name=name, tool_kwargs=kwargs)


class ScriptedLLM:
    def __init__(self, steps):
        self.steps = iter(steps)
        self.messages = []

    async def achat_with_tools(self, tools, **kwargs):
        self.messages.append(list(kwargs['chat_history']))
        step = next(self.steps)
        if isinstance(step, Exception):
            raise step
        return SimpleNamespace(calls=step, message=SimpleNamespace(content='不要使用模型自身的政策猜测'),
                               raw={'usage': {'prompt_tokens': 10, 'completion_tokens': 5}})

    def get_tool_calls_from_response(self, response, **kwargs):
        return response.calls


class AgentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('ragcore.agent'), 'Missing bounded evidence agent')
        self.agent = importlib.import_module('ragcore.agent')

    async def collect(self, steps, **kwargs):
        return await self.agent.collect_evidence('试用期请病假，工资和材料分别怎么规定？',
            llm=ScriptedLLM(steps), search=kwargs.pop('search', lambda q: [chunk('a')]),
            read=kwargs.pop('read', lambda i: [chunk('a'), chunk('neighbor')]), **kwargs)

    async def test_adaptive_search_adds_missing_evidence_and_tool_results(self):
        llm = ScriptedLLM([[call('search_handbook', query='试用期 病假')],
                           [call('search_handbook', query='病假 工资 材料')], []])
        result = await self.agent.collect_evidence('试用期病假工资和材料', llm=llm,
            search=lambda q: [chunk('a' if '试用期' in q else 'b')])
        self.assertEqual([c['id'] for c in result['chunks']], ['a', 'b'])
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(len(result['usage']), 3)
        self.assertIn('"id": "a"', str(llm.messages[1]))
        self.assertEqual([s['query'] for s in result['trace']], ['试用期 病假', '病假 工资 材料'])

    async def test_fourth_search_is_not_executed_even_in_one_batch(self):
        searched = []
        result = await self.collect([[call('search_handbook', query=str(i)) for i in range(4)]],
            search=lambda q: searched.append(q) or [chunk(q)])
        self.assertEqual(searched, ['0', '1', '2'])
        self.assertEqual(result['status'], 'limited')
        self.assertEqual([c['id'] for c in result['chunks']], ['0', '1', '2'])

    async def test_third_read_is_not_executed_and_duplicates_are_removed(self):
        reads = []
        result = await self.collect([[call('search_handbook', query='病假')],
            [call('read_adjacent', chunk_id='a') for _ in range(3)]],
            read=lambda i: reads.append(i) or [chunk('a'), chunk('b')])
        self.assertEqual(len(reads), 2)
        self.assertEqual([c['id'] for c in result['chunks']], ['a', 'b'])
        self.assertEqual(result['status'], 'limited')

    async def test_unknown_anchor_never_reads_corpus(self):
        reads = []
        result = await self.collect([[call('read_adjacent', chunk_id='invented')]],
            read=lambda i: reads.append(i) or [chunk('invented')])
        self.assertEqual(reads, [])
        self.assertEqual(result['status'], 'agent_failed')

    async def test_invalid_tool_arguments_do_not_reach_search(self):
        searches = []
        result = await self.collect([[call('search_handbook', query='x', unexpected='secret')]],
            search=lambda q: searches.append(q) or [])
        self.assertEqual(searches, [])
        self.assertEqual(result['status'], 'agent_failed')

    async def test_failure_preserves_sources_and_redacts_exception(self):
        result = await self.collect([[call('search_handbook', query='病假')], RuntimeError('secret-api-key')])
        self.assertEqual([c['id'] for c in result['chunks']], ['a'])
        self.assertEqual(result['status'], 'agent_failed')
        self.assertNotIn('secret-api-key', str(result))
        self.assertIsNone(result['usage'][-1]['prompt_tokens'])

    async def test_timeout_preserves_preceding_evidence(self):
        class SlowLLM(ScriptedLLM):
            async def achat_with_tools(self, tools, **kwargs):
                if self.messages:
                    await asyncio.sleep(1)
                return await super().achat_with_tools(tools, **kwargs)
        result = await self.agent.collect_evidence('病假', llm=SlowLLM([[call('search_handbook', query='病假')], []]),
            search=lambda q: [chunk('a')], limits=self.agent.AgentLimits(orchestration_seconds=.05))
        self.assertEqual(result['status'], 'timeout')
        self.assertEqual([c['id'] for c in result['chunks']], ['a'])

    async def test_tool_timeout_is_not_counted_as_an_extra_model_request(self):
        def slow(query):
            time.sleep(.3)
            return [chunk('late')]
        result = await self.agent.collect_evidence('病假',
            llm=ScriptedLLM([[call('search_handbook', query='病假')]]), search=slow,
            limits=self.agent.AgentLimits(orchestration_seconds=.02))
        self.assertEqual(result['status'], 'timeout')
        self.assertEqual(result['trace'][-1]['status'], 'timeout')
        self.assertEqual(len(result['usage']), 1)

    async def test_sync_entry_does_not_wait_for_timed_out_local_tool(self):
        from llama_index.core.tools import FunctionTool  # Warm imports outside the measurement.
        def run():
            def slow(query):
                time.sleep(.4)
                return [chunk('late')]
            started = time.perf_counter()
            result = asyncio.run(self.agent.collect_evidence('病假',
                llm=ScriptedLLM([[call('search_handbook', query='病假')]]), search=slow,
                limits=self.agent.AgentLimits(orchestration_seconds=.02)))
            return result, time.perf_counter()-started
        result, elapsed = await asyncio.to_thread(run)
        self.assertEqual(result['status'], 'timeout')
        self.assertLess(elapsed, .2)

    async def test_model_prose_without_evidence_is_not_used_as_an_answer(self):
        result = await self.collect([[]])
        self.assertEqual(result['status'], 'no_evidence')
        self.assertEqual(result['chunks'], [])

    async def test_orchestration_call_cap_prevents_endless_loop(self):
        result = await self.collect([[call('search_handbook', query='病假')], []],
            limits=self.agent.AgentLimits(model_calls=1))
        self.assertEqual(result['status'], 'limited')
        self.assertEqual(len(result['usage']), 1)

    async def test_final_context_excludes_overflow_and_sources_match_citations(self):
        async def synth(messages):
            self.assertNotIn('巨大', messages[-1].content)
            return SimpleNamespace(message=SimpleNamespace(content='规则〔来源1〕'),
                raw={'usage': {'prompt_tokens': 20, 'completion_tokens': 10}, 'choices': [{'finish_reason': 'stop'}]})
        result = await self.agent.answer_question_async('病假',
            llm=ScriptedLLM([[call('search_handbook', query='病假')], []]),
            search=lambda q: [chunk('big', '巨大'*4000), chunk('small', '病假规则')], synthesizer=synth)
        self.assertEqual(result['source_map'], {'来源1': 'small'})
        self.assertEqual(result['answer'], '规则〔来源1〕')
        self.assertEqual(result['metrics']['total_tokens'], 60)
        self.assertFalse(result['sources'][0]['in_context'])

    async def test_unknown_citation_blocks_final_answer(self):
        async def synth(messages):
            return SimpleNamespace(message=SimpleNamespace(content='错误〔来源99〕'), raw={})
        result = await self.agent.answer_question_async('病假',
            llm=ScriptedLLM([[call('search_handbook', query='病假')], []]),
            search=lambda q: [chunk('a')], synthesizer=synth)
        self.assertEqual(result['status'], 'citation_failed')
        self.assertEqual(result['answer'], '')
        self.assertIsNone(result['metrics']['total_tokens'])

    async def test_synthesis_timeout_does_not_lose_collected_sources(self):
        async def synth(messages):
            await asyncio.sleep(1)
        result = await self.agent.answer_question_async('病假',
            llm=ScriptedLLM([[call('search_handbook', query='病假')], []]),
            search=lambda q: [chunk('a')], synthesizer=synth,
            limits=self.agent.AgentLimits(synthesis_seconds=.01))
        self.assertEqual(result['status'], 'timeout')
        self.assertEqual(result['sources'][0]['id'], 'a')
        self.assertIsNone(result['metrics']['total_tokens'])

    async def test_truncated_synthesis_retries_and_counts_both_responses(self):
        attempts = []
        async def synth(messages):
            attempts.append(1)
            return SimpleNamespace(message=SimpleNamespace(content='半截' if len(attempts)==1 else '完整〔来源1〕'),
                raw={'usage': {'prompt_tokens': 20, 'completion_tokens': 10},
                     'choices': [{'finish_reason': 'length' if len(attempts)==1 else 'stop'}]})
        result = await self.agent.answer_question_async('病假',
            llm=ScriptedLLM([[call('search_handbook', query='病假')], []]),
            search=lambda q: [chunk('a')], synthesizer=synth)
        self.assertEqual(result['answer'], '完整〔来源1〕')
        self.assertEqual(result['metrics']['attempts'], 4)
        self.assertEqual(result['metrics']['total_tokens'], 90)

    async def test_retry_token_budgets_reach_real_llamaindex_request_transport(self):
        import httpx
        import json
        from llama_index.llms.openai_like import OpenAILike
        budgets = []
        def handler(request):
            budgets.append(json.loads(request.content)['max_tokens'])
            return httpx.Response(200, json={'id':'test','object':'chat.completion','created':0,'model':'fake',
                'choices':[{'index':0,'message':{'role':'assistant','content':'规则〔来源1〕'},
                            'finish_reason':'length' if len(budgets)==1 else 'stop'}],
                'usage':{'prompt_tokens':20,'completion_tokens':10,'total_tokens':30}})
        @asynccontextmanager
        async def model_session(*args, **kwargs):
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                yield OpenAILike(model='fake', api_key='test', api_base='https://example.test/v1',
                    is_chat_model=True, is_function_calling_model=True, max_tokens=2048,
                    async_http_client=client, max_retries=0)
        with patch.object(self.agent, 'configured_llm', model_session):
            result = await self.agent.answer_question_async('病假',
                llm=ScriptedLLM([[call('search_handbook', query='病假')], []]), search=lambda q: [chunk('a')])
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(budgets, [2048, 4096])

    async def test_adjacent_reader_uses_source_order(self):
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            corpus = Path(folder)/'chunks.jsonl'
            corpus.write_text('\n'.join(__import__('json').dumps(chunk(i)) for i in ('z', 'a', 'b', 'c')), encoding='utf-8')
            with patch('ragcore.config.CHUNKS_JSONL', corpus):
                self.assertEqual([c['id'] for c in self.agent.read_adjacent_chunks('a')], ['z', 'a', 'b'])


if __name__ == '__main__':
    unittest.main()
