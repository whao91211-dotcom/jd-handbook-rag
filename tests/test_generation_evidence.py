import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from ragcore import llm


class GenerationTests(unittest.TestCase):
    def test_exhausted_balance_is_not_retried_with_larger_budget(self):
        class BalanceError(Exception):
            status_code = 402
        calls = []
        def create(**kwargs):
            calls.append(kwargs)
            raise BalanceError('balance exhausted')
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        chunks=[{'id':'a','text':'规则','page_start':1,'path':'政策'}]
        usage=[]
        with patch.object(llm,'get_client',return_value=client), patch.object(llm,'env_config',return_value={'model':'fake'}):
            with self.assertRaises(BalanceError):
                llm.generate('问题',chunks,usage_sink=usage)
        self.assertEqual(len(calls),1)
        self.assertEqual(usage[0]['status_code'],402)

    def prepare(self, chunks):
        function = getattr(llm, 'prepare_context', None)
        self.assertIsNotNone(function, 'Missing traceable evidence context')
        return function(chunks, 'evidence')

    def test_source_labels_match_included_chunks(self):
        chunks=[{'id':'a','text':'假期规则','page_start':21,'path':'休假'},
                {'id':'b','text':'餐补规则','page_start':30,'path':'福利'}]
        text, mapping=self.prepare(chunks)
        self.assertIn('〔来源1〕',text)
        self.assertIn('〔来源2〕',text)
        self.assertEqual(mapping, {'来源1':'a','来源2':'b'})

    def test_oversized_chunk_does_not_hide_later_small_evidence(self):
        chunks=[{'id':'big','text':'x'*6001,'page_start':1,'path':'规则'},
                {'id':'small','text':'可回答的规则','page_start':2,'path':'规则'}]
        text,mapping=self.prepare(chunks)
        self.assertLessEqual(len(text),6000)
        self.assertEqual(mapping, {'来源1':'small'})

    def test_retry_usage_is_per_request_and_preserves_every_response(self):
        params=inspect.signature(llm.generate).parameters
        self.assertIn('usage_sink',params,'Missing per-request usage capture')
        attempts=[]
        def create(**kwargs):
            attempts.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='答复'),
                finish_reason='length' if len(attempts)==1 else 'stop')],
                usage=SimpleNamespace(prompt_tokens=10,completion_tokens=20,completion_tokens_details=None))
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        chunks=[{'id':'a','text':'规则','page_start':1,'path':'政策'}]
        first=[]
        with patch.object(llm,'get_client',return_value=client), patch.object(llm,'env_config',return_value={'model':'fake'}):
            llm.generate('问题',chunks,prompt_version='evidence',usage_sink=first)
            second=[]
            llm.generate('另一个问题',chunks,prompt_version='evidence',usage_sink=second)
        self.assertEqual(len(first),2)
        self.assertEqual(sum(u['completion_tokens'] for u in first),40)
        self.assertEqual(len(second),1)
        self.assertEqual(len(first),2)


if __name__=='__main__':
    unittest.main()
