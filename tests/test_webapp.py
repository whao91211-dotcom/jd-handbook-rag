import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
import webapp

CHUNKS=[{'id':'a','text':'事假属于无薪假。','pages':[23],'path':'休假','rrf':.03}, {'id':'b','text':'后续条款','pages':[21],'path':'总则','rrf':.02}]
class WebTests(unittest.TestCase):
 def setUp(self):
  self.client=TestClient(webapp.app)
  self.patches=[patch('webapp.retrieve',return_value=CHUNKS),patch('webapp.prepare_context',return_value=('context',{'来源1':'a'})),patch('webapp.api_ready',return_value=True)]
  for p in self.patches:p.start()
 def tearDown(self):
  for p in self.patches:p.stop()
 def test_validation(self):
  for body in [{'question':'   '},{'question':'x'*501},{'question':'test','profile':'other'}]:self.assertEqual(self.client.post('/api/ask',json=body).status_code,422)
 def test_retrieval_only_has_all_hits_and_does_not_call_model(self):
  with patch('webapp.generate') as g:
   r=self.client.post('/api/ask',json={'question':'事假','retrieval_only':True}).json()
   self.assertEqual(len(r['sources']),2);self.assertEqual(r['status'],'retrieval_only');g.assert_not_called()
 def test_actual_context_and_usage(self):
  def fake(*a,**kw):kw['usage_sink'].append({'prompt_tokens':10,'completion_tokens':5,'finish_reason':'stop'});return '回答〔来源1〕'
  with patch('webapp.generate',side_effect=fake):
   r=self.client.post('/api/ask',json={'question':'事假'}).json()
   self.assertEqual([s['id'] for s in r['sources']],['a']);self.assertEqual(r['metrics']['total_tokens'],15);self.assertEqual(r['answer'],'回答〔来源1〕')
 def test_failure_retains_sources_and_redacts_exception(self):
  with patch('webapp.generate',side_effect=RuntimeError('secret-api-key')):
   r=self.client.post('/api/ask',json={'question':'事假'}).json()
   self.assertEqual(r['status'],'generation_failed');self.assertTrue(r['sources']);self.assertNotIn('secret-api-key',str(r))
 def test_missing_configuration(self):
  with patch('webapp.api_ready',return_value=False):
   r=self.client.post('/api/ask',json={'question':'事假'}).json();self.assertEqual(r['status'],'not_configured');self.assertTrue(r['sources'])
 def test_busy(self):
  webapp.pipeline_lock.acquire()
  try:self.assertEqual(self.client.post('/api/ask',json={'question':'test'}).status_code,409)
  finally:webapp.pipeline_lock.release()
 def test_truncated_is_not_presented_as_complete(self):
  def fake(*a,**kw):kw['usage_sink'].append({'finish_reason':'length','prompt_tokens':1,'completion_tokens':2});return '部分'
  with patch('webapp.generate',side_effect=fake):
   r=self.client.post('/api/ask',json={'question':'事假','profile':'baseline'}).json();self.assertEqual(r['status'],'truncated')
 def test_health_does_not_expose_credentials(self):
  r=self.client.get('/api/health').json();self.assertEqual(set(r),{'status','api_configured','busy'})
if __name__=='__main__':unittest.main()
