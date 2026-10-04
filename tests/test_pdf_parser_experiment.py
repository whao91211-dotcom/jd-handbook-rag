import unittest

from evaluation.run_pdf_parser_experiment import audit_page, evidence_coverage


class PdfParserEvaluationTests(unittest.TestCase):
    def test_present_words_with_wrong_order_do_not_pass_order_check(self):
        result = audit_page('第一条。第三条。第二条。', ['第一条', '第二条', '第三条'])
        self.assertEqual(result['present_n'], 3)
        self.assertFalse(result['order_complete'])

    def test_table_relation_requires_tuple_not_just_individual_numbers(self):
        result = audit_page('5年以上10年以下 12个月 15个月内累计病休时间', [],
                            [['5年以上10年以下', '9个月', '15个月内累计病休时间']])
        self.assertEqual(result['table_tuple_n'], 0)

    def test_retrieved_text_on_wrong_page_cannot_satisfy_source_probe(self):
        probes = [{'page': 22, 'fragments': ['9个月', '15个月内累计病休时间']}]
        hits = [{'pages': [21], 'text': '9个月15个月内累计病休时间'}]
        self.assertEqual(evidence_coverage(probes, hits), 0)
        hits[0]['pages'] = [22]
        self.assertEqual(evidence_coverage(probes, hits), 1)


if __name__ == '__main__':
    unittest.main()
