import unittest

from evaluation.run_ingestion_experiment import build_parents, covered_fraction, locate_quote


class IngestionEvaluationTests(unittest.TestCase):
    def test_overlap_is_counted_once_and_clipped_to_reference(self):
        gold = {'parent': 'a', 'start': 10, 'end': 30}
        hits = [{'parent': 'a', 'start': 0, 'end': 20},
                {'parent': 'a', 'start': 15, 'end': 25}]
        self.assertEqual(covered_fraction(gold, hits), .75)

    def test_other_section_cannot_cover_reference(self):
        self.assertEqual(covered_fraction({'parent': 'a', 'start': 0, 'end': 10},
            [{'parent': 'b', 'start': 0, 'end': 10}]), 0)

    def test_quote_mapping_ignores_whitespace_but_requires_unique_match(self):
        parent = {'parent': 'a', 'content': '前言。病假\n需要证明。结束。'}
        span = locate_quote('病假需要证明。', [parent])
        self.assertEqual(parent['content'][span['start']:span['end']], '病假\n需要证明。')
        self.assertIsNone(locate_quote('', [parent]))
        self.assertIsNone(locate_quote('病假需要证明。', [parent, parent]))

    def test_reference_offsets_recover_content_without_labels_in_parent(self):
        rows = [{'id': 'old-a', 'chapter': '第一章', 'section': '第一节',
                 'content': '第一条规则。', 'pages': [1]},
                {'id': 'old-b', 'chapter': '第一章', 'section': '第一节',
                 'content': '第二条规则。', 'pages': [2]}]
        parents, references = build_parents(rows)
        self.assertEqual(len(parents), 1)
        text = parents[0]['content']
        for row in rows:
            span = references[row['id']]
            self.assertEqual(text[span['start']:span['end']], row['content'])
        self.assertNotIn('old-a', str(parents))
        self.assertNotIn('old-b', str(parents))


if __name__ == '__main__':
    unittest.main()
