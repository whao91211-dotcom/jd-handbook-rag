import unittest

import pdfplumber

from ragcore.config import PDF_PATH
from ragcore.extract import extract_page_content, page_markdown


class ExtractionTests(unittest.TestCase):
    def test_failed_table_detection_still_preserves_readable_body(self):
        class FailedTablePage:
            page_number = 1

            def find_tables(self):
                raise ValueError('broken table geometry')

            def extract_text(self):
                return 'Native text remains readable.'

        text, tables = extract_page_content(FailedTablePage())
        self.assertEqual(text, 'Native text remains readable.')
        self.assertEqual(tables, [])

    def test_table_region_filter_preserves_url_outside_detected_cells(self):
        # The PDF visibly has this URL; table detection only captures its label.
        with pdfplumber.open(PDF_PATH) as pdf:
            text, tables = extract_page_content(pdf.pages[47])
        rendered = page_markdown({'text': text, 'tables': tables})
        self.assertIn('访问地址：ssc.jd.com', rendered)
        self.assertEqual(rendered.count('ssc.jd.com'), 1)

    def test_medical_table_retains_row_values_without_body_duplication(self):
        with pdfplumber.open(PDF_PATH) as pdf:
            text, tables = extract_page_content(pdf.pages[21])
        rendered = page_markdown({'text': text, 'tables': tables})
        compact = ''.join(rendered.split())
        self.assertIn('15个月内累计病休时间', compact)
        self.assertEqual(compact.count('15个月内累计病休时间'), 1)
        self.assertTrue(tables)


if __name__ == '__main__':
    unittest.main()
