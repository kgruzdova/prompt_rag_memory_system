"""Offline regression tests: python -m unittest discover -s tests -v"""
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class GroundingTests(unittest.TestCase):
    def test_both_backends(self):
        for backend in ('bot_openai', 'bot_proxy'):
            with self.subTest(backend=backend):
                spec = importlib.util.spec_from_file_location(
                    'grounding', ROOT / backend / 'rag' / 'grounding.py')
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                # The relevant section is beyond the old 3000-character cutoff.
                text = (ROOT / 'data/docs/sample3.txt').read_text(encoding='utf-8')
                self.assertGreater(text.index('10. Консалтинг и аудит'), 3000)
                metadata = [
                    {'text': text, 'source': 'sample3.txt (часть 1/1)'},
                    {'text': 'Организация рабочего дня и перерывы.', 'source': 'Test2.txt'},
                ]
                context, sources = module.build_evidence(
                    'Вы проводите консалтинг и аудит?', metadata,
                    [('Организация рабочего дня', 'Test2.txt', 0.1)])
                self.assertLessEqual(len(context), 3000)
                self.assertIn('Консалтинг и аудит', context)
                self.assertIn('Аудит кодовой базы', context)
                self.assertEqual(sources[0], 'sample3.txt')
                answer, used = module.cited_sources('Да, проводим аудит [1].', sources)
                self.assertEqual(used, ['sample3.txt'])
                self.assertEqual(answer, 'Да, проводим аудит.')
                self.assertEqual(module.cited_sources('Выдуманный ответ', sources)[1], [])
                self.assertEqual(module.cited_sources('Ответ [99]', sources)[1], [])
                self.assertEqual(module.build_evidence('Вопрос', [], []), ('', []))
                # Repeated citation must not duplicate filenames.
                self.assertEqual(module.cited_sources('Аудит [1], консультации [1].', sources)[1], ['sample3.txt'])


if __name__ == '__main__':
    unittest.main()