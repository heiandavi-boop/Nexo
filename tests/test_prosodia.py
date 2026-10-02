"""Análisis prosódico relativo: sin clasificador, red ni persistencia."""
import unittest

import numpy as np

from prosodia import ProsodyAnalyzer, RATE


def voice(seconds=3.0, amplitude=0.06, pitch=170, syllabic_rate=3.2, modulation=0.12):
    samples = np.arange(int(seconds * RATE), dtype=np.float32) / RATE
    envelope = 1 + modulation * np.sin(2 * np.pi * syllabic_rate * samples)
    fundamental = np.sin(2 * np.pi * pitch * samples)
    harmonic = 0.25 * np.sin(4 * np.pi * pitch * samples)
    return (amplitude * envelope * (fundamental + harmonic)).astype(np.float32)


class ProsodyTests(unittest.TestCase):
    def test_three_neutral_turns_build_only_an_in_memory_baseline(self):
        analyzer = ProsodyAnalyzer()
        for _ in range(3):
            result = analyzer.assess(analyzer.analyze(voice()))
            self.assertIsNone(result['context'])
        self.assertEqual(len(analyzer._baseline['energy']), 3)
        self.assertEqual(len(analyzer._baseline['pitch']), 3)

    def test_relative_energy_pitch_and_rate_change_is_a_qualified_signal(self):
        analyzer = ProsodyAnalyzer()
        for _ in range(3):
            analyzer.assess(analyzer.analyze(voice()))
        result = analyzer.assess(analyzer.analyze(
            voice(amplitude=0.10, pitch=220, syllabic_rate=4.4, modulation=0.3)))
        self.assertTrue(result['usable'])
        self.assertEqual(result['candidate'], 'activación vocal superior a la habitual')
        self.assertEqual(result['level'], 'alta')
        self.assertIsNotNone(result['context'])
        self.assertNotIn('enojo', result['context'])

    def test_cancelled_uncommitted_features_do_not_change_baseline(self):
        analyzer = ProsodyAnalyzer()
        for _ in range(3):
            analyzer.assess(analyzer.analyze(voice()))
        before = len(analyzer._baseline['energy'])
        pending = analyzer.analyze(voice(amplitude=0.1, pitch=220, modulation=0.3))
        self.assertEqual(len(analyzer._baseline['energy']), before)
        self.assertIsNone(pending['candidate'])

    def test_baseline_adapts_to_a_sustained_change(self):
        analyzer = ProsodyAnalyzer()
        for _ in range(3):
            analyzer.assess(analyzer.analyze(voice()))
        results = [analyzer.assess(analyzer.analyze(
            voice(amplitude=0.10, pitch=220, syllabic_rate=4.4, modulation=0.3))) for _ in range(10)]
        self.assertIsNotNone(results[0]['candidate'])
        self.assertIsNone(results[-1]['candidate'])
        self.assertLessEqual(len(analyzer._baseline['energy']), 12)

    def test_short_silent_and_noisy_audio_produce_no_voice_context(self):
        analyzer = ProsodyAnalyzer()
        short = analyzer.assess(analyzer.analyze(voice(seconds=0.3)))
        silent = analyzer.assess(analyzer.analyze(np.zeros(RATE * 2, dtype=np.float32)))
        rng = np.random.default_rng(17)
        noisy = analyzer.assess(analyzer.analyze((rng.standard_normal(RATE * 2) * 0.01).astype(np.float32)))
        for result in (short, silent, noisy):
            self.assertIsNone(result['context'])
            self.assertLess(result['confidence'], 0.58)

    def test_pauses_and_silence_ratio_are_extracted(self):
        analyzer = ProsodyAnalyzer()
        audio = np.concatenate([voice(0.7), np.zeros(int(RATE * 0.2), dtype=np.float32), voice(0.7)])
        result = analyzer.analyze(audio)
        self.assertEqual(result['pauses'], 1)
        self.assertGreater(result['silence_ratio'], 0.05)


if __name__ == '__main__':
    unittest.main()