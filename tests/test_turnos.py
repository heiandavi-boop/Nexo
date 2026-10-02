"""Pruebas del fin de turno con un VAD simulado: cada muestra contiene la probabilidad de voz."""
import unittest
from concurrent.futures import Future

import numpy as np

from transcripcion import build_transcript
from turnos import FRAME, TurnConfig, TurnDetector, looks_incomplete

DT = FRAME / 16000
CHUNK = 1536


def tone(prob, seconds):
    return np.full(int(round(seconds / DT)) * FRAME, prob, dtype=np.float32)


class FakeVAD:
    def reset(self):
        pass

    def __call__(self, frame):
        return float(frame[0])


class FakeASR:
    def __init__(self, *texts):
        self.texts, self.calls = list(texts), []

    def __call__(self, audio):
        self.calls.append(audio)
        text = self.texts[min(len(self.calls), len(self.texts)) - 1]
        words = [{'word': ' ' + w, 'probability': 0.95} for w in text.split()]
        return build_transcript([{'text': text, 'avg_logprob': -0.1, 'no_speech_prob': 0.01, 'compression_ratio': 1.1, 'words': words}])


class ManualExecutor:
    def __init__(self):
        self.jobs = []

    def submit(self, fn, *args):
        future = Future()
        self.jobs.append((future, fn, args))
        return future

    def finish(self, index=-1):
        future, fn, args = self.jobs[index]
        if future.set_running_or_notify_cancel():
            future.set_result(fn(*args))


def run(detector, audio, start=0.0):
    """Devuelve (segundo, evento) para cada turno, descarte o error."""
    events = []
    for i in range(0, len(audio), CHUNK):
        event = detector.feed(audio[i:i + CHUNK])
        if event['turn'] or event['discarded'] or event['error']:
            events.append((start + (i + CHUNK) / 16000, event))
    return events


def turns(events):
    return [event['turn'] for _, event in events if event['turn']]


class TurnTests(unittest.TestCase):
    def make(self, *texts, executor=None, **config):
        self.asr = FakeASR(*texts) if texts else FakeASR('Hola.')
        return TurnDetector(FakeVAD(), self.asr, TurnConfig.create(config), executor)

    def test_silence_and_background_noise_never_open_a_turn(self):
        detector = self.make()
        self.assertEqual(run(detector, np.concatenate([tone(0.0, 5), tone(0.3, 5)])), [])
        self.assertEqual(self.asr.calls, [])

    def test_short_noise_burst_is_discarded_without_transcribing(self):
        detector = self.make()
        events = run(detector, np.concatenate([tone(0.95, 0.1), tone(0.0, 2)]))
        self.assertEqual(turns(events), [])
        self.assertEqual([e['discarded'] for _, e in events], ['ruido breve'])
        self.assertEqual(self.asr.calls, [])

    def test_complete_sentence_ends_after_short_pause(self):
        detector = self.make('Quiero saber el clima de mañana.')
        events = run(detector, np.concatenate([tone(0.0, 0.5), tone(0.9, 1.5), tone(0.0, 3)]))
        found = turns(events)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]['transcript'].text, 'Quiero saber el clima de mañana.')
        self.assertGreaterEqual(found[0]['timings']['fin_de_turno'], 0.7)
        self.assertLess(found[0]['timings']['fin_de_turno'], 0.85)
        self.assertLess(events[0][0] - 2.0, 1.0, 'Ya no hay espera fija de 3 segundos')

    def test_pause_inside_sentence_does_not_split_the_turn(self):
        detector = self.make('Mañana quiero ir a la', 'Mañana quiero ir a la playa con mis amigos.')
        audio = np.concatenate([tone(0.9, 1.2), tone(0.0, 1.1), tone(0.9, 1.0), tone(0.0, 3)])
        found = turns(run(detector, audio))
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]['transcript'].text, 'Mañana quiero ir a la playa con mis amigos.')
        self.assertGreater(len(self.asr.calls[-1]), 2.1 * 16000, 'Incluye ambas partes de la frase')

    def test_incomplete_phrase_still_ends_after_long_pause(self):
        detector = self.make('Quiero ir a la')
        found = turns(run(detector, np.concatenate([tone(0.9, 1.0), tone(0.0, 3)])))
        self.assertEqual(len(found), 1)
        self.assertTrue(found[0]['incomplete'])
        self.assertGreaterEqual(found[0]['timings']['fin_de_turno'], 1.6)
        self.assertLess(found[0]['timings']['fin_de_turno'], 1.75)

    def test_prebuffer_keeps_first_syllables(self):
        detector = self.make('Pablo vino.')
        run(detector, np.concatenate([tone(0.0, 1), tone(0.2, 0.4), tone(0.9, 1.0), tone(0.0, 2)]))
        clip = self.asr.calls[0]
        weak = np.flatnonzero(np.isclose(clip, 0.2))
        self.assertEqual(len(weak), len(tone(0.2, 0.4)), 'Conserva el inicio suave antes de que el VAD confirme la voz')
        self.assertLess(weak[0] / 16000, 0.1)

    def test_tail_keeps_last_syllables_without_long_silence(self):
        detector = self.make('Compré manzanas verdes.')
        run(detector, np.concatenate([tone(0.9, 1.0), tone(0.4, 0.2), tone(0.0, 2)]))
        clip = self.asr.calls[0]
        self.assertEqual(int(np.sum(np.isclose(clip, 0.4))), len(tone(0.4, 0.2)), 'Final suave conservado por histéresis')
        trailing = len(clip) - (np.flatnonzero(clip > 0.1)[-1] + 1)
        self.assertGreaterEqual(trailing / 16000, 0.25)
        self.assertLessEqual(trailing / 16000, 0.35)

    def test_turn_is_emitted_only_once(self):
        detector = self.make('Hola, ¿qué tal?')
        audio = np.concatenate([tone(0.9, 1.0), tone(0.0, 6)])
        self.assertEqual(len(turns(run(detector, audio))), 1)
        self.assertEqual(len(self.asr.calls), 1)

    def test_resumed_speech_cancels_stale_transcription(self):
        executor = ManualExecutor()
        detector = self.make('Necesito que me recuerdes llamar a Marta.', executor=executor)
        run(detector, np.concatenate([tone(0.9, 1.0), tone(0.0, 0.4)]))
        self.assertEqual(len(executor.jobs), 1)
        run(detector, np.concatenate([tone(0.9, 1.0), tone(0.0, 0.4)]))
        self.assertEqual(len(executor.jobs), 2)
        self.assertTrue(executor.jobs[0][0].cancelled())
        executor.finish()
        found = turns(run(detector, tone(0.0, 1.0)))
        self.assertEqual([t['transcript'].text for t in found], ['Necesito que me recuerdes llamar a Marta.'])

    def test_waits_for_slow_transcription_instead_of_cutting(self):
        executor = ManualExecutor()
        detector = self.make('Vale.', executor=executor)
        events = run(detector, np.concatenate([tone(0.9, 1.0), tone(0.0, 3)]))
        self.assertEqual(events, [])
        self.assertEqual(detector.feed(np.zeros(0, np.float32))['state'], 'transcribiendo')
        executor.finish()
        self.assertEqual(len(turns(run(detector, tone(0.0, 0.1)))), 1)

    def test_noise_hallucination_is_discarded(self):
        detector = self.make('Subtítulos realizados por la comunidad de Amara.org')
        events = run(detector, np.concatenate([tone(0.9, 0.6), tone(0.0, 2)]))
        self.assertEqual(turns(events), [])
        self.assertEqual(events[0][1]['discarded'], 'frase típica de ruido')

    def test_long_monologue_is_closed_at_maximum(self):
        detector = self.make('Una historia larga.', turno_maximo=5)
        found = turns(run(detector, tone(0.9, 7)))
        self.assertEqual(len(found), 1)
        self.assertLessEqual(found[0]['audio'], 5.4)

    def test_config_is_clamped(self):
        config = TurnConfig.create({'pausa_corta': 99, 'pausa_larga': 0.1, 'umbral_inicio': 'x', 'otro': 1})
        self.assertEqual(config.pausa_corta, 2.5)
        self.assertEqual(config.pausa_larga, 2.5)
        self.assertEqual(config.umbral_inicio, 0.5)


class IncompleteTests(unittest.TestCase):
    def test_linguistic_cues(self):
        for text in ['Quiero ir a la', 'Me gusta porque', 'Tengo dos perros y', 'Pues, o sea', 'Estaba pensando en…', 'Eh,']:
            self.assertTrue(looks_incomplete(text), text)
        for text in ['¿Qué hora es?', 'Hasta luego.', 'Así es.', 'No sé qué.', 'Para mí.', 'Nada más.', '¡Qué bien!']:
            self.assertFalse(looks_incomplete(text), text)


if __name__ == '__main__':
    unittest.main()
