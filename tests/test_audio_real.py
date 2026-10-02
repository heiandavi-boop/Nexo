"""Prueba con audio real sintetizado por la voz de macOS (no es tu micrófono).

Requiere los modelos en modelos/ y la voz Paulina; si faltan, se omite.
"""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from transcripcion import MODEL_DIR, VAD_PATH, Transcriber, load_wav, normalize
from turnos import SileroVAD, TurnConfig, TurnDetector

READY = (MODEL_DIR / 'weights.safetensors').exists() and VAD_PATH.exists() and shutil.which('say')


def speech(text):
    with tempfile.TemporaryDirectory() as directory:
        aiff, wav = Path(directory) / 'v.aiff', Path(directory) / 'v.wav'
        subprocess.run(['say', '-v', 'Paulina', '-o', str(aiff)], input=text, text=True, check=True, capture_output=True)
        subprocess.run(['afconvert', '-f', 'WAVE', '-d', 'LEI16@16000', '-c', '1', str(aiff), str(wav)], check=True, capture_output=True)
        return load_wav(wav)


def silence(seconds):
    return np.zeros(int(seconds * 16000), dtype=np.float32)


@unittest.skipUnless(READY, 'Faltan modelos locales o la voz Paulina')
class RealAudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.asr = Transcriber()
        cls.asr.load()
        cls.vad = SileroVAD()

    def detect(self, audio, hints=()):
        detector = TurnDetector(self.vad, lambda clip: self.asr.transcribe(clip, hints), TurnConfig())
        found = []
        for i in range(0, len(audio), 1536):
            event = detector.feed(audio[i:i + 1536])
            if event['turn']:
                found.append(event['turn'])
        return found

    def test_first_and_last_words_survive(self):
        found = self.detect(np.concatenate([silence(0.3), speech('Pablo compró tres manzanas verdes en el mercado.'), silence(2)]))
        self.assertEqual(len(found), 1)
        words = normalize(found[0]['transcript'].text).split()
        self.assertEqual(words[0], 'pablo')
        self.assertEqual(words[-1], 'mercado')

    def test_pause_inside_sentence_keeps_one_turn(self):
        audio = np.concatenate([speech('Mañana quiero ir a la'), silence(1.0), speech('playa con mis amigos.'), silence(2.5)])
        found = self.detect(audio)
        self.assertEqual(len(found), 1, [t['transcript'].text for t in found])
        text = normalize(found[0]['transcript'].text)
        self.assertIn('manana', text)
        self.assertIn('playa con mis amigos', text)

    def test_silence_and_noise_produce_nothing(self):
        rng = np.random.default_rng(1)
        noise = (rng.standard_normal(16000 * 4) * 0.05).astype(np.float32)
        self.assertEqual(self.detect(np.concatenate([silence(3), noise, silence(2)])), [])

    def test_name_hint_is_not_forced_into_unrelated_speech(self):
        found = self.detect(np.concatenate([speech('Hoy hace un día muy agradable.'), silence(2)]), ['Anderson Trujillo'])
        self.assertEqual(len(found), 1)
        self.assertNotIn('anderson', normalize(found[0]['transcript'].text))


if __name__ == '__main__':
    unittest.main()
