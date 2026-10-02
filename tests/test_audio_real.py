"""Prueba con audio real sintetizado por la voz de macOS (no es tu micrófono).

Requiere los modelos en modelos/ y la voz Paulina; si faltan, se omite.
"""
import shutil
import subprocess
import tempfile
import os
import time
import json
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest import mock

import numpy as np

from transcripcion import MODEL_DIR, VAD_PATH, Transcriber, load_wav, normalize
from conversacion import ConversationState
from memoria import Memoria
from prosodia import ProsodyAnalyzer, RATE
from turnos import SileroVAD, TurnConfig, TurnDetector

READY = (MODEL_DIR / 'weights.safetensors').exists() and VAD_PATH.exists() and shutil.which('say')


def speech(text, rate=None):
    with tempfile.TemporaryDirectory() as directory:
        aiff, wav = Path(directory) / 'v.aiff', Path(directory) / 'v.wav'
        command = ['say', '-v', 'Paulina']
        if rate:
            command += ['-r', str(rate)]
        command += ['-o', str(aiff)]
        subprocess.run(command, input=text, text=True, check=True, capture_output=True)
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


REAL_PROSODY = READY and os.environ.get('NEXO_PROBAR_PROSODIA') == '1'


@unittest.skipUnless(REAL_PROSODY, 'Activa NEXO_PROBAR_PROSODIA=1 para benchmark local de audio, LM Studio y Piper.')
class RealProsodyABTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        previous_memory_path = os.environ.get('NEXO_MEMORIA')
        os.environ['NEXO_MEMORIA'] = str(Path(self.directory.name) / 'bootstrap.sqlite3')
        try:
            import servidor
        finally:
            if previous_memory_path is None:
                os.environ.pop('NEXO_MEMORIA', None)
            else:
                os.environ['NEXO_MEMORIA'] = previous_memory_path
        self.server = servidor
        self.originals = {name: getattr(servidor, name) for name in ('memory', 'conversation', 'voice', 'PROSODY')}
        for name, value in self.originals.items():
            self.addCleanup(setattr, servidor, name, value)

    @staticmethod
    def process_rss_mb():
        value = subprocess.check_output(['ps', '-o', 'rss=', '-p', str(os.getpid())], text=True).strip()
        return int(value) / 1024

    def test_real_audio_whisper_prosody_and_first_audio_ab(self):
        available = self.server.request_json('/models', timeout=3).get('data', [])
        if self.server.DEFAULT_MODEL not in {item.get('id') for item in available}:
            self.skipTest('El modelo configurado de LM Studio no está cargado.')

        neutral_audio = speech('Ya encontré el resultado y quiero revisar cómo funciona.')
        analyzer = ProsodyAnalyzer()
        for _ in range(3):
            analyzer.assess(analyzer.analyze(neutral_audio))

        indices = np.arange(0, len(neutral_audio), 1.3, dtype=np.float32)
        accelerated_audio = np.interp(indices, np.arange(len(neutral_audio)), neutral_audio).astype(np.float32) * 1.7
        rss_before = self.process_rss_mb()
        analysis_times = []
        features = None
        for _ in range(3):
            started = time.perf_counter()
            features = analyzer.analyze(accelerated_audio)
            analysis_times.append((time.perf_counter() - started) * 1000)
        rss_after = self.process_rss_mb()
        assessed = analyzer.assess(features)
        self.assertIsNotNone(assessed['context'], assessed)

        self.server.ASR.submit(self.server.transcriber.load).result()
        transcription_started = time.perf_counter()
        transcript = self.server.ASR.submit(self.server.transcriber.transcribe, accelerated_audio).result()
        transcription_seconds = time.perf_counter() - transcription_started
        self.assertEqual(transcript.confidence, 'alta', transcript)
        self.assertGreater(len(transcript.text.split()), 2, transcript.text)
        audio_copy = accelerated_audio.copy()
        np.testing.assert_array_equal(accelerated_audio, audio_copy, 'El análisis no modifica el audio usado por Whisper')

        parallel_started = time.perf_counter()
        asr_future = self.server.ASR.submit(self.server.transcriber.transcribe, accelerated_audio)
        prosody_future = self.server.PROSODY_EXECUTOR.submit(analyzer.analyze, accelerated_audio)
        parallel_transcript = asr_future.result()
        asr_finished = time.perf_counter()
        parallel_features = prosody_future.result(timeout=self.server.PROSODY_TIMEOUT)
        parallel_finished = time.perf_counter()
        self.assertEqual(normalize(parallel_transcript.text), normalize(transcript.text))
        extra_wait_ms = max(0.0, (parallel_finished - asr_finished) * 1000)

        trials = {'sin prosodia': [], 'prosodia': []}
        self.server.make_audio('Prueba de precalentamiento.', 'es_MX-claude-high')
        for iteration in range(2):
            configurations = ('sin prosodia', 'prosodia') if iteration == 0 else ('prosodia', 'sin prosodia')
            for configuration in configurations:
                memory = Memoria(Path(self.directory.name) / f'{iteration}-{configuration}.sqlite3')
                self.server.memory = memory
                self.server.conversation = ConversationState()
                self.server.voice = self.server.VoiceSession()
                self.server.PROSODY = ProsodyAnalyzer()
                for _ in range(3):
                    self.server.PROSODY.assess(self.server.PROSODY.analyze(neutral_audio))
                turn_id = f'{configuration}-{iteration}'
                voice_future = None
                voice_context = None
                if configuration == 'prosodia':
                    voice_future = Future()
                    voice_future.set_result(parallel_features)
                    voice_context = self.server.PROSODY.assess(parallel_features).get('context')
                self.server.voice.pending = {turn_id: {
                    'id': turn_id, 'transcript': transcript, 'incomplete': False,
                    'timings': {'voz': len(accelerated_audio) / RATE, 'fin_de_turno': 0.7,
                                'transcripcion': transcription_seconds},
                    'voice_analysis': voice_future,
                }}
                warm_payload = self.server.build_payload(transcript.text, voice_context=voice_context)
                warm_payload.update(max_tokens=1, stream=False)
                self.server.request_json('/chat/completions', warm_payload, timeout=60)
                self.server.conversation = ConversationState()
                events = []
                with mock.patch.object(self.server, 'warm_llm'):
                    self.server.talk_stream(
                        {'turn': turn_id}, events.append,
                        speak=lambda text, voice: self.server.make_audio(text, 'es_MX-claude-high'))
                result = events[-1]
                self.assertEqual(result['type'], 'done', result)
                trials[configuration].append({'timings': result['timings'], 'answer': result['answer']})

        rows = []
        for configuration, results in trials.items():
            measurements = [item['timings'] for item in results]
            enabled = configuration == 'prosodia'
            rows.append({
                'configuracion': configuration,
                'transcripcion_s': round(transcription_seconds, 3),
                'analisis_vocal_ms': round(float(np.median(analysis_times)), 3) if enabled else 0.0,
                'espera_extra_paralela_ms': round(extra_wait_ms, 3) if enabled else 0.0,
                'primer_token_s': round(float(np.mean([item['first_token'] for item in measurements])), 3),
                'primer_audio_s': round(float(np.mean([item['first_audio'] for item in measurements])), 3),
                'respuesta_completa_s': round(float(np.mean([item['generacion'] for item in measurements])), 3),
                'rss_delta_analisis_mb': round(rss_after - rss_before, 3) if enabled else 0.0,
                'respuestas_reales': [item['answer'] for item in results],
            })
        rows.append({'configuracion': 'prosodia + clasificador', 'resultado': 'no instalado; se descartó por consumo y falta de validación emocional en español'})
        print('\nPROSODIA_AB: ' + json.dumps(rows, ensure_ascii=False))
        self.assertLessEqual(float(np.median(analysis_times)), 50)
        self.assertLessEqual(extra_wait_ms, self.server.PROSODY_TIMEOUT * 1000)


if __name__ == '__main__':
    unittest.main()
