import io
import os
import unittest
import wave
from unittest import mock

import numpy as np

import voz

PIPER_READY = (voz.VOICES_DIR / 'es_MX-claude-high.onnx').exists()


def duration(data):
    with wave.open(io.BytesIO(data)) as wav:
        return wav.getnframes() / wav.getframerate()


def tiny_wav(seconds=0.1, rate=16000):
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b'\x00\x00' * int(seconds * rate))
    return buffer.getvalue()


class VoiceTests(unittest.TestCase):
    def test_existing_voices_are_still_listed(self):
        ids = [v['id'] for v in voz.installed()]
        self.assertIn('paulina', ids)
        if PIPER_READY:
            for key in ['es_MX-claude-high', 'es_MX-ald-medium', 'es_AR-daniela-high', 'es_ES-davefx-medium']:
                self.assertIn(key, ids)

    @unittest.skipUnless(PIPER_READY, 'Voz Piper no descargada')
    def test_piper_produces_speech_and_reports_engine(self):
        result = voz.synthesize('Hola, soy Nexo.', 'es_MX-claude-high')
        self.assertGreater(duration(result.audio), 0.5)
        self.assertEqual((result.engine, result.voice, result.fallback), ('Piper', 'Claude', False))

    def test_paulina_still_works(self):
        result = voz.synthesize('Hola, soy Nexo.', 'paulina')
        self.assertGreater(duration(result.audio), 0.5)
        self.assertEqual((result.engine, result.voice, result.fallback), ('macOS say', 'Paulina', False))

    def test_conversation_fallback_is_reported_not_silent(self):
        with mock.patch.object(voz, '_paulina', return_value=tiny_wav()), \
             mock.patch.object(voz, '_piper', side_effect=RuntimeError('modelo dañado')):
            result = voz.synthesize('Hola', 'es_MX-claude-high')
        info = result.info()
        self.assertEqual((info['solicitada'], info['motor'], info['voz'], info['respaldo']),
                         ('es_MX-claude-high', 'macOS say', 'Paulina', True))
        self.assertIn('modelo dañado', info['motivo'])
        with mock.patch.object(voz, '_paulina', return_value=tiny_wav()):
            self.assertTrue(voz.synthesize('Hola', '../../etc/passwd').fallback)

    def test_sample_mode_never_substitutes_a_failed_voice(self):
        with mock.patch.object(voz, '_paulina') as paulina, \
             mock.patch.object(voz, '_piper', side_effect=RuntimeError('modelo dañado')), \
             mock.patch.object(voz.QWEN, 'available', return_value=True), \
             mock.patch.object(voz.QWEN, 'synthesize', side_effect=voz.VoiceError('Qwen3-TTS falló')):
            for voice in ['es_MX-claude-high', 'qwen3-serena', 'qwen3-desconocida']:
                with self.assertRaises(voz.VoiceError):
                    voz.synthesize('Hola', voice, strict=True)
        paulina.assert_not_called()

    def test_qwen_voices_are_identified_without_claiming_native_spanish(self):
        with mock.patch.object(voz.QWEN, 'available', return_value=True):
            qwen = [v for v in voz.installed() if v['motor'] == 'Qwen3-TTS']
        self.assertEqual({v['id'] for v in qwen}, {'qwen3-' + k for k in voz.QWEN_SPEAKERS})
        for v in qwen:
            self.assertIn('Qwen3-TTS', v['nombre'])
            self.assertNotIn('latino', v['nombre'].lower())

    def test_selected_qwen_voice_matches_engine_used(self):
        reply = {'ok': True, 'seconds': 1.0, 'peak_gb': 2.0}
        with mock.patch.object(voz.QWEN, 'available', return_value=True), \
             mock.patch.object(voz.QWEN, 'synthesize', return_value=(tiny_wav(rate=24000), 2.1, reply)) as qwen:
            result = voz.synthesize('Hola, Anderson.', 'qwen3-ryan', strict=True)
        qwen.assert_called_once_with('Hola, Anderson.', 'ryan')
        self.assertEqual((result.engine, result.voice, result.load_seconds, result.fallback), ('Qwen3-TTS', 'Ryan', 2.1, False))


class SpeechTextTests(unittest.TestCase):
    def test_examples(self):
        cases = {
            '¡Hola, Anderson! 😊': '¡Hola, Anderson!',
            'Eso es **importante**.': 'Eso es importante.',
            'Puedes consultar [la guía](https://example.com).': 'Puedes consultar la guía.',
        }
        for text, expected in cases.items():
            self.assertEqual(voz.preparar_texto_para_voz(text), expected)

    def test_compound_emojis_leave_no_residue(self):
        compound = '👨‍👩‍👧‍👦 familia, 👍🏽 bien, 🇨🇴 Colombia, ❤️ amor, 1️⃣ uno, 🏳️‍🌈 bandera, ©️ marca, 🏴󠁧󠁢󠁳󠁣󠁴󠁿 Escocia'
        self.assertEqual(voz.preparar_texto_para_voz(compound), 'familia, bien, Colombia, amor, uno, bandera, marca, Escocia')

    def test_plain_text_accents_numbers_and_punctuation_are_kept(self):
        text = '¿Mañana a las 9:30 vienes? Sí, compré 3,5 kg de café a 12.000 pesos; ¡qué bien!… 5*3 = 15 y 20 °C.'
        self.assertEqual(voz.preparar_texto_para_voz(text), text)

    def test_only_emojis_or_symbols_is_not_pronounceable(self):
        for text in ['😊', '👍🏽🎉', ' ** ', '---', ':) ;-)', '']:
            self.assertEqual(voz.preparar_texto_para_voz(text), '', text)

    def test_markdown_structure_becomes_natural_pauses(self):
        text = ('# Resumen\n\n> Nota importante\n\n1. Instala `piper`\n2) Prueba la voz\n* Sin __formato__ ni ~~ruido~~\n\n'
                '---\n| Voz | Tiempo |\n|---|---|\n| Claude | 0,16 s |\n![diagrama](img.png) final')
        self.assertEqual(voz.preparar_texto_para_voz(text),
                         'Resumen.\nNota importante\nInstala piper.\nPrueba la voz.\nSin formato ni ruido.\n'
                         'Voz, Tiempo\nClaude, 0,16 s\ndiagrama final')

    def test_asterisks_keep_their_words(self):
        self.assertEqual(voz.preparar_texto_para_voz('*sonríe* Claro que *sí* puedes.'), 'sonríe Claro que sí puedes.')
        self.assertEqual(voz.preparar_texto_para_voz('variable_de_ejemplo y snake_case'), 'variable_de_ejemplo y snake_case')

    def test_emoticons_removed_but_times_and_urls_kept(self):
        self.assertEqual(voz.preparar_texto_para_voz('Nos vemos a las 10:30 :) en https://ejemplo.com <3'),
                         'Nos vemos a las 10:30 en https://ejemplo.com')

    def test_cleaning_happens_once_before_every_engine(self):
        with mock.patch.object(voz, '_paulina', return_value=tiny_wav()) as paulina:
            voz.synthesize('**Hola** 😊', 'paulina')
        paulina.assert_called_once_with('Hola')
        with mock.patch.object(voz, '_paulina') as paulina, mock.patch.object(voz, '_piper') as piper, \
             mock.patch.object(voz.QWEN, 'synthesize') as qwen:
            for voice in ['es_MX-claude-high', 'qwen3-serena', 'paulina']:
                self.assertIsNone(voz.synthesize('🎉🎉', voice).audio)
        paulina.assert_not_called()
        piper.assert_not_called()
        qwen.assert_not_called()


@unittest.skipUnless(voz.QWEN.available() and os.environ.get('NEXO_PROBAR_QWEN') == '1',
                     'Prueba real de Qwen3-TTS: NEXO_PROBAR_QWEN=1 (requiere .venv-tts y el modelo)')
class QwenRealTests(unittest.TestCase):
    def test_spanish_audio_reuses_process_and_is_understood(self):
        from math import gcd
        from scipy.signal import resample_poly
        from transcripcion import Transcriber, normalize
        worker = voz.QwenWorker(idle=0)
        try:
            with mock.patch.object(voz, 'QWEN', worker):
                first = voz.synthesize('Mañana compramos pan y café en el mercado.', 'qwen3-serena', strict=True)
                pid = worker.process.pid
                second = voz.synthesize('Gracias por escucharme hoy.', 'qwen3-serena', strict=True)
            self.assertEqual(worker.starts, 1)
            self.assertEqual(worker.process.pid, pid, 'La segunda síntesis reutiliza el mismo proceso')
            self.assertIsNotNone(first.load_seconds)
            self.assertIsNone(second.load_seconds)
            with wave.open(io.BytesIO(first.audio)) as wav:
                rate = wav.getframerate()
                samples = np.frombuffer(wav.readframes(wav.getnframes()), '<i2').astype(np.float32) / 32768
            self.assertGreater(len(samples) / rate, 1.0)
            g = gcd(rate, 16000)
            text = normalize(Transcriber().transcribe(resample_poly(samples, 16000 // g, rate // g).astype(np.float32)).text)
            for word in ['manana', 'pan', 'cafe', 'mercado']:
                self.assertIn(word, text, text)
        finally:
            worker.unload()
        self.assertFalse(worker.loaded)


if __name__ == '__main__':
    unittest.main()
