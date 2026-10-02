import io
import unittest
import wave
from unittest import mock

import voz


def duration(data):
    with wave.open(io.BytesIO(data)) as wav:
        return wav.getnframes() / wav.getframerate()


class VoiceTests(unittest.TestCase):
    def test_paulina_is_always_available(self):
        self.assertIn('paulina', [v['id'] for v in voz.installed()])

    @unittest.skipUnless((voz.VOICES_DIR / 'es_MX-claude-high.onnx').exists(), 'Voz Piper no descargada')
    def test_piper_produces_speech(self):
        self.assertGreater(duration(voz.synthesize('Hola, soy Nexo.', 'es_MX-claude-high')), 0.5)

    def test_unknown_or_broken_voice_falls_back_to_paulina(self):
        with mock.patch.object(voz, '_paulina', return_value=b'WAV') as paulina:
            self.assertEqual(voz.synthesize('Hola', '../../etc/passwd'), b'WAV')
            with mock.patch.object(voz, '_piper', side_effect=RuntimeError('roto')):
                self.assertEqual(voz.synthesize('Hola', 'es_MX-claude-high'), b'WAV')
        self.assertEqual(paulina.call_count, 2)


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
        with mock.patch.object(voz, '_paulina', return_value=b'WAV') as paulina:
            voz.synthesize('**Hola** 😊', 'paulina')
        paulina.assert_called_once_with('Hola')
        with mock.patch.object(voz, '_paulina') as paulina, mock.patch.object(voz, '_piper') as piper:
            self.assertIsNone(voz.synthesize('🎉🎉', 'es_MX-claude-high'))
        paulina.assert_not_called()
        piper.assert_not_called()


if __name__ == '__main__':
    unittest.main()
