import unittest

from transcripcion import build_transcript, clean_text, hint_prompt, normalize


def segment(text, probs=None, logprob=-0.1, no_speech=0.01, compression=1.1):
    words = text.split()
    probs = probs or [0.95] * len(words)
    return {'text': ' ' + text, 'avg_logprob': logprob, 'no_speech_prob': no_speech, 'compression_ratio': compression,
            'words': [{'word': ' ' + w, 'probability': p} for w, p in zip(words, probs)]}


class CleanTextTests(unittest.TestCase):
    def test_punctuation_without_changing_words(self):
        cases = {
            ' hola  nexo ': 'Hola nexo.',
            'oye, cómo estás?': 'Oye, ¿cómo estás?',
            'qué bien!': '¡Qué bien!',
            'me llamo anderson trujillo . tengo 40 años': 'Me llamo anderson trujillo. Tengo 40 años.',
            '¿Vienes? sí, claro.': '¿Vienes? Sí, claro.',
            'quiero llamar a mi hermana y... contarle algo': 'Quiero llamar a mi hermana y... contarle algo.',
        }
        for raw, expected in cases.items():
            self.assertEqual(clean_text(raw), expected)
            self.assertEqual(normalize(clean_text(raw)), normalize(raw), 'Nunca añade ni quita palabras')


class TranscriptTests(unittest.TestCase):
    def test_empty_and_noise_hallucinations_are_discarded(self):
        self.assertEqual(build_transcript([]).discarded, 'sin palabras')
        for text in ['Subtítulos realizados por la comunidad de Amara.org', '¡Gracias por ver el video!', '[Música]', 'Suscríbete al canal.']:
            self.assertTrue(build_transcript([segment(text)]).discarded, text)
        self.assertFalse(build_transcript([segment('Gracias por ayudarme con la tarea.')]).discarded)
        self.assertTrue(build_transcript([segment('Hola', logprob=-1.5, no_speech=0.9)]).discarded)

    def test_prompt_echo_is_discarded(self):
        prompt = hint_prompt(['Anderson Trujillo'])
        self.assertEqual(build_transcript([segment('Conversación en español. Nombres: Anderson Trujillo.')], prompt).discarded,
                         'eco de las pistas de nombres')

    def test_names_are_only_hints_never_inserted(self):
        result = build_transcript([segment('Hoy hace buen tiempo.')], hint_prompt(['Anderson Trujillo']))
        self.assertNotIn('Anderson', result.text)

    def test_confidence_levels_and_doubtful_words(self):
        clear = build_transcript([segment('Me llamo Anderson Trujillo.')])
        self.assertEqual((clear.confidence, clear.doubtful), ('alta', []))
        partial = build_transcript([segment('Me llamo Andersen Trujillo.', [0.9, 0.9, 0.2, 0.8])])
        self.assertEqual((partial.confidence, partial.doubtful), ('media', ['Andersen']))
        poor = build_transcript([segment('Ya ves por ahí.', [0.2, 0.3, 0.9, 0.3])])
        self.assertEqual(poor.confidence, 'baja')
        loop = build_transcript([segment('sí sí sí sí sí sí sí sí', compression=3.0)])
        self.assertEqual(loop.confidence, 'baja')

    def test_hint_prompt_is_sanitized_and_limited(self):
        self.assertEqual(hint_prompt([]), '')
        self.assertEqual(hint_prompt(['Nexo', 'nexo', 'Ana<script>']), 'Conversación en español. Nombres: Nexo, Anascript.')


if __name__ == '__main__':
    unittest.main()
