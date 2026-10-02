"""Respuesta en streaming: la voz empieza con la primera frase mientras el modelo sigue escribiendo."""
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import servidor
from conversacion import ConversationState
from memoria import Memoria


def chunks(*parts):
    def fake(path, payload, timeout=120):
        assert payload['stream'] is True
        for part in parts:
            yield {'choices': [{'delta': {'content': part}}]}
    return fake


class SplitTests(unittest.TestCase):
    def test_first_sentence_is_released_early_and_rest_grouped(self):
        ready, rest = servidor.split_ready('Claro que sí. Te cuento algo sobre los ríos de Colombia y su', first=True)
        self.assertEqual((ready, rest), (['Claro que sí.'], 'Te cuento algo sobre los ríos de Colombia y su'))

    def test_decimals_and_short_fragments_are_not_cut(self):
        ready, rest = servidor.split_ready('Mide 3.5 metros y', first=True)
        self.assertEqual(ready, [])
        self.assertEqual(servidor.split_ready(rest, final=True), (['Mide 3.5 metros y'], ''))

    def test_newlines_separate_list_items(self):
        ready, _ = servidor.split_ready('Dos ideas:\n- Primera\n- Segunda\n', first=True)
        self.assertEqual(ready, ['Dos ideas:', '- Primera', '- Segunda'])

    def test_long_sentence_without_period_is_cut_at_a_comma_not_mid_phrase(self):
        text = ('Las nubes se forman cuando el aire húmedo asciende, se enfría al ganar altura, '
                'y el vapor se condensa en gotitas alrededor de partículas de polvo que')
        ready, rest = servidor.split_ready(text, first=True)
        self.assertEqual(ready, ['Las nubes se forman cuando el aire húmedo asciende, se enfría al ganar altura,'])
        self.assertTrue(rest.startswith('y el vapor'))

    def test_semicolon_ends_a_fragment(self):
        ready, _ = servidor.split_ready('Primero sube el aire; luego se enfría y', first=True)
        self.assertEqual(ready, ['Primero sube el aire;'])


class StreamTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.memory = Memoria(Path(self.directory.name) / 'test.sqlite3')
        for patch in [mock.patch.object(servidor, 'memory', self.memory), mock.patch.object(servidor, 'conversation', ConversationState()),
                      mock.patch.object(servidor, 'loaded_context', return_value=8192), mock.patch.object(servidor, 'voice', servidor.VoiceSession()),
                      mock.patch.object(servidor, 'warm_llm')]:
            patch.start()
            self.addCleanup(patch.stop)

    def run_stream(self, *parts, payload=None):
        events, spoken = [], []
        with mock.patch.object(servidor, 'stream_json', chunks(*parts)):
            servidor.talk_stream(payload or {'text': 'Cuéntame algo.'}, events.append,
                                 speak=lambda text, voice: (spoken.append(text), ('QUJD', {'motor': 'Piper'}))[1])
        return events, spoken

    def test_events_order_audio_per_sentence_and_memory(self):
        events, spoken = self.run_stream('Los ríos ', 'de Colombia son muchos. ', 'El Magdalena es el más largo ',
                                         'del país y cruza varias regiones importantes. Fin.')
        self.assertEqual([e['type'] for e in events][0], 'question')
        self.assertEqual(events[-1]['type'], 'done')
        self.assertEqual(spoken, ['Los ríos de Colombia son muchos.',
                                  'El Magdalena es el más largo del país y cruza varias regiones importantes.', 'Fin.'])
        self.assertEqual(sum(e['type'] == 'audio' for e in events), 3)
        self.assertTrue({'first_token', 'first_sentence', 'first_audio', 'generacion', 'sintesis_voz'} <= set(events[-1]['timings']))
        self.assertEqual(self.memory.history()['turns'][-1]['answer'],
                         'Los ríos de Colombia son muchos. El Magdalena es el más largo del país y cruza varias regiones importantes. Fin.')

    def test_repetition_loop_is_cut_before_being_spoken(self):
        line = '¿O tal vez te refieres a otra cosa distinta?\n'
        events, spoken = self.run_stream('Hola, ¿qué tal?\n', line, line, line, line)
        self.assertEqual(events[-1]['answer'], 'Hola, ¿qué tal?\n¿O tal vez te refieres a otra cosa distinta?')
        self.assertEqual(spoken.count('¿O tal vez te refieres a otra cosa distinta?'), 1)

    def test_lm_studio_down_sends_error_event(self):
        def down(path, payload, timeout=120):
            raise ConnectionRefusedError()
            yield
        events = []
        with mock.patch.object(servidor, 'stream_json', down):
            servidor.talk_stream({'text': 'Hola'}, events.append, speak=lambda t, v: (None, {}))
        self.assertEqual(events[-1]['type'], 'error')
        self.assertIn('LM Studio no responde', events[-1]['error'])

    def test_duplicate_turn_is_rejected(self):
        events = []
        servidor.talk_stream({'turn': 'no-existe'}, events.append)
        self.assertEqual((events[0]['type'], events[0]['duplicate']), ('error', True))

    def test_error_mid_stream_saves_nothing_and_reports(self):
        def broken(path, payload, timeout=120):
            yield {'choices': [{'delta': {'content': 'Empiezo a responder bien. '}}]}
            raise ConnectionResetError()
        events = []
        with mock.patch.object(servidor, 'stream_json', broken):
            servidor.talk_stream({'text': 'Hola'}, events.append, speak=lambda t, v: ('QUJD', {}))
        self.assertEqual(events[-1]['type'], 'error')
        self.assertEqual(self.memory.history()['count'], 0, 'No se guarda una respuesta truncada')

    def test_empty_answer_is_an_error_not_a_memory(self):
        events, spoken = self.run_stream('', '   ')
        self.assertEqual(events[-1]['type'], 'error')
        self.assertEqual((spoken, self.memory.history()['count']), ([], 0))

    def test_cancel_when_page_disconnects_stops_and_saves_nothing(self):
        produced = []

        def endless(path, payload, timeout=120):
            for i in range(200):
                produced.append(i)
                time.sleep(0.005)  # como un modelo real, que tarda en escribir cada fragmento
                yield {'choices': [{'delta': {'content': f'Frase número {i} de una respuesta muy larga. '}}]}
        events = []

        def emit(event):
            events.append(event)
            return event['type'] != 'audio'  # la página cierra la conexión al recibir audio
        with mock.patch.object(servidor, 'stream_json', endless):
            servidor.talk_stream({'text': 'Hola'}, emit, speak=lambda t, v: ('QUJD', {}))
        self.assertLess(len(produced), 200, 'La generación se detiene')
        self.assertNotEqual(events[-1]['type'], 'done')
        self.assertEqual(self.memory.history()['count'], 0)

    def test_audio_fragments_keep_order_and_are_not_duplicated(self):
        parts = [f'Esta es la frase número {i} de la respuesta completa. ' for i in range(8)]
        events, spoken = self.run_stream(*parts)
        audio = [e['text'] for e in events if e['type'] == 'audio']
        self.assertEqual(audio, spoken)
        self.assertEqual(' '.join(audio), ''.join(parts).strip())
        self.assertEqual(len(audio), len(set(audio)))

    def test_tts_failure_keeps_text_and_warns(self):
        events = []

        def failing(text, voice):
            raise servidor.voz.VoiceError('roto')
        with mock.patch.object(servidor, 'stream_json', chunks('Una respuesta completa sin voz.')):
            servidor.talk_stream({'text': 'Hola'}, events.append, speak=failing)
        self.assertEqual(events[-1]['answer'], 'Una respuesta completa sin voz.')
        self.assertIn('voz', events[-1]['warning'])


class WindowTests(unittest.TestCase):
    def test_dialogue_window_only_grows_then_restarts(self):
        state = ConversationState()
        starts = [state.window_start(last) for last in range(4, 12)]
        self.assertEqual(starts[:3], [1, 1, 1], 'Crece sin mover el inicio (prefijo estable para LM Studio)')
        self.assertTrue(all(last - start + 1 <= 6 for last, start in zip(range(4, 12), starts)))
        state.boundary = 20
        self.assertEqual(state.window_start(25), 20, 'Tras un cambio de tema empieza en el límite')


if __name__ == '__main__':
    unittest.main()
