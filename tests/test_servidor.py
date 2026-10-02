import http.client
import tempfile
import threading
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

import servidor
from memoria import Memoria
from personalidad import CHARS_PER_TOKEN, MAX_TOKENS, TEMPERATURE
from transcripcion import build_transcript


def voice_turn(text, probs=None, logprob=-0.1):
    words = text.split()
    segment = {'text': text, 'avg_logprob': logprob, 'no_speech_prob': 0.01, 'compression_ratio': 1.1,
               'words': [{'word': w, 'probability': p} for w, p in zip(words, probs or [0.95] * len(words))]}
    return {'id': 'abc', 'transcript': build_transcript([segment]), 'incomplete': False,
            'timings': {'voz': 1.0, 'fin_de_turno': 0.7, 'transcripcion': 0.4}}


def answer(text):
    return {'choices': [{'message': {'content': text}}]}


class TalkTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.memory = Memoria(Path(self.directory.name) / 'test.sqlite3')
        patches = [mock.patch.object(servidor, 'memory', self.memory), mock.patch.object(servidor, 'voice', servidor.VoiceSession()),
                   mock.patch.object(servidor, 'loaded_context', return_value=4096)]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.addCleanup(self.directory.cleanup)

    def test_voice_turn_is_answered_once(self):
        servidor.voice.pending = {'abc': voice_turn('¿Qué hora es?')}
        spoken = []
        with mock.patch.object(servidor, 'request_json', return_value=answer('<think>x</think>Son las tres.')):
            status, data = servidor.talk({'turn': 'abc', 'voice': 'es_MX-claude-high'}, speak=lambda text, v: spoken.append(v) or 'QUJD')
            self.assertEqual((status, data['answer']), (200, 'Son las tres.'))
            self.assertEqual(spoken, ['es_MX-claude-high'])
            self.assertEqual(set(data['timings']), {'voz', 'fin_de_turno', 'transcripcion', 'generacion', 'sintesis_voz'})
            status, data = servidor.talk({'turn': 'abc'}, speak=lambda text, v: 'QUJD')
        self.assertEqual(status, 409)
        self.assertTrue(data['duplicate'])
        self.assertEqual(self.memory.history()['count'], 1)

    def test_low_confidence_asks_to_repeat_without_model_or_memory(self):
        servidor.voice.pending = {'abc': voice_turn('Ya ves por ahí.', [0.2, 0.3, 0.9, 0.3])}
        with mock.patch.object(servidor, 'request_json') as request:
            status, data = servidor.talk({'turn': 'abc'}, speak=lambda text, v: '')
        request.assert_not_called()
        self.assertIn('¿Puedes repetirlo', data['answer'])
        self.assertFalse(data['saved'])
        self.assertEqual(self.memory.history()['count'], 0)

    def test_doubtful_name_is_flagged_to_model_and_not_saved(self):
        servidor.voice.pending = {'abc': voice_turn('Me llamo Andersen Trujillo.', [0.9, 0.9, 0.2, 0.9])}
        with mock.patch.object(servidor, 'request_json', return_value=answer('¿Dijiste Andersen?')) as request:
            servidor.talk({'turn': 'abc'}, speak=lambda text, v: '')
        system = request.call_args[0][1]['messages'][0]['content']
        self.assertIn('Andersen', system.split('AVISO')[1])
        self.assertNotIn('nombre', self.memory.profile())

    def test_lm_studio_down_gives_clear_error(self):
        with mock.patch.object(servidor, 'request_json', side_effect=urllib.error.URLError('refused')):
            with self.assertRaises(servidor.LMStudioError) as caught:
                servidor.talk({'text': 'Hola'})
        self.assertEqual(caught.exception.status, 503)
        self.assertIn('LM Studio no responde', str(caught.exception))

    def test_missing_model_is_reported(self):
        session = servidor.VoiceSession(asr=mock.Mock(available=lambda: False))
        with self.assertRaises(servidor.ModelMissing):
            session.start()

    def test_stale_or_finished_session_rejects_audio(self):
        self.assertIsNone(servidor.voice.chunk('otro', b'\x00\x00' * 512))

    def test_payload_uses_central_personality_profile_and_budget(self):
        self.memory.save('Mi nombre es Anderson Trujillo y tengo 40 años.', 'Encantada.')
        self.memory.save('No soy emprendedor.', 'Anotado.')
        for i in range(30):
            self.memory.save(f'Tema largo {i} ' + 'x' * 1500, 'y' * 1100)
        with mock.patch.object(servidor, 'request_json', return_value=answer('Hola.')) as request:
            servidor.talk({'text': '¿Cómo me llamo?'}, speak=lambda text, v: '')
        payload = request.call_args[0][1]
        self.assertEqual((payload['temperature'], payload['max_tokens']), (TEMPERATURE, MAX_TOKENS))
        system = payload['messages'][0]['content']
        self.assertEqual(system.count('Tu nombre es Nexo. Eres una asistente'), 1)
        self.assertIn('Anderson Trujillo', system)
        self.assertIn('NO es emprendedor', system)
        total = sum(len(m['content']) for m in payload['messages'])
        self.assertLessEqual(total / CHARS_PER_TOKEN + MAX_TOKENS, 4096, 'Queda espacio para la respuesta')

    def test_screen_and_memory_keep_original_while_voice_gets_clean_text(self):
        original = '¡Hola! 😊 Dos ideas:\n- **Primera** parte\n- Segunda parte'
        spoken = []
        with mock.patch.object(servidor, 'request_json', return_value=answer(original)), \
             mock.patch.object(servidor.voz, '_piper', side_effect=RuntimeError), \
             mock.patch.object(servidor.voz, '_paulina', side_effect=lambda text: spoken.append(text) or b'WAV'):
            status, data = servidor.talk({'text': 'Dame ideas', 'voice': 'paulina'})
        self.assertEqual(data['answer'], original)
        self.assertEqual(self.memory.history()['turns'][-1]['answer'], original)
        self.assertEqual(spoken, ['¡Hola! Dos ideas:\nPrimera parte.\nSegunda parte.'])
        self.assertIn('audio', data)

    def test_emoji_only_answer_skips_synthesis_without_error(self):
        with mock.patch.object(servidor, 'request_json', return_value=answer('😊👍🏽')), \
             mock.patch.object(servidor.voz, '_piper') as piper, mock.patch.object(servidor.voz, '_paulina') as paulina:
            status, data = servidor.talk({'text': 'Hola'})
        piper.assert_not_called()
        paulina.assert_not_called()
        self.assertEqual(data['answer'], '😊👍🏽')
        self.assertNotIn('audio', data)
        self.assertNotIn('warning', data)

    def test_answer_cleanup_cuts_loops_but_keeps_original_text(self):
        loop = 'Hola 😄 **sí**.\n' + '¿O tal vez te refieres a otra cosa distinta?\n' * 5
        self.assertEqual(servidor.clean_answer(loop), 'Hola 😄 **sí**.\n¿O tal vez te refieres a otra cosa distinta?')


class OriginTests(unittest.TestCase):
    def test_127_redirects_to_single_localhost_origin(self):
        server = servidor.ThreadingHTTPServer(('127.0.0.1', 0), servidor.Handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with mock.patch.object(servidor, 'PORT', port):
                connection = http.client.HTTPConnection('127.0.0.1', port)
                connection.request('GET', '/', headers={'Host': f'127.0.0.1:{port}'})
                response = connection.getresponse()
                self.assertEqual(response.status, 302)
                self.assertEqual(response.getheader('Location'), f'http://localhost:{port}/')
        finally:
            server.shutdown()
            server.server_close()


if __name__ == '__main__':
    unittest.main()
