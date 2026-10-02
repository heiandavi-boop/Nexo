"""Preferencias conversacionales, estado temporal, fecha local y transcripciones imperfectas.

No dependen de frases exactas del modelo: comprueban qué se guarda y qué recibe el modelo.
"""
import sqlite3
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import servidor
from conversacion import AVOID_PERSISTENCE, REJECTED_TOPICS, ConversationState, analyze, local_context
from memoria import Memoria
from transcripcion import build_transcript


def answer(text):
    return {'choices': [{'message': {'content': text}}]}


class DetectionTests(unittest.TestCase):
    def test_a_general_preference_is_detected(self):
        for text in ['A partir de ahora no insistas tanto en el mismo tema.',
                     'Cuando te hable de un tema no quiero que persistas en lo mismo.']:
            found = analyze(text)
            self.assertEqual(found['preferences'], {AVOID_PERSISTENCE: True}, text)
            self.assertEqual(found['abandon'], [], 'Una preferencia general no es un tema')

    def test_b_explicit_topic_rejection(self):
        self.assertEqual(analyze('No vuelvas a hablar del café.')['reject'], ['café'])
        found = analyze('Ya no quiero que menciones más el café y la taza.')
        self.assertEqual(found['abandon'], ['café', 'taza'])

    def test_e_change_topic(self):
        for text in ['Cambiemos de tema.', 'Hablemos de otra cosa', 'Dejemos ese tema.', 'Pasemos a otra cosa.', 'No insistas con eso.']:
            self.assertTrue(analyze(text)['change_topic'], text)

    def test_questions_examples_and_hypotheses_are_not_instructions(self):
        for text in ['¿No vuelvas a hablar del café?', 'Imagina que no quiero hablar del clima.',
                     'Por ejemplo: no menciones más el fútbol.', 'Creo que un café estaría bien.', '«No hables del café», dijo él.']:
            found = analyze(text)
            self.assertEqual((found['abandon'], found['preferences'], found['change_topic']), ([], {}, False), text)

    def test_doubtful_words_block_instructions(self):
        self.assertEqual(analyze('No vuelvas a hablar del café.', ['café'])['abandon'], [])
        self.assertEqual(analyze('A partir de ahora no insistas en el mismo tema.', ['insistas'])['preferences'], {})


class StateTests(unittest.TestCase):
    def test_abandoned_topic_and_explicit_reintroduction(self):
        state = ConversationState()
        state.observe('Creo que un café estaría bien.', last_turn_id=1)
        state.observe('No quiero que sigas insistiendo con el café y la taza. Cambiemos de tema.', last_turn_id=2)
        self.assertEqual(state.abandoned, ['café', 'taza'])
        self.assertEqual(state.current_topic, [])
        self.assertEqual(state.boundary, 4, 'El diálogo anterior al cambio, y el propio pedido, dejan de enviarse como recientes')
        section = state.prompt_section()
        self.assertIn('Temas abandonados:\n- café\n- taza', section)
        self.assertIn('acaba de pedir cambiar de tema', section)
        state.observe('¿Qué tema quieres proponer?', last_turn_id=3)
        self.assertIn('café', state.abandoned)
        state.observe('¿Qué tipos de café produce Colombia?', last_turn_id=4)
        self.assertNotIn('café', state.abandoned, 'El usuario lo reintrodujo')
        self.assertIn('taza', state.abandoned)

    def test_no_state_section_when_nothing_is_relevant(self):
        state = ConversationState()
        state.observe('Hola, ¿cómo estás?')
        self.assertEqual(state.prompt_section(), '')


class LocalTimeTests(unittest.TestCase):
    def test_local_context_comes_from_the_clock(self):
        text = local_context(datetime(2026, 10, 1, 22, 45))
        self.assertIn('Fecha: jueves 1 de octubre de 2026', text)
        self.assertIn('Hora: 22:45', text)
        self.assertRegex(text, r'Zona horaria: \S+')
        now = datetime.now()
        self.assertIn(str(now.year), local_context())


class MemoryPreferenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'test.sqlite3'

    def test_a_preference_persists_and_can_be_corrected(self):
        memory = Memoria(self.path)
        memory.save('A partir de ahora no insistas tanto en el mismo tema.', 'De acuerdo.')
        self.assertEqual(Memoria(self.path).preferences(), {AVOID_PERSISTENCE: True})
        memory.save('Ya puedes volver a profundizar en los temas.', 'Bien.')
        self.assertEqual(Memoria(self.path).preferences(), {AVOID_PERSISTENCE: False})
        self.assertNotIn(AVOID_PERSISTENCE, memory.profile(), 'Las preferencias no se mezclan con el perfil factual')

    def test_b_d_rejected_topic_persists_until_user_reopens_it(self):
        memory = Memoria(self.path)
        memory.save('No vuelvas a hablar del café.', 'Vale.')
        self.assertEqual(Memoria(self.path).preferences()[REJECTED_TOPICS], ['café'])
        self.assertNotIn(REJECTED_TOPICS, memory.preferences('¿Cuál es el mejor café colombiano?'))
        memory.save('¿Cuál es el mejor café colombiano?', 'Depende de la región.')
        self.assertNotIn(REJECTED_TOPICS, Memoria(self.path).preferences())

    def test_f_doubtful_instruction_is_not_stored(self):
        memory = Memoria(self.path)
        memory.save('No vuelvas a hablar del café.', 'Vale.', doubtful=['café'])
        memory.save('A partir de ahora no insistas en el mismo tema.', 'Vale.', doubtful=['insistas'])
        self.assertEqual(memory.preferences(), {})

    def test_g_old_database_still_readable(self):
        with sqlite3.connect(str(self.path)) as db:
            db.execute('CREATE TABLE turns (id INTEGER PRIMARY KEY, question TEXT NOT NULL, answer TEXT NOT NULL, created TEXT DEFAULT CURRENT_TIMESTAMP)')
            db.execute("CREATE VIRTUAL TABLE search USING fts5(question, answer, tokenize='unicode61 remove_diacritics 2')")
            db.execute('CREATE TABLE profile (key TEXT PRIMARY KEY, value TEXT NOT NULL, source TEXT NOT NULL)')
            db.execute('CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
            db.execute("INSERT INTO metadata VALUES ('profile_v1','1')")
            db.execute("INSERT INTO profile VALUES ('nombre','Anderson Trujillo','Mi nombre es Anderson Trujillo.')")
            db.execute("INSERT INTO turns(question,answer) VALUES ('Mi mascota se llama Mandarina.','Bonito nombre.')")
            db.execute("INSERT INTO search(rowid,question,answer) VALUES (1,'Mi mascota se llama Mandarina.','Bonito nombre.')")
        memory = Memoria(self.path)
        self.assertEqual(memory.profile()['nombre'], 'Anderson Trujillo')
        self.assertEqual(memory.history()['count'], 1)
        self.assertTrue(any('Mandarina' in m['content'] for m in memory.context('¿Cómo se llama mi mascota?')))
        self.assertEqual(memory.preferences(), {})

    def test_h_clear_removes_preferences(self):
        memory = Memoria(self.path)
        memory.save('No vuelvas a hablar del café.', 'Vale.')
        memory.save('A partir de ahora no insistas en el mismo tema.', 'Vale.')
        memory.clear()
        self.assertEqual(Memoria(self.path).preferences(), {})

    def test_k_doubtful_personal_name_is_not_stored(self):
        memory = Memoria(self.path)
        memory.save('Me llamo Andersen Trujillo.', '¿Puedes confirmar tu nombre?', doubtful=['Andersen'])
        self.assertNotIn('nombre', memory.profile())


class PromptTests(unittest.TestCase):
    """Lo que recibe el modelo, sin llamar a LM Studio."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.memory = Memoria(Path(self.directory.name) / 'test.sqlite3')
        for patch in [mock.patch.object(servidor, 'memory', self.memory), mock.patch.object(servidor, 'conversation', ConversationState()),
                      mock.patch.object(servidor, 'loaded_context', return_value=8192)]:
            patch.start()
            self.addCleanup(patch.stop)

    def say(self, text, doubtful=(), reply_text='Respuesta.'):
        with mock.patch.object(servidor, 'request_json', return_value=answer(reply_text)) as request:
            servidor.reply(text, list(doubtful))
        return request.call_args[0][1]['messages']

    @staticmethod
    def full(messages):
        """Todo lo que recibe el modelo como instrucciones: sistema fijo y contexto del turno."""
        return messages[0]['content'] + '\n' + messages[-1]['content']

    def test_static_instructions_are_identical_between_turns(self):
        first = self.say('Hola.')[0]['content']
        second = self.say('No vuelvas a hablar del café.')[0]['content']
        self.assertEqual(first, second, 'La parte fija no cambia: LM Studio puede reutilizarla')

    def test_c_abandoned_topic_is_explicit_and_old_dialogue_is_not_resent(self):
        self.say('Hola, tengo frío aquí en Bogotá.', reply_text='En Bogotá un café caliente ayuda.')
        self.say('Creo que un café estaría bien.', reply_text='Una taza de café siempre reconforta.')
        self.say('Cuéntame algo.', reply_text='Mientras tomas tu taza de café, te cuento algo.')
        self.say('No quiero que sigas insistiendo con el café y la taza. Cambiemos de tema.', reply_text='Hablemos de música.')
        messages = self.say('¿Qué tema quieres proponer?')
        system, dialogue = self.full(messages), messages[1:-1]
        self.assertIn('Temas abandonados:\n- café\n- taza', system)
        self.assertIn('No reintroduzcas temas abandonados', system)
        self.assertFalse(any('café' in m['content'] for m in dialogue), 'El diálogo con café no se reenvía como reciente')
        self.assertEqual(self.memory.history()['count'], 5, 'El historial no se borra')

    def test_d_user_can_reopen_topic(self):
        self.say('No vuelvas a hablar del café.')
        messages = self.say('¿Qué tipos de café produce Colombia?')
        self.assertNotIn('Temas abandonados', self.full(messages))
        self.assertTrue(messages[-1]['content'].endswith('¿Qué tipos de café produce Colombia?'))

    def test_preference_reaches_the_prompt(self):
        messages = self.say('A partir de ahora no insistas tanto en el mismo tema.')
        self.assertIn('Evitar insistir repetidamente en un mismo tema.', self.full(messages))

    def test_local_date_and_time_are_sent(self):
        system = self.full(self.say('¿Qué día es hoy?'))
        self.assertIn('CONTEXTO LOCAL ACTUAL', system)
        self.assertIn(str(datetime.now().year), system)

    def test_i_imperfect_transcription_keeps_context_and_inference_rules(self):
        self.say('¿Cuántos departamentos tiene Colombia?', reply_text='Colombia tiene 32 departamentos.')
        messages = self.say('¿Cuál es la capital del Baje del Cauca?')
        self.assertIn('Colombia tiene 32 departamentos.', [m['content'] for m in messages])
        self.assertIn('Creo que hablas de', self.full(messages))
        self.assertTrue(messages[-1]['content'].endswith('¿Cuál es la capital del Baje del Cauca?'), 'La transcripción no se reescribe')

    def test_j_l_doubtful_word_guides_model_but_is_never_stored(self):
        messages = self.say('¿Cuál es la capital del Baje del Cauca?', doubtful=['Baje'])
        self.assertIn('AVISO', self.full(messages))
        self.assertIn('si hay varias opciones, pregunta', self.full(messages))
        turn = self.memory.history()['turns'][-1]
        self.assertEqual(turn['question'], '¿Cuál es la capital del Baje del Cauca?')
        self.assertEqual((self.memory.profile(), self.memory.preferences()), ({}, {}))

    def test_very_doubtful_voice_turn_asks_to_repeat_without_model(self):
        words = 'él que bajo mesa planeta'.split()
        segment = {'text': ' '.join(words), 'avg_logprob': -1.3, 'no_speech_prob': 0.1, 'compression_ratio': 1.1,
                   'words': [{'word': w, 'probability': 0.2} for w in words]}
        servidor.voice.pending = {'t': {'id': 't', 'transcript': build_transcript([segment]), 'incomplete': False, 'timings': {}}}
        with mock.patch.object(servidor, 'request_json') as request:
            status, data = servidor.talk({'turn': 't'}, speak=lambda text, v: (None, {}))
        request.assert_not_called()
        self.assertIn('repetirlo', data['answer'])

    def test_profile_questions_still_use_confirmed_data(self):
        self.memory.save('Mi nombre es Anderson Trujillo, soy ingeniero de sistemas y tengo 40 años.', 'Encantada.')
        self.memory.save('No soy emprendedor.', 'Anotado.')
        system = self.full(self.say('¿Cómo me llamo?'))
        for expected in ['Anderson Trujillo', '40 años', 'ingeniero de sistemas', 'NO es emprendedor']:
            self.assertIn(expected, system)


if __name__ == '__main__':
    unittest.main()
