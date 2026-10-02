"""Batería de calidad conversacional; las pruebas de modelo real son optativas.

Para probar LM Studio: NEXO_PROBAR_CONVERSACION=1 NEXO_MEMORIA=/tmp/nexo-eval.sqlite3
python -m unittest discover -s tests -p test_calidad_conversacional.py -v
Las pruebas reales sustituyen esa base por SQLite temporal antes de guardar turnos.
"""
import json
import os
import re
import tempfile
import unittest
from pathlib import Path

from conversacion import ConversationState
from memoria import Memoria
from personalidad import REMINDER

REAL_MODEL = os.environ.get('NEXO_PROBAR_CONVERSACION') == '1'


def evaluate_reply(question, reply):
    """Señales para revisar respuestas; no modifica ni recorta el texto."""
    return {
        'words': len(re.findall(r"\b[\wáéíóúñü]+\b", reply, re.I)),
        'questions': reply.count('?'),
        'filler': bool(re.search(
            r'¿quieres|¿o prefieres|¿te gustaría|¿te interesa|estoy aquí para ayudarte|'
            r'¿en qué puedo ayudarte|¿qué quieres saber|anderson,|¿y tú\?|si necesitas algo más', reply, re.I)),
        'invented_activity': bool(re.search(
            r'\b(?:hoy\s+(?:fui|hice|estuve|pasé|revisé|respondí|trabajé|descansé)|'
            r'me tomé un café|me quedé en casa)\b', reply, re.I)),
        'question_is_simple': bool(re.search(r'¿(?:me escuchas|cuántos años tengo|cómo me llamo)', question, re.I)),
    }


def evaluate_dialogue(turns):
    """Resume exceso de preguntas, muletillas y temas viejos en turnos nuevos."""
    followup_questions = 0
    longest_question_streak = 0
    streak = 0
    unrelated_old_references = []
    old_topics = {'café': r'café|taza|bebida', 'nubes': r'nube'}
    for turn in turns:
        metrics = evaluate_reply(turn['question'], turn['answer'])
        streak = streak + 1 if metrics['questions'] else 0
        longest_question_streak = max(longest_question_streak, streak)
        if metrics['questions']:
            followup_questions += 1
        for topic, pattern in old_topics.items():
            if not re.search(pattern, turn['question'], re.I) and re.search(pattern, turn['answer'], re.I):
                unrelated_old_references.append({'turn': turn['number'], 'topic': topic})
    return {
        'turns': len(turns),
        'turns_with_questions': followup_questions,
        'longest_question_streak': longest_question_streak,
        'unrelated_old_references': unrelated_old_references,
        'filler_turns': sum(evaluate_reply(t['question'], t['answer'])['filler'] for t in turns),
    }


class ConversationalHeuristicTests(unittest.TestCase):
    def test_detects_long_simple_answer_and_extra_questions(self):
        result = evaluate_reply('¿Me escuchas?', 'Sí, te escucho. ¿Quieres que lo explique? ¿O prefieres otra cosa?')
        self.assertEqual(result['questions'], 2)
        self.assertTrue(result['filler'])
        self.assertTrue(result['question_is_simple'])

    def test_detects_repeated_old_topics_outside_their_turn(self):
        report = evaluate_dialogue([
            {'number': 1, 'question': 'Me tomaría un café.', 'answer': 'Un café suena bien.'},
            {'number': 2, 'question': 'Háblame del espacio.', 'answer': 'La taza de café puede esperar; hablemos de estrellas.'},
        ])
        self.assertEqual(report['unrelated_old_references'], [{'turn': 2, 'topic': 'café'}])

    def test_reminder_prioritizes_intent_proportionality_and_stopping(self):
        self.assertIn('responde eso primero', REMINDER)
        self.assertIn('detente', REMINDER)
        self.assertIn('Si pidió iniciativa', REMINDER)
        self.assertIn('No inventes experiencias personales', REMINDER)


@unittest.skipUnless(REAL_MODEL, 'Activa NEXO_PROBAR_CONVERSACION=1 para consultar LM Studio real.')
class RealModelConversationTests(unittest.TestCase):
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

        self.servidor = servidor
        self.original_memory = servidor.memory
        self.original_conversation = servidor.conversation
        self.original_temperature = servidor.TEMPERATURE
        self.addCleanup(setattr, servidor, 'memory', self.original_memory)
        self.addCleanup(setattr, servidor, 'conversation', self.original_conversation)
        self.addCleanup(setattr, servidor, 'TEMPERATURE', self.original_temperature)
        self.reset_chat('initial')

    def reset_chat(self, name):
        self.servidor.memory = Memoria(Path(self.directory.name) / f'{name}.sqlite3')
        with self.servidor.memory.connect() as db:
            db.executemany('INSERT INTO profile(key,value,source) VALUES (?,?,?)', [
                ('nombre', 'Anderson Trujillo', 'perfil temporal de prueba'),
                ('edad declarada', '40 años', 'perfil temporal de prueba'),
                ('profesión', 'ingeniero de sistemas', 'perfil temporal de prueba'),
            ])
        self.servidor.conversation = ConversationState()

    def ask(self, question, number=None):
        answer = self.servidor.reply(question)[0]
        turn = {'number': number or len(self.servidor.memory.history()['turns']),
                'question': question, 'answer': answer}
        print(f"\nENTRADA: {question}\nRESPUESTA REAL: {answer}")
        return turn

    def test_real_temperature_experiment(self):
        prompts = [
            ('simple', '¿Me escuchas?'),
            ('factual', '¿Cómo se forman las nubes?'),
            ('initiative', '¿Qué me quieres contar?'),
            ('personal', '¿Qué hiciste hoy?'),
        ]
        report = {}
        for temperature in (0.4, 0.5, 0.6):
            self.reset_chat(f'temperature-{temperature}')
            self.servidor.TEMPERATURE = temperature
            results = []
            for label, question in prompts:
                self.reset_chat(f'temperature-{temperature}-{label}')
                self.servidor.TEMPERATURE = temperature
                turn = self.ask(question)
                results.append({'prompt': label, **evaluate_reply(question, turn['answer'])})
            report[str(temperature)] = results
        print('\nTEMPERATURE_REPORT: ' + json.dumps(report, ensure_ascii=False))

    def test_real_everyday_greeting_is_brief_and_non_theatrical(self):
        self.reset_chat('everyday-greeting')
        self.servidor.TEMPERATURE = 0.6
        self.servidor.memory.save(
            'Me llamo Anderson Trujillo, tengo 40 años y soy ingeniero de sistemas.',
            'Anotado: Anderson Trujillo, 40 años, ingeniero de sistemas.')
        self.servidor.memory.save('No soy emprendedor.', 'Entendido, no eres emprendedor.')
        turn = self.ask('Hola, ¿cómo estás?')
        self.assertLessEqual(evaluate_reply(turn['question'], turn['answer'])['words'], 10)
        self.assertEqual(evaluate_reply(turn['question'], turn['answer'])['questions'], 0)
        self.assertNotRegex(turn['answer'], r'(?i)anderson|silencio|quédate|cariño|😊|\*|\(')

    def test_real_conversation_over_twenty_turns(self):
        self.reset_chat('twenty-turn-dialogue')
        self.servidor.TEMPERATURE = 0.6
        questions = [
            'Hola, ¿me escuchas?',
            '¿Cuántos días tiene un año?',
            '¿Cómo se forman las nubes?',
            'Háblame un poco más de las nubes.',
            '¿Cuántos años tengo?',
            'Háblame del espacio.',
            'Me tomaría un café.',
            'No vuelvas a insistir con el café.',
            'Háblame del espacio.',
            '¿Qué hiciste hoy?',
            '¿Cuántos años tienes tú?',
            'Cuéntame algo interesante.',
            'Cuéntame un chiste.',
            'Estamos hablando de departamentos de Colombia.',
            '¿Cuál es la capital del Baje del Cauca?',
            'Háblame de Java.',
            'Explícame bien qué es una API.',
            'No soy emprendedor.',
            '¿Cuál es mi profesión?',
            'manglar tornillo debajo canción planeta',
            '¿Qué opinas de aprender programación después de los 40?',
            'Chao, hablamos luego.',
        ]
        turns = [self.ask(question, index) for index, question in enumerate(questions, 1)]
        report = evaluate_dialogue(turns)
        print('\nDIALOGUE_REPORT: ' + json.dumps(report, ensure_ascii=False))

        self.assertEqual(len(turns), 22)
        self.assertLessEqual(evaluate_reply(turns[0]['question'], turns[0]['answer'])['words'], 25)
        self.assertLessEqual(evaluate_reply(turns[1]['question'], turns[1]['answer'])['words'], 25)
        self.assertLessEqual(evaluate_reply(turns[2]['question'], turns[2]['answer'])['words'], 90)
        self.assertEqual(evaluate_reply(turns[2]['question'], turns[2]['answer'])['questions'], 0)
        self.assertGreater(evaluate_reply(turns[3]['question'], turns[3]['answer'])['words'],
                   evaluate_reply(turns[2]['question'], turns[2]['answer'])['words'])
        self.assertNotRegex(turns[4]['answer'], r'(?i)nube|café|taza')
        self.assertNotRegex(turns[8]['answer'], r'(?i)café|taza|bebida')
        self.assertFalse(evaluate_reply(turns[9]['question'], turns[9]['answer'])['invented_activity'])
        self.assertNotRegex(turns[10]['answer'], r'(?i)\b40\b')
        self.assertNotRegex(turns[11]['answer'], r'(?i)nada específico|no tengo nada')
        self.assertEqual(evaluate_reply(turns[11]['question'], turns[11]['answer'])['questions'], 0)
        self.assertNotRegex(turns[12]['answer'], r'\*|Anderson|te lo dejé|en serio')
        self.assertEqual(evaluate_reply(turns[14]['question'], turns[14]['answer'])['questions'], 0)
        self.assertLessEqual(evaluate_reply(turns[15]['question'], turns[15]['answer'])['words'], 35)
        self.assertEqual(evaluate_reply(turns[15]['question'], turns[15]['answer'])['questions'], 1)
        self.assertLessEqual(evaluate_reply(turns[19]['question'], turns[19]['answer'])['questions'], 1)
        self.assertRegex(turns[14]['answer'], r'(?i)valle del cauca|cali')
        self.assertNotRegex(turns[17]['answer'], r'(?i)\bemprendedor\b')
        self.assertRegex(turns[18]['answer'], r'(?i)ingeniero de sistemas')
        self.assertEqual(report['unrelated_old_references'], [])
        self.assertLessEqual(report['longest_question_streak'], 1)


if __name__ == '__main__':
    unittest.main()