import sqlite3
import tempfile
import unittest
from pathlib import Path
from memoria import Memoria


class MemoryTests(unittest.TestCase):
    def test_corrected_identity_persists_beyond_recent_context(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.sqlite3'
            memory = Memoria(path)
            memory.save('Mi nombre es Andrés Entrepreneurio, soy ingenieros sistemas y tengo 40 años.', 'Eres emprendedor.')
            memory.save('No, mi nombre es Anderson Trujillo.', 'Soy Anderson, emprendedor.')
            memory.save('Yo no soy emprendedor.', 'Entendido.')
            for i in range(12):
                memory.save(f'Tema sin relación {i}', 'No sé tu profesión.')
            memory = Memoria(path)
            facts = memory.profile()
            self.assertEqual(facts['nombre'], 'Anderson Trujillo')
            self.assertEqual(facts['edad declarada'], '40 años')
            self.assertEqual(facts['profesión'], 'ingeniero de sistemas')
            self.assertIn('NO', facts['emprendimiento'])
            self.assertEqual(memory.profile('Tengo 41 años.')['edad declarada'], '41 años')
            self.assertEqual(memory.profile()['edad declarada'], '40 años')
            memory.clear()
            self.assertEqual(memory.profile(), {})

    def test_questions_and_examples_are_not_facts(self):
        for text in ['¿Tengo 50 años?', 'Imagina que tengo 20 años.', 'Mi hermano dijo que tengo 30 años.', 'No tengo 50 años.']:
            self.assertEqual(Memoria.extract_facts(text), {}, text)

    def test_persistence_retrieval_clear(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.sqlite3'
            memory = Memoria(path)
            memory.save('Mi mascota se llama Mandarina.', 'Lo tendré presente.')
            for i in range(9):
                memory.save(f'Calcula {i} más dos.', str(i+2))
            reloaded = Memoria(path)
            self.assertEqual(reloaded.history()['count'], 10)
            context = reloaded.context('¿Cómo se llama mi mascota?')
            self.assertTrue(any('Mandarina' in item['content'] for item in context))
            self.assertLessEqual(sum(len(item['content']) for item in context), 7000)
            reloaded.clear()
            self.assertEqual(Memoria(path).history()['count'], 0)
            self.assertEqual(reloaded.context('mascota'), [])

    def test_literal_quotes_and_unicode(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = Memoria(Path(directory) / 'test.sqlite3')
            memory.save('Prefiero café sin azúcar.', 'Entendido.')
            self.assertTrue(memory.context('"café" OR (azúcar):*'))

    def test_doubtful_voice_words_do_not_become_confirmed_facts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.sqlite3'
            memory = Memoria(path)
            memory.save('Mi nombre es Anderson Trujillo.', 'Encantado.')
            question = 'Me llamo Andersen Trujillo y tengo 40 años.'
            self.assertEqual(Memoria.unconfirmed(question, ['Andersen']), {'nombre': 'Andersen Trujillo'})
            self.assertEqual(memory.profile(question, ['Andersen'])['nombre'], 'Anderson Trujillo')
            memory.save(question, '¿Puedes confirmar tu nombre?', ['Andersen'])
            facts = Memoria(path).profile()
            self.assertEqual(facts['nombre'], 'Anderson Trujillo')
            self.assertEqual(facts['edad declarada'], '40 años', 'Los datos no dudosos sí se guardan')
            self.assertEqual(Memoria(path).history()['count'], 2, 'La conversación se conserva')
            self.assertEqual(Memoria(path).name_hints(), ['Anderson Trujillo'])

    def test_existing_database_is_preserved_on_upgrade(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'old.sqlite3'
            with sqlite3.connect(str(path)) as db:
                db.execute('CREATE TABLE turns (id INTEGER PRIMARY KEY, question TEXT NOT NULL, answer TEXT NOT NULL, created TEXT DEFAULT CURRENT_TIMESTAMP)')
                db.execute("CREATE VIRTUAL TABLE search USING fts5(question, answer, tokenize='unicode61 remove_diacritics 2')")
                db.execute("INSERT INTO turns(question,answer) VALUES ('Me llamo Lucía.','Hola, Lucía.')")
                db.execute("INSERT INTO search(rowid,question,answer) VALUES (1,'Me llamo Lucía.','Hola, Lucía.')")
            memory = Memoria(path)
            self.assertEqual(memory.history()['turns'][0]['question'], 'Me llamo Lucía.')
            self.assertEqual(memory.profile()['nombre'], 'Lucía')
            self.assertEqual(Memoria(path).history()['count'], 1)


if __name__ == '__main__':
    unittest.main()
