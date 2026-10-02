"""Historial persistente con recuperación de recuerdos por palabras clave."""
import re
import sqlite3
import json
from pathlib import Path

from conversacion import REJECTED_TOPICS, analyze, mentions, reintroduced

DEFAULT_PATH = Path(__file__).resolve().parent / 'datos' / 'memoria.sqlite3'


class Memoria:
    def __init__(self, path=DEFAULT_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS turns (id INTEGER PRIMARY KEY, question TEXT NOT NULL, answer TEXT NOT NULL, created TEXT DEFAULT CURRENT_TIMESTAMP)')
            db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(question, answer, tokenize='unicode61 remove_diacritics 2')")
            db.execute('CREATE TABLE IF NOT EXISTS profile (key TEXT PRIMARY KEY, value TEXT NOT NULL, source TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
            # Preferencias sobre cómo conversar; separadas del perfil factual.
            db.execute('CREATE TABLE IF NOT EXISTS conversation_preferences (key TEXT PRIMARY KEY, value TEXT NOT NULL, source TEXT NOT NULL, updated TEXT DEFAULT CURRENT_TIMESTAMP)')
            if not db.execute("SELECT 1 FROM metadata WHERE key='profile_v1'").fetchone():
                for (question,) in db.execute('SELECT question FROM turns ORDER BY id').fetchall():
                    self._update_profile(db, question)
                db.execute("INSERT INTO metadata VALUES ('profile_v1','1')")

    @staticmethod
    def extract_facts(text):
        # Solo declaraciones explícitas del usuario, nunca respuestas del modelo.
        # Conservador: no aprender de preguntas, citas, ejemplos o hipótesis.
        if re.search(r'[¿?"“”]|\b(si fuera|imagina|supongamos|por ejemplo|dijo|dice que)\b', text, re.I):
            return {}
        facts = {}
        match = re.search(r'\b(?:mi nombre es|me llamo)\s+([^.,;!\n]+)', text, re.I)
        if match:
            name = re.split(r'\s+(?:y\s+)?(?:soy|tengo|vivo|trabajo)\b', match[1], flags=re.I)[0].strip()
            if 1 <= len(name.split()) <= 6 and len(name) <= 80 and not re.search(r'\b(no|no es|desconocido)\b', name, re.I):
                facts['nombre'] = name
        match = re.search(r'(?<!\w)(?<!no )(?<!No )(?:tengo|tenía)\s+(\d{1,3})\s+años\b', text, re.I)
        if match and 1 <= int(match[1]) <= 120:
            facts['edad declarada'] = match[1] + ' años'
        match = re.search(r'\b(?:soy|trabajo como)\s+(?:un[ao]?\s+)?(ingenier[oa]s?\s+(?:de\s+|en\s+)?sistemas)\b', text, re.I)
        if match and not re.search(r'\bno\s+(?:soy|trabajo como)\s', text, re.I):
            facts['profesión'] = 'ingeniera de sistemas' if match[1].lower().startswith('ingeniera') else 'ingeniero de sistemas'
        match = re.search(r'\bmi profesión es\s+([^.,;!\n]+)', text, re.I)
        if match and len(match[1]) < 100:
            facts['profesión'] = match[1].strip()
        if re.search(r'\bno soy\s+(?:un[ao]?\s+)?emprendedor[ao]?\b', text, re.I):
            facts['emprendimiento'] = 'El usuario aclaró que NO es emprendedor.'
        elif re.search(r'\bsoy\s+(?:un[ao]?\s+)?emprendedor[ao]?\b', text, re.I):
            facts['emprendimiento'] = 'El usuario declaró que es emprendedor.'
        return facts

    @staticmethod
    def _is_doubtful(value, doubtful):
        words = set(re.findall(r'\w+', value.lower()))
        return any(set(re.findall(r'\w+', d.lower())) & words for d in doubtful or ())

    @classmethod
    def unconfirmed(cls, text, doubtful=()):
        """Datos que el mensaje parece declarar, pero contienen palabras mal reconocidas."""
        return {k: v for k, v in cls.extract_facts(text).items() if cls._is_doubtful(v, doubtful)}

    @classmethod
    def _update_profile(cls, db, question, doubtful=()):
        for key, value in cls.extract_facts(question).items():
            if cls._is_doubtful(value, doubtful):
                continue
            db.execute('INSERT INTO profile(key,value,source) VALUES (?,?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value, source=excluded.source', (key, value, question))

    def profile(self, question='', doubtful=()):
        with self.connect() as db:
            facts = dict(db.execute('SELECT key,value FROM profile').fetchall())
        facts.update({k: v for k, v in self.extract_facts(question).items() if not self._is_doubtful(v, doubtful)})
        return facts

    def profile_prompt(self, question='', doubtful=()):
        return json.dumps(self.profile(question, doubtful), ensure_ascii=False)

    def name_hints(self):
        name = self.profile().get('nombre')
        return [name] if name else []

    @staticmethod
    def _apply_preferences(current, question, doubtful=()):
        """Preferencias tras aplicar el mensaje: instrucciones explícitas y temas que el usuario reabre."""
        found = analyze(question, doubtful)
        updated = dict(current)
        updated.update(found['preferences'])
        rejected = [t for t in current.get(REJECTED_TOPICS, [])
                    if t not in reintroduced(question, current.get(REJECTED_TOPICS, []), found['abandon'])]
        rejected += [t for t in found['reject'] if t not in rejected]
        updated[REJECTED_TOPICS] = rejected
        if not rejected:
            updated.pop(REJECTED_TOPICS)
        return updated

    def preferences(self, question='', doubtful=()):
        with self.connect() as db:
            stored = {k: json.loads(v) for k, v in db.execute('SELECT key,value FROM conversation_preferences')}
        return self._apply_preferences(stored, question, doubtful) if question else stored

    @classmethod
    def _update_preferences(cls, db, question, doubtful=()):
        stored = {k: json.loads(v) for k, v in db.execute('SELECT key,value FROM conversation_preferences')}
        updated = cls._apply_preferences(stored, question, doubtful)
        for key in set(stored) - set(updated):
            db.execute('DELETE FROM conversation_preferences WHERE key=?', (key,))
        for key, value in updated.items():
            if stored.get(key) != value:
                db.execute('INSERT INTO conversation_preferences(key,value,source) VALUES (?,?,?) ON CONFLICT(key) DO UPDATE SET '
                           'value=excluded.value, source=excluded.source, updated=CURRENT_TIMESTAMP',
                           (key, json.dumps(value, ensure_ascii=False), question))

    def last_id(self):
        with self.connect() as db:
            return db.execute('SELECT COALESCE(MAX(id), 0) FROM turns').fetchone()[0]

    def connect(self):
        return sqlite3.connect(str(self.path))

    def save(self, question, answer, doubtful=()):
        with self.connect() as db:
            cursor = db.execute('INSERT INTO turns(question,answer) VALUES (?,?)', (question, answer))
            db.execute('INSERT INTO search(rowid,question,answer) VALUES (?,?,?)', (cursor.lastrowid, question, answer))
            self._update_profile(db, question, doubtful)
            self._update_preferences(db, question, doubtful)

    def history(self):
        with self.connect() as db:
            rows = db.execute('SELECT id,question,answer,created FROM turns ORDER BY id DESC LIMIT 100').fetchall()
            count = db.execute('SELECT COUNT(*) FROM turns').fetchone()[0]
        return {'turns': [dict(zip(('id','question','answer','created'), row)) for row in reversed(rows)], 'count': count, 'profile': self.profile()}

    def _select(self, question, budget, since_id=0, exclude=(), notes_budget=1500):
        stop = {'que','como','para','una','por','con','del','las','los','esto','eso','eres','tienes','puedes','recuerdas','dime'}
        words = [w for w in re.findall(r'\w+', question.lower()) if len(w) > 2 and w not in stop][:16]
        aliases = {'edad': ['años', 'tengo'], 'profesión': ['soy', 'ingeniero', 'trabajo'], 'profesion': ['soy', 'ingeniero', 'trabajo'], 'llamo': ['nombre', 'llamo']}
        for word in list(words):
            words.extend(aliases.get(word, []))
        with self.connect() as db:
            if since_id:
                # Ventana anclada: crece turno a turno, así el inicio del diálogo no cambia entre peticiones.
                recent = db.execute('SELECT id,question,answer FROM turns WHERE id >= ? ORDER BY id DESC LIMIT 8', (since_id,)).fetchall()
            else:
                recent = db.execute('SELECT id,question,answer FROM turns ORDER BY id DESC LIMIT 4').fetchall()
            older = []
            if words:
                query = ' OR '.join('"' + word + '"' for word in words)
                older = db.execute('SELECT rowid, question, answer FROM search WHERE search MATCH ? ORDER BY rank LIMIT 4', ('question : (' + query + ')',)).fetchall()
        recent_ids = {row[0] for row in recent}
        kept = []
        # El presupuesto (en caracteres) lo calcula quien conoce el contexto disponible del modelo.
        for row in recent:
            # Respuestas antiguas recortadas: bastan para el hilo y alargan menos la espera.
            q, a = row[1][:1600], row[2][:700]
            if len(q) + len(a) > budget:
                break
            kept.append((row[0], q, a, True))
            budget -= len(q) + len(a)
        budget = min(budget, notes_budget)
        for row in older:
            # Un recuerdo de un tema abandonado no vuelve como referencia, salvo que el usuario lo reabra.
            if row[0] in recent_ids or any(mentions(row[1] + ' ' + row[2], topic) for topic in exclude):
                continue
            q, a = row[1][:1600], row[2][:700]
            if len(q) + len(a) <= budget:
                kept.append((row[0], q, a, False))
                budget -= len(q) + len(a)
        return sorted(kept)

    def context(self, question, budget=5000):
        return [message for _, q, a, _ in self._select(question, budget, notes_budget=budget) for message in (
            {'role': 'user', 'content': q}, {'role': 'assistant', 'content': a})]

    def context_parts(self, question, budget=5000, since_id=0, exclude=()):
        """Recuerdos antiguos como notas de referencia y solo lo reciente como diálogo.

        Las respuestas antiguas pueden tener errores ya corregidos o un estilo distinto;
        como notas, el modelo las consulta sin tomarlas como ejemplo de cómo responder.
        since_id: el diálogo reciente empieza en ese turno (tras un cambio de tema).
        exclude: temas abandonados que no deben volver como recuerdos relacionados."""
        kept = self._select(question, budget, since_id, exclude)
        notes = '\n'.join(f'- [{turn_id}] Usuario: {q}\n  Nexo respondió: {a}' for turn_id, q, a, recent in kept if not recent)
        messages = [message for _, q, a, recent in kept if recent for message in (
            {'role': 'user', 'content': q}, {'role': 'assistant', 'content': a})]
        return notes, messages

    def clear(self):
        with self.connect() as db:
            db.execute('PRAGMA secure_delete=ON')
            db.execute('DELETE FROM turns')
            db.execute('DELETE FROM search')
            db.execute('DELETE FROM profile')
            db.execute('DELETE FROM conversation_preferences')
            db.execute("INSERT INTO search(search) VALUES('optimize')")
