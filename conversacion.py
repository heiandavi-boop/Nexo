"""Preferencias conversacionales y estado temporal de la conversación.

Tres capas distintas:
- Perfil factual (memoria.py, tabla profile): nombre, edad, profesión…
- Preferencias conversacionales persistentes (tabla conversation_preferences): cómo quiere conversar el usuario.
- Estado temporal (ConversationState, solo en memoria del proceso): tema actual, temas abandonados recientes.

La detección es conservadora: solo instrucciones explícitas del usuario, nunca preguntas, ejemplos,
hipótesis, citas ni palabras reconocidas con poca seguridad.
"""
import os
import re
import unicodedata
from datetime import datetime

AVOID_PERSISTENCE = 'evitar_insistir_en_un_tema'
REJECTED_TOPICS = 'temas_rechazados'
PREFERENCE_TEXT = {AVOID_PERSISTENCE: 'Evitar insistir repetidamente en un mismo tema.'}

HYPOTHETICAL = re.compile(r'[¿?"“”«»]|\b(si fuera|si te dijera|imagina|supongamos|por ejemplo|dijo que|dice que)\b', re.I)
CHANGE_TOPIC = re.compile(
    r'\b(cambiemos de tema|cambia(?:r)? de tema|hablemos de otra cosa|pasemos a otra cosa|'
    r'dejemos (?:ese|este|el) tema|no quiero seguir (?:hablando de eso|con esto|con eso|con ese tema|con este tema))\b', re.I)
# Verbo de rechazo seguido del tema: «no vuelvas a hablar del café», «no menciones más la taza»…
ABANDON = re.compile(
    r'\b(?:no\s+(?:quiero\s+(?:que\s+)?)?(?:vuelvas\s+a\s+|sigas\s+)?(?:me\s+|te\s+)?(?:hables|hablar|hablando|menciones|mencionar|mencionando|'
    r'insistas|insistir|insistiendo|nombres|nombrar)|deja\s+de\s+(?:hablar|mencionar|insistir))'
    r'(?P<more>\s+m[aá]s)?\s+(?:(?:del|de|sobre|con|en|acerca\s+de)\b)?\s*(?P<topic>[^.;!\n]+)', re.I)
# Formas que piden no volver al tema nunca: se guardan como preferencia persistente.
FOREVER = re.compile(r'\b(?:vuelvas\s+a|nunca|jam[aá]s)\b', re.I)
AVOID_RULE = re.compile(
    r'\bno\s+(?:quiero\s+que\s+)?(?:insistas|persistas|sigas\s+insistiendo|te\s+quedes|repitas)\b[^.?!]*'
    r'\b(?:mismo\s+tema|lo\s+mismo|la\s+misma\s+cosa|un\s+(?:solo\s+)?tema|los\s+temas)\b', re.I)
ALLOW_PERSISTENCE = re.compile(r'\b(?:ya\s+)?puedes\s+(?:volver\s+a\s+)?(?:insistir|retomar|profundizar)\s+(?:en\s+)?(?:los|un|el)\s+(?:mismo\s+)?tema', re.I)
ARTICLES = {'a', 'al', 'el', 'la', 'los', 'las', 'lo', 'un', 'una', 'unos', 'unas', 'mi', 'tu', 'su', 'ese', 'esa', 'este', 'esta', 'tema', 'de', 'del', 'mas', 'más',
            'hoy', 'ahora', 'ya', 'nunca', 'jamás', 'tanto', 'porfa'}
PRONOUNS = {'eso', 'esto', 'ello', 'ese tema', 'este tema', 'lo mismo', 'mismo tema', 'el mismo tema', 'tema'}
STOP = set('que como para una por con del las los esto eso eres tienes puedes dime hola algo creo estaria estar esta este quiero '
           'cuentame aqui tengo bien muy mas pero porque cuando donde sobre tambien hace hoy ahora'.split())

DAYS = ['lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado', 'domingo']
MONTHS = ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre']


def plain(text):
    text = unicodedata.normalize('NFKD', text.lower())
    return ''.join(c for c in text if not unicodedata.combining(c))


def _sentences(text):
    return [s for s in re.split(r'(?<=[.!?…])\s+|\n+', text) if s.strip()]


def _doubtful(fragment, doubtful):
    words = set(re.findall(r'\w+', plain(fragment)))
    return any(set(re.findall(r'\w+', plain(d))) & words for d in doubtful or ())


def _topics(fragment):
    topics = []
    fragment = re.split(r'\bpor\s+favor\b', fragment, flags=re.I)[0]
    for part in re.split(r',|\s+y\s+|\s+ni\s+|\s+o\s+', fragment):
        words = [w for w in re.findall(r'\w+', part.lower()) if w not in ARTICLES]
        topic = ' '.join(words[:3]).strip()
        if topic and topic not in PRONOUNS and len(topic) >= 3:
            topics.append(topic)
    return topics


def mentions(text, topic):
    """¿El texto nombra el tema? Ignora tildes y admite plurales simples (café/cafés)."""
    words = plain(text)
    return all(re.search(r'\b' + re.escape(w) + r'(?:s|es)?\b', words) for w in plain(topic).split())


def analyze(text, doubtful=()):
    """Lo que el mensaje pide explícitamente sobre el rumbo de la conversación."""
    result = {'change_topic': False, 'abandon': [], 'reject': [], 'preferences': {}}
    for sentence in _sentences(text or ''):
        if HYPOTHETICAL.search(sentence) or _doubtful(sentence, doubtful):
            continue
        if CHANGE_TOPIC.search(sentence):
            result['change_topic'] = True
        if AVOID_RULE.search(sentence):
            # Preferencia general sobre cómo conversar, no un tema concreto.
            result['preferences'][AVOID_PERSISTENCE] = True
            continue
        if ALLOW_PERSISTENCE.search(sentence):
            result['preferences'][AVOID_PERSISTENCE] = False
            continue
        for match in ABANDON.finditer(sentence):
            fragment = CHANGE_TOPIC.split(match['topic'])[0]
            topics = [t for t in _topics(fragment) if 'mismo' not in t.split()]
            if not topics:
                # «no sigas con eso»: se refiere al tema en curso.
                result['change_topic'] = True
                continue
            result['abandon'] += [t for t in topics if t not in result['abandon']]
            if match['more'] or FOREVER.search(match[0]):
                result['reject'] += [t for t in topics if t not in result['reject']]
    return result


def reintroduced(text, topics, abandoned_now=()):
    """Temas rechazados que el propio usuario vuelve a abrir en este mensaje."""
    return [t for t in topics if t not in abandoned_now and mentions(text, t)]


def keywords(text, limit=4):
    words = [w for w in re.findall(r'\w+', plain(text)) if len(w) > 3 and w not in STOP]
    return list(dict.fromkeys(words))[:limit]


def local_now(now=None):
    """Fecha, hora y zona horaria del reloj del sistema (nunca del conocimiento del modelo)."""
    now = (now or datetime.now()).astimezone()
    zone = os.environ.get('TZ') or ''
    if not zone:
        target = os.path.realpath('/etc/localtime')
        zone = target.split('zoneinfo/', 1)[1] if 'zoneinfo/' in target else (now.tzname() or '')
    return {'fecha': f'{DAYS[now.weekday()]} {now.day} de {MONTHS[now.month - 1]} de {now.year}',
            'hora': now.strftime('%H:%M'), 'zona': zone}


def local_context(now=None):
    info = local_now(now)
    return f"CONTEXTO LOCAL ACTUAL (reloj de este Mac)\nFecha: {info['fecha']}\nHora: {info['hora']}\nZona horaria: {info['zona']}"


class ConversationState:
    """Estado de esta ejecución: no se guarda en disco ni sustituye al historial."""

    def __init__(self):
        self.current_topic = []
        self.abandoned = []
        self.boundary = 0          # los turnos con id <= boundary no se envían como diálogo reciente
        self.changed_turns = 0     # turnos que quedan con el aviso de cambio de tema
        self.anchor = 0            # primer turno de la ventana de diálogo reciente

    def window_start(self, last_id, maximum=6, keep=2):
        """Primer turno que se envía como diálogo. La ventana solo crece (LM Studio reutiliza lo ya calculado)
        y, al superar `maximum` turnos, vuelve a empezar con los `keep` más recientes."""
        if not self.anchor or self.anchor > last_id + 1:
            self.anchor = max(1, last_id - 3)
        self.anchor = max(self.anchor, self.boundary)
        if last_id - self.anchor + 1 > maximum:
            self.anchor = max(self.boundary, last_id - keep + 1)
        return self.anchor

    def observe(self, text, doubtful=(), last_turn_id=0, rejected=()):
        """Actualiza el estado con el mensaje del usuario antes de responderle."""
        found = analyze(text, doubtful)
        for topic in reintroduced(text, list(self.abandoned), found['abandon']):
            self.abandoned.remove(topic)
        self.changed_turns = max(0, self.changed_turns - 1)
        if found['abandon'] or found['change_topic']:
            for topic in found['abandon']:
                if topic not in self.abandoned:
                    self.abandoned.append(topic)
            self.current_topic = []
            # El diálogo anterior al cambio, incluido el propio pedido (que nombra el tema), deja de enviarse
            # como conversación reciente; sigue en la base y el estado resume la instrucción.
            self.boundary = last_turn_id + 2
            self.changed_turns = 2
        else:
            self.current_topic = keywords(text) or self.current_topic
        return found

    def prompt_section(self, preferences=None, rejected=()):
        preferences = preferences or {}
        topics = list(dict.fromkeys(list(self.abandoned) + list(rejected)))
        active = [PREFERENCE_TEXT[k] for k, v in preferences.items() if v and k in PREFERENCE_TEXT]
        if not (topics or active or self.changed_turns):
            return ''
        lines = ['ESTADO CONVERSACIONAL ACTUAL (tiene prioridad sobre la continuidad con mensajes anteriores)']
        if topics:
            lines += ['Temas abandonados:'] + [f'- {t}' for t in topics]
        if active:
            lines += ['Preferencias activas:'] + [f'- {p}' for p in active]
        if self.changed_turns:
            lines.append('El usuario acaba de pedir cambiar de tema: no retomes lo que se hablaba antes de ese pedido.')
        lines += ['Reglas:',
                  '- No reintroduzcas temas abandonados ni sus asociaciones, homónimos, juegos de palabras o ejemplos; solo vuelve a ellos si el usuario los menciona explícitamente.',
                  '- No los menciones ni siquiera para confirmar que dejarás de hacerlo.',
                  '- Si el usuario vuelve a mencionarlos explícitamente, puedes tratarlos con normalidad.']
        return '\n'.join(lines)
