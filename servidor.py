"""Interfaz web privada: http://localhost:8765. Todo se procesa en este Mac."""
import base64
import json
import os
import queue
import re
import secrets
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

import voz
from conversacion import REJECTED_TOPICS, ConversationState, local_context
from main import DEFAULT_MODEL, request_json, stream_json
from memoria import Memoria
from personalidad import CONTEXT_FALLBACK, MAX_TOKENS, TEMPERATURE, history_budget, prompt_parts, user_turn
from prosodia import ProsodyAnalyzer
from transcripcion import DOWNLOAD_HINT, VAD_PATH, ModelMissing, Transcriber
from turnos import SileroVAD, TurnConfig, TurnDetector

ROOT = Path(__file__).resolve().parent
PORT = int(os.environ.get('NEXO_PUERTO', '8765'))
ASSISTANT_NAME = 'Nexo'
LOCK = threading.Lock()
# NEXO_MEMORIA permite probar con otro archivo sin tocar datos/memoria.sqlite3.
memory = Memoria(os.environ['NEXO_MEMORIA']) if os.environ.get('NEXO_MEMORIA') else Memoria()
transcriber = Transcriber()
# MLX se usa siempre desde el mismo hilo; las transcripciones se encolan.
ASR = ThreadPoolExecutor(max_workers=1, thread_name_prefix='whisper')
# Extracción NumPy medida <2 ms p95 en 5 s sintéticos; no carga modelo adicional.
PROSODY_TIMEOUT = 0.05
PROSODY = ProsodyAnalyzer()
PROSODY_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix='prosodia')
LM_DOWN = f'LM Studio no responde en localhost:1234. Ábrelo, carga {DEFAULT_MODEL} y activa Start Server.'
SAMPLE_TEXT = ('Hola, Anderson. Qué bueno seguir conversando contigo. Hoy podemos avanzar un poco más con tu asistente. '
               '¿Prefieres probar una idea nueva o terminar lo que dejamos pendiente?')


class LMStudioError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class VoiceSession:
    """Una sesión de escucha a la vez. Cada turno detectado se entrega una sola vez."""

    def __init__(self, asr=transcriber, executor=ASR, vad_factory=SileroVAD, hints=memory.name_hints,
                 analysis=None, analysis_executor=None):
        self.asr, self.executor, self.vad_factory, self.hints = asr, executor, vad_factory, hints
        self.analysis, self.analysis_executor = analysis, analysis_executor
        self.lock = threading.Lock()
        self.token = self.detector = self.vad = None
        self.pending = {}

    def start(self, settings=None):
        if not self.asr.available():
            raise ModelMissing('Falta el modelo de reconocimiento de voz. ' + DOWNLOAD_HINT)
        with self.lock:
            if self.vad is None:
                self.vad = self.vad_factory()
            hints = [ASSISTANT_NAME] + list(self.hints())
            config = TurnConfig.create(settings)
            self.detector = TurnDetector(self.vad, lambda audio: self.asr.transcribe(audio, hints), config,
                                         self.executor, self.analysis, self.analysis_executor)
            self.token = secrets.token_hex(12)
            return {'token': self.token, 'config': asdict(config)}

    def stop(self):
        with self.lock:
            if self.detector:
                self.detector.cancel()
            for turn in self.pending.values():
                future = turn.get('voice_analysis')
                if future:
                    future.cancel()
            self.pending.clear()
            self.token = self.detector = None

    def chunk(self, token, data):
        with self.lock:
            if not token or token != self.token or self.detector is None:
                return None
            samples = np.frombuffer(data[:len(data) // 2 * 2], dtype='<i2').astype(np.float32) / 32768
            event = self.detector.feed(samples)
            result = {key: event[key] for key in ('state', 'prob', 'silence', 'discarded', 'error', 'incomplete')}
            result['level'] = round(float(np.sqrt(np.mean(samples ** 2))), 4) if len(samples) else 0.0
            turn = event['turn']
            if turn or event['error']:
                self.token = self.detector = None
            if turn:
                self.pending = {turn['id']: turn}
                result['turn'] = dict(turn['transcript'].public(), id=turn['id'], incomplete=turn['incomplete'], timings=turn['timings'])
            return result

    def take(self, turn_id):
        with self.lock:
            return self.pending.pop(turn_id, None)


voice = VoiceSession(analysis=PROSODY.analyze, analysis_executor=PROSODY_EXECUTOR)
# Estado temporal de esta ejecución (tema actual, temas abandonados); no se guarda en disco.
conversation = ConversationState()


def clean_answer(text):
    text = re.sub(r'<think>.*?</think>', '', text or '', flags=re.S)
    text = re.sub(r'<think>.*$', '', text, flags=re.S)
    # Corta bucles de repetición: una misma línea larga solo puede aparecer una vez.
    lines, seen = [], set()
    for line in text.split('\n'):
        key = line.strip().lower()
        if len(key) > 20 and key in seen:
            break
        seen.add(key)
        lines.append(line)
    text = '\n'.join(lines)
    return re.sub(r'[ \t]+\n', '\n', text).strip()


def format_timings(timings):
    return ' · '.join(f'{key} {value:.2f} {"ms" if key.endswith("_ms") else "s"}'
                       for key, value in timings.items())


_context_cache = {'tokens': None, 'checked': 0.0}


def loaded_context():
    """Tokens de contexto con que LM Studio cargó el modelo; se consulta como mucho cada minuto."""
    if time.monotonic() - _context_cache['checked'] > 60:
        _context_cache['checked'] = time.monotonic()
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(f'http://127.0.0.1:1234/api/v0/models/{DEFAULT_MODEL}', timeout=2) as response:
                _context_cache['tokens'] = int(json.load(response).get('loaded_context_length') or 0) or None
        except (OSError, ValueError):
            _context_cache['tokens'] = None
    return _context_cache['tokens'] or CONTEXT_FALLBACK


def build_payload(question, doubtful=(), voice_context=None):
    note = ''
    if voice_context:
        note += '\n' + voice_context
    if doubtful:
        note += ('\nAVISO: este mensaje se transcribió de voz y estas palabras se reconocieron con poca seguridad: '
                 + json.dumps(doubtful, ensure_ascii=False) + '. Aplica las reglas sobre transcripciones imperfectas: si el contexto '
                 'deja casi claro qué quiso decir, responde con «Creo que hablas de…»; si hay varias opciones, pregunta; '
                 'si es un dato personal, no lo deduzcas.')
    pending = memory.unconfirmed(question, doubtful)
    if pending:
        note += (' No confirmes ni guardes estos datos hasta que el usuario los repita claramente o los escriba: '
                 + json.dumps(pending, ensure_ascii=False))
    conversation.observe(question, doubtful, memory.last_id())
    since = conversation.window_start(memory.last_id())
    preferences = memory.preferences(question, doubtful)
    rejected = preferences.pop(REJECTED_TOPICS, [])
    state = conversation.prompt_section(preferences, rejected)
    local = local_context()
    profile = memory.profile(question, doubtful)
    static, dynamic = prompt_parts(profile, note, local=local, state=state)
    budget = history_budget(loaded_context(), static + dynamic + user_turn('', ''), question)
    notes, recent = memory.context_parts(question, budget, since, conversation.abandoned + rejected)
    if notes:
        note += ('\n\nCONVERSACIONES ANTERIORES RELACIONADAS (solo referencia: pueden contener errores ya corregidos '
                 'y un estilo que no debes imitar; los DATOS CONFIRMADOS mandan):\n' + notes)
        static, dynamic = prompt_parts(profile, note, local=local, state=state)
    # Instrucciones fijas primero (LM Studio reutiliza su cálculo); lo que cambia va con el mensaje actual.
    payload = {
        'model': DEFAULT_MODEL,
        'messages': [{'role': 'system', 'content': static}]
        + recent + [{'role': 'user', 'content': user_turn(dynamic, question)}],
        'temperature': TEMPERATURE, 'max_tokens': MAX_TOKENS, 'stream': False,
    }
    return payload


def _lm_error(error):
    if isinstance(error, urllib.error.HTTPError):
        return LMStudioError(502, f'LM Studio respondió con error HTTP {error.code}. Comprueba que {DEFAULT_MODEL} esté cargado.')
    return LMStudioError(503, LM_DOWN)


def _finish_reply(question, raw, doubtful):
    text = clean_answer(raw)
    if not text:
        raise LMStudioError(502, 'Qwen no devolvió texto. Revisa el modelo cargado en LM Studio.')
    memory.save(question, text, doubtful)
    return text


def reply(question, doubtful=(), voice_context=None):
    payload = build_payload(question, doubtful, voice_context)
    start = time.perf_counter()
    try:
        result = request_json('/chat/completions', payload)
    except OSError as error:
        raise _lm_error(error)
    seconds = time.perf_counter() - start
    try:
        raw = result['choices'][0]['message'].get('content')
    except (KeyError, IndexError, TypeError):
        raw = ''
    return _finish_reply(question, raw, doubtful), seconds


def _repeats(text):
    lines = [line.strip().lower() for line in text.split('\n')[:-1] if len(line.strip()) > 20]
    return len(lines) != len(set(lines))


def _cut_repeat(raw):
    """Recorta en la primera línea larga repetida, conservando el texto previo tal cual."""
    kept, seen = [], set()
    for line in raw.split('\n'):
        key = line.strip().lower()
        if len(key) > 20 and key in seen:
            break
        seen.add(key)
        kept.append(line)
    return '\n'.join(kept)


class Cancelled(Exception):
    """La página cerró la conexión (pausa o nuevo turno): se deja de generar y no se guarda nada."""


def reply_stream(question, doubtful=(), on_text=None, cancelled=None, marks=None, voice_context=None):
    """Como reply(), pero entrega el texto acumulado a on_text mientras Qwen lo genera.

    marks recibe 'first_token' (segundos desde la petición hasta el primer texto útil).
    Solo se guarda en memoria si el stream termina bien; un corte o error no deja respuestas truncadas."""
    payload = dict(build_payload(question, doubtful, voice_context), stream=True)
    start, raw = time.perf_counter(), ''
    marks = {} if marks is None else marks
    events = stream_json('/chat/completions', payload)
    try:
        for event in events:
            if cancelled and cancelled():
                raise Cancelled()
            try:
                delta = event['choices'][0].get('delta', {}).get('content') or ''
            except (KeyError, IndexError, TypeError):
                continue
            if not delta:
                continue
            if 'first_token' not in marks and delta.strip():
                marks['first_token'] = time.perf_counter() - start
            raw += delta
            # Un bucle de líneas repetidas se corta aquí, antes de seguir generándolo y leyéndolo en voz alta.
            if '\n' in delta and _repeats(raw):
                raw = _cut_repeat(raw)
                break
            if on_text:
                on_text(raw)
    except OSError as error:
        raise _lm_error(error)
    finally:
        events.close()
    text = _finish_reply(question, raw, doubtful)
    if on_text:
        on_text(raw, final=True)
    return text, time.perf_counter() - start


def warm_llm():
    """Deja calculados en LM Studio las instrucciones fijas y el diálogo actual (1 token, sin guardar nada),
    para que en el próximo turno solo falte procesar el mensaje nuevo."""
    try:
        last = memory.last_id()
        static, dynamic = prompt_parts(memory.profile())
        budget = history_budget(loaded_context(), static + dynamic, '')
        _, recent = memory.context_parts('', budget, conversation.window_start(last), conversation.abandoned)
        request_json('/chat/completions', {'model': DEFAULT_MODEL, 'max_tokens': 1, 'temperature': TEMPERATURE, 'stream': False,
                                           'messages': [{'role': 'system', 'content': static}] + recent + [{'role': 'user', 'content': '.'}]},
                     timeout=60)
    except Exception:
        pass  # solo es una optimización


SENTENCE_END = re.compile(r'(?<=[.!?…;])\s+|\n+')
SOFT_BREAK = re.compile(r'(?<=[,:—])\s+')
# Criterio de troceo para la voz:
# - primera frase: desde 12 caracteres, para empezar a hablar cuanto antes («Tienes 40 años.» ya sirve);
# - siguientes: al menos 60 caracteres por fragmento, para no partir la entonación en trozos de pocas palabras;
# - sin puntuación fuerte: si el texto pendiente pasa de 90 (primera) o 220 caracteres, se corta en la última
#   coma o dos puntos, que es una pausa natural (~5 y ~13 s de habla), en vez de esperar al final de una frase larga.
FIRST_MIN, NEXT_MIN, FIRST_SOFT, NEXT_SOFT = 12, 60, 90, 220


def split_ready(text, final=False, first=False):
    """Separa fragmentos listos para la voz; devuelve (fragmentos, resto pendiente)."""
    chunks, minimum, soft = [], FIRST_MIN if first else NEXT_MIN, FIRST_SOFT if first else NEXT_SOFT
    while True:
        cut = next((m for m in SENTENCE_END.finditer(text) if m.start() >= minimum or '\n' in m.group()), None)
        if not cut and len(text) > soft:
            breaks = [m for m in SOFT_BREAK.finditer(text) if minimum <= m.start() <= soft]
            cut = breaks[-1] if breaks else None
        if not cut:
            break
        if text[:cut.start()].strip():
            chunks.append(text[:cut.start()].strip())
            minimum, soft = NEXT_MIN, NEXT_SOFT
        text = text[cut.end():]
    if final and text.strip():
        chunks.append(text.strip())
        text = ''
    return chunks, text


class SpeechPipeline:
    """Sintetiza fragmento a fragmento en otro hilo mientras el modelo sigue escribiendo.

    La página reproduce en orden; mientras suena el fragmento N, aquí ya se sintetiza el N+1."""

    def __init__(self, speak, emit, start):
        self.speak, self.emit, self.start = speak, emit, start
        self.queue = queue.Queue()
        self.offset, self.first_sentence, self.first_audio, self.synth_seconds = 0, None, None, 0.0
        self.infos, self.failed, self.chunks, self.cancelled = [], False, 0, False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def update(self, raw, final=False):
        visible = re.sub(r'<think>.*?(?:</think>|$)', '', raw, flags=re.S)
        chunks, rest = split_ready(visible[self.offset:], final, first=self.offset == 0)
        self.offset = len(visible) - len(rest)
        for chunk in chunks:
            if self.first_sentence is None:
                self.first_sentence = time.perf_counter() - self.start
            self.chunks += 1
            self.queue.put(chunk)

    def cancel(self):
        self.cancelled = True
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

    def _run(self):
        while True:
            text = self.queue.get()
            if text is None:
                return
            if self.cancelled:
                continue
            started = time.perf_counter()
            try:
                audio, info = self.speak(text)
            except (OSError, subprocess.SubprocessError, voz.VoiceError):
                self.failed = True
                continue
            self.synth_seconds += time.perf_counter() - started
            if info:
                self.infos.append(info)
            if audio and not self.cancelled:
                if self.first_audio is None:
                    self.first_audio = time.perf_counter() - self.start
                if self.emit({'type': 'audio', 'audio': audio, 'text': text, 'voz': info}) is False:
                    self.cancel()

    def finish(self):
        self.queue.put(None)
        self.thread.join()


def make_audio(text, voice_id=None, strict=False):
    """(WAV en base64 o None si no hay nada pronunciable, información del motor realmente usado)."""
    result = voz.synthesize(text, voice_id, strict=strict)
    audio = base64.b64encode(result.audio).decode('ascii') if result.audio else None
    return audio, result.info()


def talk(payload, speak=make_audio):
    timings, doubtful, confidence = {}, [], 'escrito'
    error, question, doubtful, confidence, voice_context = _question(payload, timings)
    if error:
        return error
    result = {'question': question, 'doubtful': doubtful, 'confidence': confidence, 'saved': confidence != 'baja'}
    if confidence == 'baja':
        answer = low_confidence_answer(question)
    else:
        answer, timings['generacion'] = reply(question, doubtful, voice_context)
    result.update(answer=answer, count=memory.history()['count'], profile=memory.profile())
    start = time.perf_counter()
    try:
        # Se envía la respuesta original; la limpieza para voz ocurre una sola vez dentro de voz.synthesize.
        audio, info = speak(answer, payload.get('voice'))
        if audio:
            result['audio'] = audio
            timings['sintesis_voz'] = time.perf_counter() - start
        if info:
            result['voz'] = info
            if info.get('respaldo'):
                result['warning'] = f'No pude usar la voz elegida ({info["motivo"]}). Esta respuesta suena con Paulina.'
    except (OSError, subprocess.SubprocessError, voz.VoiceError):
        result['warning'] = 'No pude generar la voz; la respuesta está en pantalla.'
    result['timings'] = {key: round(value, 2) for key, value in timings.items()}
    # Solo tiempos, nunca contenido de la conversación.
    print('Tiempos: ' + format_timings(result['timings']), flush=True)
    return 200, result


def _question(payload, timings):
    """(error, pregunta, palabras dudosas, confianza, contexto vocal temporal)."""
    doubtful, confidence = [], 'escrito'
    turn = None
    if payload.get('turn'):
        turn = voice.take(str(payload['turn']))
        if not turn:
            return (409, {'error': 'Ese audio ya se envió o expiró.', 'duplicate': True}), '', [], '', None
        transcript = turn['transcript']
        question, doubtful, confidence = transcript.text, transcript.doubtful, transcript.confidence
        timings.update(turn['timings'])
    else:
        question = payload.get('text', '')
        question = question.strip() if isinstance(question, str) else ''
    if len(question) > 2000:
        return (400, {'error': 'Usa un mensaje de hasta 2000 caracteres.'}), '', [], '', None
    if not question:
        return (200, {'empty': True}), '', [], '', None
    voice_context = _voice_context(turn, question, confidence, timings) if turn else None
    return None, question, doubtful, confidence, voice_context


def _voice_context(turn, question, confidence, timings):
    """Consume una señal aceptada por Whisper con espera limitada; cualquier fallo equivale a texto normal."""
    future = turn.get('voice_analysis')
    if not future:
        return None
    if confidence == 'baja' or len(re.findall(r'\w+', question)) <= 2:
        future.cancel()
        return None

    started = time.perf_counter()
    try:
        features = future.result(timeout=PROSODY_TIMEOUT)
    except FutureTimeoutError:
        future.cancel()
        timings['espera_prosodia_ms'] = (time.perf_counter() - started) * 1000
        return None
    except Exception:
        timings['espera_prosodia_ms'] = (time.perf_counter() - started) * 1000
        return None

    timings['espera_prosodia_ms'] = (time.perf_counter() - started) * 1000
    try:
        timings['analisis_prosodia_ms'] = float(features.get('analysis_ms', 0.0))
        assessed = PROSODY.assess(features)
        if assessed.get('confidence', 0.0) < 0.58:
            return None
        return assessed.get('context')
    except Exception:
        return None


def low_confidence_answer(question):
    # Sin consultar al modelo ni guardar: evita responder o aprender sobre algo que quizá no dijiste.
    return f'No estoy seguro de haberte entendido. Escuché: «{question}». ¿Puedes repetirlo o escribirlo?'


def talk_stream(payload, emit, speak=make_audio):
    """Versión por eventos de talk(): pregunta, audio por frases (mientras se genera) y resumen final."""
    timings = {}
    error, question, doubtful, confidence, voice_context = _question(payload, timings)
    if error:
        emit(dict(error[1], type='error' if error[0] != 200 else 'empty', status=error[0]))
        return
    emit({'type': 'question', 'question': question, 'doubtful': doubtful, 'confidence': confidence})
    start = time.perf_counter()
    pipeline = SpeechPipeline(lambda text: speak(text, payload.get('voice')), emit, start)
    marks = {}
    try:
        if confidence == 'baja':
            answer = low_confidence_answer(question)
            pipeline.update(answer, final=True)
        else:
            answer, timings['generacion'] = reply_stream(question, doubtful, pipeline.update,
                                                         cancelled=lambda: pipeline.cancelled, marks=marks,
                                                         voice_context=voice_context)
    except Cancelled:
        pipeline.cancel()
        pipeline.finish()
        print('Respuesta cancelada por la página (no se guardó).', flush=True)
        return
    except LMStudioError as failure:
        pipeline.cancel()
        pipeline.finish()
        emit({'type': 'error', 'error': str(failure), 'lmstudio': True})
        return
    pipeline.finish()
    # Todas desde el inicio de la respuesta (t0 = turno recibido, ya transcrito); no se mezclan tramos.
    for key, value in (('first_token', marks.get('first_token')), ('first_sentence', pipeline.first_sentence),
                       ('first_audio', pipeline.first_audio)):
        if value is not None:
            timings[key] = value
    if pipeline.first_audio is not None:
        timings['sintesis_voz'] = pipeline.synth_seconds
    result = {'type': 'done', 'question': question, 'answer': answer, 'saved': confidence != 'baja',
              'count': memory.history()['count'], 'profile': memory.profile(),
              'timings': {key: round(value, 2) for key, value in timings.items()}}
    fallback = next((info for info in pipeline.infos if info.get('respaldo')), None)
    if fallback:
        result['warning'] = f'No pude usar la voz elegida ({fallback["motivo"]}). Esta respuesta suena con Paulina.'
    elif pipeline.failed:
        result['warning'] = 'No pude generar parte de la voz; la respuesta está en pantalla.'
    if pipeline.infos:
        result['voz'] = pipeline.infos[-1]
    # Solo métricas y tamaños, nunca contenido.
    print('Tiempos: ' + format_timings(result['timings'])
          + f' · {len(answer)} caracteres · {pipeline.chunks} fragmentos', flush=True)
    emit(result)
    if confidence != 'baja':
        threading.Thread(target=warm_llm, daemon=True).start()


def status():
    try:
        models = [item['id'] for item in request_json('/models', timeout=2)['data']]
        found = DEFAULT_MODEL in models
        lm = {'ok': found, 'mensaje': '' if found else f'LM Studio está activo, pero no encuentro {DEFAULT_MODEL}.'}
    except Exception:
        lm = {'ok': False, 'mensaje': LM_DOWN}
    ready = transcriber.available() and VAD_PATH.exists()
    message = transcriber.error or ('' if ready else 'Falta el modelo de voz o el detector. ' + DOWNLOAD_HINT)
    if ready and not (transcriber.model_dir / 'weights.safetensors').exists():
        message = 'Usando Whisper base de respaldo (menos preciso). ' + DOWNLOAD_HINT
    return {'lmstudio': lm, 'voz': {'ok': ready and not transcriber.error, 'motor': transcriber.name,
                                     'cargado': bool(transcriber.engine), 'mensaje': message}}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Evita registrar conversaciones o audios.
        pass

    def send(self, status, data, content_type='application/json'):
        body = json.dumps(data, ensure_ascii=False).encode() if content_type == 'application/json' else data
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; media-src 'self' blob:; frame-ancestors 'none'")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def valid(self, write=False):
        hosts = {f'localhost:{PORT}', f'127.0.0.1:{PORT}'}
        if self.headers.get('Host') not in hosts:
            self.send(403, {'error': 'Solo disponible desde localhost.'})
            return False
        origin = self.headers.get('Origin')
        if origin and origin not in {'http://' + host for host in hosts}:
            self.send(403, {'error': 'Origen no permitido.'})
            return False
        if write and self.headers.get('X-Local-App') != '1':
            self.send(403, {'error': 'Solicitud no permitida.'})
            return False
        return True

    def do_GET(self):
        if self.headers.get('Host') == f'127.0.0.1:{PORT}' and self.path == '/':
            # Un solo origen (localhost) para que el permiso del micrófono se reutilice.
            self.send_response(302)
            self.send_header('Location', f'http://localhost:{PORT}/')
            self.send_header('Content-Length', '0')
            self.end_headers()
            return
        if not self.valid():
            return
        if self.path == '/api/history':
            self.send(200, memory.history())
            return
        if self.path == '/api/status':
            self.send(200, status())
            return
        if self.path == '/api/voices':
            self.send(200, {'voces': voz.installed(), 'predeterminada': voz.default_voice(), 'muestra': SAMPLE_TEXT,
                            'qwen_cargado': voz.QWEN.loaded})
            return
        assets = {'/assets/avatar.png': ('assets/avatar.png', 'image/png'), '/': ('index.html', 'text/html; charset=utf-8'), '/app.js': ('app.js', 'text/javascript'), '/style.css': ('style.css', 'text/css'), '/anime.css': ('anime.css', 'text/css'), '/capture.js': ('capture.js', 'text/javascript'), '/vad.js': ('vad.js', 'text/javascript')}
        if self.path not in assets:
            self.send(404, {'error': 'No encontrado'})
            return
        name, mime = assets[self.path]
        self.send(200, (ROOT / 'web' / name).read_bytes(), mime)

    def read_body(self, limit):
        length = int(self.headers.get('Content-Length', '0'))
        if length < 0 or length > limit:
            raise ValueError('Solicitud demasiado grande.')
        return self.rfile.read(length)

    def do_POST(self):
        if not self.valid(write=True):
            return
        try:
            if self.path == '/api/voice/chunk':
                event = voice.chunk(self.headers.get('X-Voice-Token'), self.read_body(1_000_000))
                self.send(200, event) if event is not None else self.send(409, {'stale': True})
                return
            if self.path == '/api/voice/start':
                self.send(200, voice.start(json.loads(self.read_body(10_000) or b'{}')))
                return
            if self.path == '/api/voice/stop':
                voice.stop()
                self.send(200, {'ok': True})
                return
            if self.path == '/api/voice/sample':
                body = json.loads(self.read_body(8_000) or b'{}')
                text = str(body.get('text') or SAMPLE_TEXT).strip()[:600]
                try:
                    audio, info = make_audio(text, body.get('voice'), strict=True)
                except voz.VoiceError as error:
                    self.send(502, {'error': f'La voz elegida falló y no la sustituí: {error}'})
                    return
                if not audio:
                    self.send(400, {'error': 'El texto de prueba no tiene nada pronunciable.'})
                    return
                self.send(200, {'audio': audio, 'voz': info})
                return
        except ModelMissing as error:
            self.send(503, {'error': str(error), 'model': True})
            return
        except Exception as error:
            self.send(500, {'error': 'Falló la escucha local: ' + str(error)[:250]})
            return
        if self.path not in {'/api/talk', '/api/clear'}:
            self.send(404, {'error': 'No encontrado'})
            return
        if not LOCK.acquire(blocking=False):
            self.send(409, {'error': 'Hay otra conversación en proceso. Espera un momento.'})
            return
        try:
            data = self.read_body(100_000)
            if self.path == '/api/clear':
                memory.clear()
                conversation.__init__()
                self.send(200, {'ok': True})
                return
            payload = json.loads(data or b'{}')
            if payload.get('stream'):
                self.stream_events(payload)
                return
            self.send(*talk(payload))
        except LMStudioError as error:
            self.send(error.status, {'error': str(error), 'lmstudio': True})
        except Exception as error:
            self.send(500, {'error': 'No pude completar la respuesta. ' + str(error)[:250]})
        finally:
            LOCK.release()

    def stream_events(self, payload):
        """Una línea JSON por evento; la conexión se cierra al terminar (HTTP/1.0)."""
        self.send_response(200)
        self.send_header('Content-Type', 'application/x-ndjson; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.close_connection = True
        write_lock = threading.Lock()

        def emit(event):
            with write_lock:
                try:
                    self.wfile.write((json.dumps(event, ensure_ascii=False) + '\n').encode())
                    self.wfile.flush()
                    return True
                except (BrokenPipeError, ConnectionResetError):
                    return False  # la página pausó o empezó otro turno: se cancela la respuesta
        try:
            talk_stream(payload, emit)
        except Exception as error:
            emit({'type': 'error', 'error': 'No pude completar la respuesta. ' + str(error)[:250]})


if __name__ == '__main__':
    server = ThreadingHTTPServer(('127.0.0.1', PORT), Handler)
    if transcriber.available():
        print('Cargando el reconocimiento de voz local en segundo plano…', flush=True)
        ASR.submit(transcriber.load).add_done_callback(
            lambda f: print('Voz lista: ' + transcriber.name if not f.exception() else 'Error de voz: ' + str(f.exception()), flush=True))
    else:
        print('Aviso: falta el modelo de voz. ' + DOWNLOAD_HINT, flush=True)
    # Solo se precalienta Piper (ligero); Qwen3-TTS se carga cuando se elige o se prueba.
    if voz.default_voice() in voz.PIPER_VOICES:
        threading.Thread(target=voz.synthesize, args=('Hola.',), daemon=True).start()
    # Un token en LM Studio deja calculadas las instrucciones fijas: el primer turno empieza antes.
    threading.Thread(target=warm_llm, daemon=True).start()
    print(f'Abre http://localhost:{PORT} en tu navegador. Ctrl+C para cerrar.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        ASR.shutdown(wait=False, cancel_futures=True)
        PROSODY_EXECUTOR.shutdown(wait=False, cancel_futures=True)
