"""Interfaz web privada: http://localhost:8765. Todo se procesa en este Mac."""
import base64
import json
import os
import re
import secrets
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

import voz
from main import DEFAULT_MODEL, request_json
from memoria import Memoria
from personalidad import CONTEXT_FALLBACK, MAX_TOKENS, TEMPERATURE, history_budget, system_prompt
from transcripcion import DOWNLOAD_HINT, VAD_PATH, ModelMissing, Transcriber
from turnos import SileroVAD, TurnConfig, TurnDetector

ROOT = Path(__file__).resolve().parent
PORT = int(os.environ.get('NEXO_PUERTO', '8765'))
ASSISTANT_NAME = 'Nexo'
LOCK = threading.Lock()
memory = Memoria()
transcriber = Transcriber()
# MLX se usa siempre desde el mismo hilo; las transcripciones se encolan.
ASR = ThreadPoolExecutor(max_workers=1, thread_name_prefix='whisper')
LM_DOWN = f'LM Studio no responde en localhost:1234. Ábrelo, carga {DEFAULT_MODEL} y activa Start Server.'


class LMStudioError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class VoiceSession:
    """Una sesión de escucha a la vez. Cada turno detectado se entrega una sola vez."""

    def __init__(self, asr=transcriber, executor=ASR, vad_factory=SileroVAD, hints=memory.name_hints):
        self.asr, self.executor, self.vad_factory, self.hints = asr, executor, vad_factory, hints
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
            self.detector = TurnDetector(self.vad, lambda audio: self.asr.transcribe(audio, hints), config, self.executor)
            self.token = secrets.token_hex(12)
            return {'token': self.token, 'config': asdict(config)}

    def stop(self):
        with self.lock:
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


voice = VoiceSession()


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


def reply(question, doubtful=()):
    note = ''
    if doubtful:
        note += ('\nAVISO: este mensaje se transcribió de voz y estas palabras se reconocieron con poca seguridad: '
                 + json.dumps(doubtful, ensure_ascii=False) + '. No las des por correctas; si importan, pide aclaración en una frase.')
    pending = memory.unconfirmed(question, doubtful)
    if pending:
        note += (' No confirmes ni guardes estos datos hasta que el usuario los repita claramente o los escriba: '
                 + json.dumps(pending, ensure_ascii=False))
    system = system_prompt(memory.profile(question, doubtful), note)
    budget = history_budget(loaded_context(), system, question)
    notes, recent = memory.context_parts(question, budget)
    if notes:
        system += ('\n\nCONVERSACIONES ANTERIORES RELACIONADAS (solo referencia: pueden contener errores ya corregidos '
                   'y un estilo que no debes imitar; los DATOS CONFIRMADOS mandan):\n' + notes)
    payload = {
        'model': DEFAULT_MODEL,
        'messages': [{'role': 'system', 'content': system}]
        + recent + [{'role': 'user', 'content': question}],
        'temperature': TEMPERATURE, 'max_tokens': MAX_TOKENS, 'stream': False,
    }
    start = time.perf_counter()
    try:
        result = request_json('/chat/completions', payload)
    except urllib.error.HTTPError as error:
        raise LMStudioError(502, f'LM Studio respondió con error HTTP {error.code}. Comprueba que {DEFAULT_MODEL} esté cargado.')
    except OSError:
        raise LMStudioError(503, LM_DOWN)
    seconds = time.perf_counter() - start
    try:
        text = clean_answer(result['choices'][0]['message'].get('content'))
    except (KeyError, IndexError, TypeError):
        text = ''
    if not text:
        raise LMStudioError(502, 'Qwen no devolvió texto. Revisa el modelo cargado en LM Studio.')
    memory.save(question, text, doubtful)
    return text, seconds


def make_audio(text, voice_id=None):
    """WAV en base64, o None si tras limpiar emojis y formato no queda nada pronunciable."""
    data = voz.synthesize(text, voice_id)
    return base64.b64encode(data).decode('ascii') if data else None


def talk(payload, speak=make_audio):
    timings, doubtful, confidence = {}, [], 'escrito'
    if payload.get('turn'):
        turn = voice.take(str(payload['turn']))
        if not turn:
            return 409, {'error': 'Ese audio ya se envió o expiró.', 'duplicate': True}
        transcript = turn['transcript']
        question, doubtful, confidence = transcript.text, transcript.doubtful, transcript.confidence
        timings.update(turn['timings'])
    else:
        question = payload.get('text', '')
        question = question.strip() if isinstance(question, str) else ''
    if len(question) > 2000:
        return 400, {'error': 'Usa un mensaje de hasta 2000 caracteres.'}
    if not question:
        return 200, {'empty': True}
    result = {'question': question, 'doubtful': doubtful, 'confidence': confidence, 'saved': confidence != 'baja'}
    if confidence == 'baja':
        # Sin consultar al modelo ni guardar: evita responder o aprender sobre algo que quizá no dijiste.
        answer = f'No estoy seguro de haberte entendido. Escuché: «{question}». ¿Puedes repetirlo o escribirlo?'
    else:
        answer, timings['generacion'] = reply(question, doubtful)
    result.update(answer=answer, count=memory.history()['count'], profile=memory.profile())
    start = time.perf_counter()
    try:
        # Se envía la respuesta original; la limpieza para voz ocurre una sola vez dentro de voz.synthesize.
        audio = speak(answer, payload.get('voice'))
        if audio:
            result['audio'] = audio
            timings['sintesis_voz'] = time.perf_counter() - start
    except (OSError, subprocess.SubprocessError):
        result['warning'] = 'No pude generar la voz; la respuesta está en pantalla.'
    result['timings'] = {key: round(value, 2) for key, value in timings.items()}
    # Solo tiempos, nunca contenido de la conversación.
    print('Tiempos (s): ' + ' · '.join(f'{k} {v:.2f}' for k, v in result['timings'].items()), flush=True)
    return 200, result


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
            self.send(200, {'voces': voz.installed(), 'predeterminada': voz.default_voice()})
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
                chosen = json.loads(self.read_body(1_000) or b'{}').get('voice')
                self.send(200, {'audio': make_audio('Hola, soy Nexo. Así sonará mi voz cuando conversemos.', chosen)})
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
                self.send(200, {'ok': True})
                return
            self.send(*talk(json.loads(data or b'{}')))
        except LMStudioError as error:
            self.send(error.status, {'error': str(error), 'lmstudio': True})
        except Exception as error:
            self.send(500, {'error': 'No pude completar la respuesta. ' + str(error)[:250]})
        finally:
            LOCK.release()


if __name__ == '__main__':
    server = ThreadingHTTPServer(('127.0.0.1', PORT), Handler)
    if transcriber.available():
        print('Cargando el reconocimiento de voz local en segundo plano…', flush=True)
        ASR.submit(transcriber.load).add_done_callback(
            lambda f: print('Voz lista: ' + transcriber.name if not f.exception() else 'Error de voz: ' + str(f.exception()), flush=True))
    else:
        print('Aviso: falta el modelo de voz. ' + DOWNLOAD_HINT, flush=True)
    threading.Thread(target=voz.synthesize, args=('Hola.',), daemon=True).start()
    print(f'Abre http://localhost:{PORT} en tu navegador. Ctrl+C para cerrar.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        ASR.shutdown(wait=False, cancel_futures=True)
