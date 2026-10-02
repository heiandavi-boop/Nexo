"""Voz de Nexo: Piper (neuronal, local), Paulina de macOS y Qwen3-TTS (MLX, proceso aparte)."""
import atexit
import base64
import io
import json
import os
import re
import select
import subprocess
import tempfile
import threading
import time
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
VOICES_DIR = ROOT / 'modelos' / 'voces'
VOICES_URL = 'https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/es/'
PIPER_VOICES = {
    'es_MX-claude-high': ('Claude · México (Piper, alta calidad)', 'es_MX/claude/high/'),
    'es_MX-ald-medium': ('Ald · México (Piper)', 'es_MX/ald/medium/'),
    'es_AR-daniela-high': ('Daniela · Argentina (Piper, alta calidad)', 'es_AR/daniela/high/'),
    'es_ES-davefx-medium': ('Davefx · España (Piper)', 'es_ES/davefx/medium/'),
}
PAULINA = 'paulina'
_cache = {}
_lock = threading.Lock()

QWEN_REPO = 'mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit'
QWEN_DIR = ROOT / 'modelos' / 'qwen3-tts-0.6b-customvoice-8bit'
QWEN_PYTHON = ROOT / '.venv-tts' / 'bin' / 'python'
QWEN_PREFIX = 'qwen3-'
QWEN_HINT = 'Instrucciones en el README, sección «Qwen3-TTS».'
# Hablantes del modelo (config.json) y su idioma nativo según la ficha oficial; ninguno es hispanohablante nativo.
QWEN_SPEAKERS = {
    'serena': ('Serena', 'chino'), 'vivian': ('Vivian', 'chino'), 'uncle_fu': ('Uncle Fu', 'chino'),
    'dylan': ('Dylan', 'chino de Pekín'), 'eric': ('Eric', 'chino de Sichuan'), 'ryan': ('Ryan', 'inglés'),
    'aiden': ('Aiden', 'inglés'), 'ono_anna': ('Ono Anna', 'japonés'), 'sohee': ('Sohee', 'coreano'),
}


class VoiceError(RuntimeError):
    pass


def wav_seconds(data):
    with wave.open(io.BytesIO(data)) as wav:
        return wav.getnframes() / wav.getframerate()


@dataclass
class Synthesis:
    audio: Optional[bytes]
    requested: str
    engine: str = ''
    voice: str = ''
    fallback: bool = False
    reason: str = ''
    load_seconds: Optional[float] = None
    seconds: float = 0.0
    peak_gb: Optional[float] = None

    def info(self):
        return {'solicitada': self.requested, 'motor': self.engine, 'voz': self.voice, 'respaldo': self.fallback,
                'motivo': self.reason, 'carga_s': self.load_seconds, 'sintesis_s': round(self.seconds, 2),
                'audio_s': round(wav_seconds(self.audio), 2) if self.audio else 0.0, 'memoria_pico_gb': self.peak_gb}


class QwenWorker:
    """Proceso persistente en .venv-tts: se arranca al primer uso, se reutiliza y se cierra tras inactividad."""

    def __init__(self, python=QWEN_PYTHON, model_dir=QWEN_DIR, idle=None, timeout=180):
        self.python, self.model_dir, self.timeout = Path(python), Path(model_dir), timeout
        self.idle = float(os.environ.get('NEXO_QWEN_INACTIVIDAD', '600')) if idle is None else idle
        self.lock = threading.Lock()
        self.process = self.timer = None
        self.starts = 0
        self.last_used = 0.0

    def available(self):
        return self.python.exists() and (self.model_dir / 'config.json').exists()

    @property
    def loaded(self):
        return self.process is not None and self.process.poll() is None

    def _read(self, timeout):
        ready, _, _ = select.select([self.process.stdout], [], [], timeout)
        if not ready:
            self._stop()
            raise VoiceError('Qwen3-TTS no respondió a tiempo.')
        line = self.process.stdout.readline()
        if not line:
            self._stop()
            raise VoiceError('El proceso de Qwen3-TTS se cerró inesperadamente.')
        return json.loads(line)

    def _start(self):
        self.process = subprocess.Popen(
            [str(self.python), str(ROOT / 'tts_qwen.py'), str(self.model_dir)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            env=dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1'))
        ready = self._read(self.timeout)
        if not ready.get('ready'):
            self._stop()
            raise VoiceError(ready.get('error') or 'Qwen3-TTS no arrancó.')
        self.starts += 1
        return ready['load_seconds']

    def synthesize(self, text, speaker):
        """Una síntesis a la vez; devuelve (wav, segundos de carga o None si ya estaba cargado, respuesta)."""
        with self.lock:
            if self.timer:
                self.timer.cancel()
            load_seconds = None if self.loaded else self._start()
            self.process.stdin.write(json.dumps({'text': text, 'speaker': speaker, 'language': 'spanish'}) + '\n')
            self.process.stdin.flush()
            reply = self._read(self.timeout)
            self.last_used = time.monotonic()
            if self.idle > 0:
                self.timer = threading.Timer(self.idle, self._unload_if_idle)
                self.timer.daemon = True
                self.timer.start()
        if not reply.get('ok'):
            raise VoiceError('Qwen3-TTS falló: ' + reply.get('error', 'error desconocido'))
        return base64.b64decode(reply['wav']), load_seconds, reply

    def _unload_if_idle(self):
        with self.lock:
            if time.monotonic() - self.last_used >= self.idle:
                self._stop()

    def _stop(self):
        process, self.process = self.process, None
        if process and process.poll() is None:
            try:
                process.stdin.close()
                process.wait(timeout=5)
            except Exception:
                process.kill()

    def unload(self):
        with self.lock:
            self._stop()


QWEN = QwenWorker()
atexit.register(QWEN.unload)

# Pictogramas, banderas, modificadores de tono, uniones (ZWJ), selectores de variación, etiquetas y teclas.
EMOJI = re.compile(
    '[\U0001F000-\U0001FAFF\u2600-\u27BF\u231A\u231B\u2328\u23CF\u23E9-\u23FA\u2B05-\u2B07\u2B1B\u2B1C'
    '\u2B50\u2B55\u3030\u303D\u3297\u3299\u200D\uFE00-\uFE0F\u20E3\U000E0020-\U000E007F]')
KEYCAP = re.compile('[0-9#*]\uFE0F?\u20E3')
PRESENTED_SYMBOL = re.compile('[^\\w\\s]\uFE0F')  # ©️, ↩️, ‼️…
EMOTICON = re.compile(r"(?<![\w:/])(?:[:;=][-^']?[)(\]\[DPpOo3|/\\*]+|<3+|\^_*\^|[xX]D+)(?!\w)")


def _with_pause(text):
    text = text.strip()
    return text + '.' if text and text[-1].isalnum() else text


def preparar_texto_para_voz(texto):
    """Versión para pronunciar: sin emojis, emoticonos ni marcas de formato. No cambia palabras.

    Devuelve '' si no queda nada pronunciable."""
    text = texto or ''
    text = KEYCAP.sub('', text)
    text = PRESENTED_SYMBOL.sub('', text)
    text = EMOJI.sub('', text)
    text = EMOTICON.sub('', text)
    text = re.sub(r'^[ \t]*(```|~~~).*$', '', text, flags=re.M)
    text = re.sub(r'!\[([^\]]*)\]\([^)]*\)', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^)\s]*(?:\s+"[^"]*")?\)', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\[[^\]]*\]', r'\1', text)
    text = re.sub(r'<((?:https?|mailto):[^>\s]+)>', r'\1', text)
    text = re.sub(r'^[ \t]*(?:[-*_][ \t]*){3,}$', '', text, flags=re.M)
    text = re.sub(r'(\*\*|__)(?=\S)(.+?)(?<=\S)\1', r'\2', text)
    text = re.sub(r'(?<![\w*])\*(?=\S)(.+?)(?<=\S)\*(?![\w*])', r'\1', text)
    text = re.sub(r'(?<!\w)_(?=\S)(.+?)(?<=\S)_(?!\w)', r'\1', text)
    text = re.sub(r'~~(.+?)~~', r'\1', text)
    text = re.sub(r'`+([^`]*)`+', r'\1', text)
    text = re.sub(r'^[ \t]*#{1,6}[ \t]+(.*?)[ \t#]*$', lambda m: _with_pause(m[1]), text, flags=re.M)
    text = re.sub(r'^[ \t]*(?:>[ \t]?)+', '', text, flags=re.M)
    text = re.sub(r'^[ \t]*(?:[-*+•]|\d+[.)])[ \t]+(.*)$', lambda m: _with_pause(m[1]), text, flags=re.M)
    text = re.sub(r'^[ \t]*\|?[ \t]*:?-{2,}:?[ \t]*(?:\|[ \t]*:?-{2,}:?[ \t]*)*\|?[ \t]*$', '', text, flags=re.M)
    text = re.sub(r'^[ \t]*\|(.*)\|[ \t]*$', lambda m: ', '.join(c.strip() for c in m[1].split('|')), text, flags=re.M)
    # Asteriscos sueltos que quedaron sin pareja; entre números se conservan (5*3).
    text = re.sub(r'(?<!\d)\*+|\*+(?!\d)', '', text)
    lines = []
    for line in text.split('\n'):
        line = re.sub(r'[ \t]+', ' ', line).strip()
        line = re.sub(r' +([,.;:!?…)\]])', r'\1', line)
        line = re.sub(r'([¿¡(\[]) +', r'\1', line)
        if re.search(r'\w', line):
            lines.append(line)
    return '\n'.join(lines)


def installed():
    voices = [{'id': key, 'nombre': label, 'motor': 'Piper'} for key, (label, _) in PIPER_VOICES.items()
              if (VOICES_DIR / f'{key}.onnx').exists() and (VOICES_DIR / f'{key}.onnx.json').exists()]
    voices.append({'id': PAULINA, 'nombre': 'Paulina · macOS', 'motor': 'macOS say'})
    if QWEN.available():
        voices += [{'id': QWEN_PREFIX + key, 'motor': 'Qwen3-TTS',
                    'nombre': f'{name} · Qwen3-TTS (voz nativa en {native}, habla español)'}
                   for key, (name, native) in QWEN_SPEAKERS.items()]
    return voices


def default_voice():
    wanted = os.environ.get('NEXO_VOZ', 'es_MX-claude-high')
    ids = [v['id'] for v in installed()]
    return wanted if wanted in ids else ids[0]


def _fix_config(path):
    # Algunas voces publicadas traen «PhonemeType.ESPEAK», que Piper no acepta.
    data = json.loads(path.read_text(encoding='utf-8'))
    if str(data.get('phoneme_type', 'espeak')).lower() not in {'espeak', 'text'}:
        data['phoneme_type'] = 'espeak'
        path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')


def download():
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    for key, (label, folder) in PIPER_VOICES.items():
        for ext in ('onnx', 'onnx.json'):
            target = VOICES_DIR / f'{key}.{ext}'
            if target.exists():
                continue
            print(f'Descargando voz {label} ({ext})…', flush=True)
            with urllib.request.urlopen(f'{VOICES_URL}{folder}{key}.{ext}', timeout=300) as response:
                target.with_suffix('.tmp').write_bytes(response.read())
            target.with_suffix('.tmp').rename(target)
        _fix_config(VOICES_DIR / f'{key}.onnx.json')


def _piper(key):
    with _lock:
        if key not in _cache:
            from piper import PiperVoice
            _fix_config(VOICES_DIR / f'{key}.onnx.json')
            _cache[key] = PiperVoice.load(str(VOICES_DIR / f'{key}.onnx'))
        return _cache[key]


def _paulina(text):
    with tempfile.TemporaryDirectory(prefix='asistente-voz-') as directory:
        aiff, wav = str(Path(directory) / 'voz.aiff'), str(Path(directory) / 'voz.wav')
        subprocess.run(['/usr/bin/say', '-v', 'Paulina', '-o', aiff], input=text, text=True, check=True, timeout=45, capture_output=True)
        subprocess.run(['/usr/bin/afconvert', '-f', 'WAVE', '-d', 'LEI16', aiff, wav], check=True, timeout=20, capture_output=True)
        return Path(wav).read_bytes()


def synthesize(text, voice=None, strict=False):
    """Sintetiza con la voz pedida y devuelve un Synthesis que dice qué motor se usó de verdad.

    strict=True (muestras): si la voz falla se lanza VoiceError, nunca se sustituye.
    strict=False (conversación): se recurre a Paulina, pero el resultado lo indica con su motivo."""
    # Único punto de limpieza: todos los motores reciben ya el texto preparado.
    text = preparar_texto_para_voz(text)
    requested = voice or default_voice()
    if not text:
        return Synthesis(None, requested)
    try:
        return _synthesize_with(requested, text)
    except Exception as error:
        reason = str(error) if isinstance(error, VoiceError) else f'{error.__class__.__name__}: {error}'
        if strict or requested == PAULINA:
            raise VoiceError(reason) from error
        result = _synthesize_with(PAULINA, text)
        result.requested, result.fallback, result.reason = requested, True, reason
        return result


def _synthesize_with(voice, text):
    start = time.perf_counter()
    if voice in PIPER_VOICES:
        if not (VOICES_DIR / f'{voice}.onnx').exists():
            raise VoiceError(f'La voz Piper {voice} no está descargada.')
        # Cargar fuera del «with»: si falla dentro, wave oculta el error real con «channels not specified».
        model = _piper(voice)
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav:
            model.synthesize_wav(text, wav)
        return Synthesis(buffer.getvalue(), voice, 'Piper', PIPER_VOICES[voice][0].split(' ·')[0],
                         seconds=time.perf_counter() - start)
    if voice == PAULINA:
        return Synthesis(_paulina(text), voice, 'macOS say', 'Paulina', seconds=time.perf_counter() - start)
    if voice.startswith(QWEN_PREFIX) and voice[len(QWEN_PREFIX):] in QWEN_SPEAKERS:
        speaker = voice[len(QWEN_PREFIX):]
        if not QWEN.available():
            raise VoiceError('Qwen3-TTS no está instalado. ' + QWEN_HINT)
        audio, load_seconds, reply = QWEN.synthesize(text, speaker)
        return Synthesis(audio, voice, 'Qwen3-TTS', QWEN_SPEAKERS[speaker][0], load_seconds=load_seconds,
                         seconds=reply['seconds'], peak_gb=reply.get('peak_gb'))
    raise VoiceError(f'La voz «{voice}» no existe.')
