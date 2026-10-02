"""Voz de Nexo: Piper (neuronal, local) con la voz Paulina de macOS como respaldo."""
import io
import json
import os
import re
import subprocess
import tempfile
import threading
import urllib.request
import wave
from pathlib import Path

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
    voices = [{'id': key, 'nombre': label} for key, (label, _) in PIPER_VOICES.items()
              if (VOICES_DIR / f'{key}.onnx').exists() and (VOICES_DIR / f'{key}.onnx.json').exists()]
    return voices + [{'id': PAULINA, 'nombre': 'Paulina · macOS'}]


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


def synthesize(text, voice=None):
    """Devuelve WAV PCM16, o None si no hay nada pronunciable. Si la voz Piper falla o no existe, usa Paulina."""
    # Único punto de limpieza: todos los motores reciben ya el texto preparado.
    text = preparar_texto_para_voz(text)
    if not text:
        return None
    voice = voice or default_voice()
    if voice in PIPER_VOICES and (VOICES_DIR / f'{voice}.onnx').exists():
        try:
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wav:
                _piper(voice).synthesize_wav(text, wav)
            return buffer.getvalue()
        except Exception:
            pass
    return _paulina(text)
