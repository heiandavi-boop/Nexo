"""Transcripción local en español con Whisper large-v3-turbo sobre MLX (GPU de Apple Silicon).

Nunca reescribe palabras: solo ajusta espacios, mayúsculas iniciales y signos de puntuación.
"""
import os
import re
import threading
import time
import unicodedata
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / 'modelos'
MODEL_DIR = MODELS / 'whisper-large-v3-turbo'
MODEL_REPO = 'mlx-community/whisper-large-v3-turbo'
FALLBACK_DIR = MODELS / 'whisper-base'
VAD_PATH = MODELS / 'silero_vad.onnx'
VAD_URL = 'https://github.com/snakers4/silero-vad/raw/v6.2/src/silero_vad/data/silero_vad.onnx'
VAD_SHA256 = '1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3'
RATE = 16000
DOWNLOAD_HINT = 'Ejecuta una vez con internet: .venv/bin/python escuchar.py --descargar'

WORD_MIN = 0.4          # probabilidad por debajo de la cual una palabra se considera dudosa
LOW_LOGPROB = -1.0      # media de log-probabilidad que indica transcripción poco fiable
MEDIUM_LOGPROB = -0.6
MAX_COMPRESSION = 2.4   # valores mayores suelen indicar bucles de repetición

# Frases que Whisper inventa sobre ruido o silencio (comparadas sin tildes ni signos).
HALLUCINATIONS = re.compile(
    r'(subtitulos?( realizados?)? (por|de)\b.*|.*\bamara org\b.*|gracias por ver( el video| este video)?'
    r'|suscribete.*|gracias por (su|tu) atencion|musica|aplausos|risas)'
)


class ModelMissing(RuntimeError):
    pass


def normalize(text):
    text = unicodedata.normalize('NFKD', text.lower())
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return ' '.join(re.findall(r'\w+', text))


def _capitalize(sentence):
    for i, char in enumerate(sentence):
        if char.isalpha():
            return sentence[:i] + char.upper() + sentence[i + 1:]
        if char not in '¿¡"«“(\'':
            return sentence
    return sentence


def clean_text(raw):
    """Mejora legibilidad sin añadir, quitar ni cambiar palabras."""
    text = re.sub(r'\s+', ' ', raw).strip()
    text = re.sub(r'\s+([,.;:?!…])', r'\1', text)
    text = re.sub(r'([¿¡])\s+', r'\1', text)
    sentences = []
    # Los puntos suspensivos no cierran oración: «y... contarle» sigue en minúscula.
    for sentence in re.split(r'(?<=[?!])\s+|(?<=(?<!\.)\.)\s+', text):
        if not sentence:
            continue
        for close, opening in (('?', '¿'), ('!', '¡')):
            if sentence.endswith(close) and opening not in sentence:
                # En español la pregunta suele empezar tras la última coma: «Oye, ¿vienes?».
                comma = sentence.rfind(', ')
                sentence = sentence[:comma + 2] + opening + sentence[comma + 2:] if comma >= 0 else opening + sentence
        sentences.append(_capitalize(sentence))
    text = ' '.join(sentences)
    if text and text[-1].isalnum():
        text += '.'
    return text


def hint_prompt(hints):
    names = []
    for hint in hints or ():
        name = re.sub(r"[^\w\s'-]", '', str(hint)).strip()[:40]
        if name and name.lower() not in {n.lower() for n in names}:
            names.append(name)
    if not names:
        return ''
    return 'Conversación en español. Nombres: ' + ', '.join(names[:10]) + '.'


@dataclass
class Transcript:
    text: str = ''
    raw: str = ''
    words: list = field(default_factory=list)
    doubtful: list = field(default_factory=list)
    confidence: str = 'alta'
    discarded: str = ''
    seconds: float = 0.0

    def public(self):
        return {'text': self.text, 'doubtful': self.doubtful, 'confidence': self.confidence}


def build_transcript(segments, prompt='', seconds=0.0):
    """Convierte segmentos de Whisper en texto limpio y una estimación de confianza."""
    segments = [s for s in segments if s.get('text', '').strip()]
    raw = re.sub(r'\s+', ' ', ' '.join(s['text'].strip() for s in segments)).strip()
    words = []
    for segment in segments:
        for word in segment.get('words') or []:
            token = word['word'].strip()
            if re.search(r'\w', token):
                words.append((token, float(word.get('probability', 1.0))))
    result = Transcript(raw=raw, words=words, seconds=seconds)
    norm = normalize(raw)
    if not norm:
        result.discarded = 'sin palabras'
    elif HALLUCINATIONS.fullmatch(norm):
        result.discarded = 'frase típica de ruido'
    elif prompt and len(norm.split()) >= 3 and norm in normalize(prompt):
        result.discarded = 'eco de las pistas de nombres'
    elif all(s.get('no_speech_prob', 0) > 0.6 and s.get('avg_logprob', 0) < LOW_LOGPROB for s in segments):
        result.discarded = 'sin voz clara'
    if result.discarded:
        return result
    result.text = clean_text(raw)
    result.doubtful = [re.sub(r'^\W+|\W+$', '', w) for w, p in words if p < WORD_MIN]
    lengths = [max(1, len(s['text'])) for s in segments]
    logprob = sum(s.get('avg_logprob', 0) * n for s, n in zip(segments, lengths)) / sum(lengths)
    compression = max(s.get('compression_ratio', 1) for s in segments)
    share = len(result.doubtful) / len(words) if words else 0
    if compression > MAX_COMPRESSION or logprob < LOW_LOGPROB or share > 0.4:
        result.confidence = 'baja'
    elif result.doubtful or logprob < MEDIUM_LOGPROB:
        result.confidence = 'media'
    return result


def load_wav(path):
    """Lee un WAV PCM de 16 bits y lo devuelve mono a 16 kHz en float32."""
    with wave.open(str(path)) as wav:
        if wav.getsampwidth() != 2:
            raise ValueError('Usa WAV PCM de 16 bits.')
        rate, channels = wav.getframerate(), wav.getnchannels()
        audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(np.float32) / 32768
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != RATE:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(rate, RATE)
        audio = resample_poly(audio, RATE // g, rate // g).astype(np.float32)
    return audio


class Transcriber:
    """Carga perezosa y uso exclusivamente local de los archivos en modelos/."""

    def __init__(self, model_dir=MODEL_DIR, fallback_dir=FALLBACK_DIR, word_confidence=None):
        self.model_dir, self.fallback_dir = Path(model_dir), Path(fallback_dir)
        # La confianza por palabra añade ~0,3 s por turno; NEXO_CONFIANZA_PALABRAS=0 la desactiva.
        self.word_confidence = os.environ.get('NEXO_CONFIANZA_PALABRAS', '1') != '0' if word_confidence is None else word_confidence
        self.lock = threading.Lock()
        self.engine = None
        self.error = ''

    def available(self):
        return (self.model_dir / 'weights.safetensors').exists() or (self.fallback_dir / 'model.bin').exists()

    @property
    def name(self):
        return {'mlx': 'Whisper large-v3-turbo (MLX)', 'base': 'Whisper base (respaldo, menos preciso)'}.get(self.engine, 'sin cargar')

    def load(self):
        with self.lock:
            if self.engine:
                return
            # Impide descargas silenciosas durante una conversación.
            os.environ['HF_HUB_OFFLINE'] = '1'
            try:
                if (self.model_dir / 'weights.safetensors').exists():
                    import mlx_whisper
                    self._mlx = mlx_whisper
                    self.engine = 'mlx'
                    self._run(np.zeros(RATE, dtype=np.float32), '')
                elif (self.fallback_dir / 'model.bin').exists():
                    from faster_whisper import WhisperModel
                    self._base = WhisperModel(str(self.fallback_dir), device='cpu', compute_type='int8', local_files_only=True)
                    self.engine = 'base'
                else:
                    raise ModelMissing('Falta el modelo de reconocimiento de voz. ' + DOWNLOAD_HINT)
                self.error = ''
            except Exception as error:
                self.engine = None
                self.error = str(error)
                raise

    def _run(self, audio, prompt):
        if self.engine == 'mlx':
            result = self._mlx.transcribe(
                audio, path_or_hf_repo=str(self.model_dir), language='es', task='transcribe', verbose=None,
                temperature=(0.0, 0.2, 0.4), condition_on_previous_text=False,
                initial_prompt=prompt or None, word_timestamps=self.word_confidence)
            return result['segments']
        segments, _ = self._base.transcribe(
            audio, language='es', beam_size=5, initial_prompt=prompt or None,
            condition_on_previous_text=False, word_timestamps=self.word_confidence, vad_filter=False)
        return [{'text': s.text, 'avg_logprob': s.avg_logprob, 'no_speech_prob': s.no_speech_prob,
                 'compression_ratio': s.compression_ratio,
                 'words': [{'word': w.word, 'probability': w.probability} for w in s.words or []]} for s in segments]

    def transcribe(self, audio, hints=()):
        self.load()
        audio = np.asarray(audio, dtype=np.float32)
        prompt = hint_prompt(hints)
        start = time.perf_counter()
        segments = self._run(audio, prompt)
        return build_transcript(segments, prompt, time.perf_counter() - start)
