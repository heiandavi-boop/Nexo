"""Detección de voz con Silero VAD y fin de turno adaptable.

El silencio solo indica que la persona dejó de sonar; no que terminó la idea. Por eso:
1. Silero VAD decide qué fragmentos contienen voz (no el volumen).
2. Tras una pausa breve se transcribe de forma especulativa mientras se sigue escuchando.
3. Si el texto termina como una frase completa, basta la pausa corta; si termina en
   «y», «que», «de la», «eh»…, se espera la pausa larga. Si la persona vuelve a hablar,
   esa transcripción se descarta y el turno continúa.
"""
import os
import re
import time
import uuid
from collections import deque
from concurrent.futures import Future
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from transcripcion import DOWNLOAD_HINT, RATE, VAD_PATH, ModelMissing

FRAME = 512  # 32 ms a 16 kHz: tamaño de ventana que exige Silero VAD
RESUME_FRAMES = 2  # tras una pausa, 64 ms de voz para considerar que se reanudó

# Palabras con las que una frase en español rara vez termina. Con tilde (qué, él, tú, sí, mí) sí pueden cerrar.
CONTINUATION = set('''
a al ante bajo con contra de del desde durante en entre hacia hasta mediante para por según sin sobre tras
el la lo los las un una unos unas mi mis tu tus su sus nuestro nuestra nuestros nuestras
y e o u ni pero sino que porque aunque como cuando donde si pues mientras entonces
me te se le les nos muy tan cuyo cuya
eh em emm mm mmm este esto ehh
'''.split())
CONTINUATION_PAIRS = {'o sea', 'es decir', 'por ejemplo', 'lo que', 'la que', 'el que', 'tal vez', 'a lo'}


def looks_incomplete(transcript):
    """Pista lingüística barata. No entiende la oración; solo decide cuánto margen dar."""
    text = getattr(transcript, 'raw', transcript).strip()
    if not text or text.endswith(('...', '…', ',', ';', ':', '-')):
        return True
    if text.endswith(('?', '!')):
        return False
    words = re.findall(r'\w+', text.lower())
    if not words:
        return True
    if words[-1] in CONTINUATION or ' '.join(words[-2:]) in CONTINUATION_PAIRS:
        return True
    probabilities = getattr(transcript, 'words', None)
    # Una última palabra muy dudosa suele estar cortada: mejor esperar un poco más.
    return bool(probabilities) and probabilities[-1][1] < 0.3


LIMITS = {
    'umbral_inicio': (0.2, 0.95), 'umbral_fin': (0.05, 0.9), 'inicio_minimo': (0.03, 0.5),
    'prebuffer': (0.1, 1.5), 'cola': (0.1, 1.0), 'pausa_corta': (0.3, 2.5), 'pausa_larga': (0.5, 5.0),
    'revisar_tras': (0.1, 1.0), 'voz_minima': (0.1, 1.5), 'turno_maximo': (5.0, 60.0),
}


@dataclass
class TurnConfig:
    umbral_inicio: float = 0.5    # probabilidad de Silero para empezar un turno
    umbral_fin: float = 0.35      # histéresis: por debajo de esto cuenta como pausa
    inicio_minimo: float = 0.096  # voz continua necesaria para abrir un turno
    prebuffer: float = 0.5        # audio previo conservado para no cortar la primera sílaba
    cola: float = 0.3             # audio tras la última voz para no cortar la última sílaba
    pausa_corta: float = 0.7      # silencio si la frase parece terminada
    pausa_larga: float = 1.6      # silencio si la frase parece continuar
    revisar_tras: float = 0.25    # silencio tras el que empieza la transcripción especulativa
    voz_minima: float = 0.25      # menos voz que esto se considera ruido
    turno_maximo: float = 60.0

    @classmethod
    def create(cls, data=None):
        """Valores por defecto, luego variables NEXO_*, luego ajustes de la página; todos acotados."""
        values = asdict(cls())
        for key in values:
            env = os.environ.get('NEXO_' + key.upper())
            if env:
                values[key] = env
        for key, value in (data or {}).items():
            if key in values:
                values[key] = value
        for key, (low, high) in LIMITS.items():
            try:
                number = float(values[key])
            except (TypeError, ValueError):
                number = getattr(cls, key)
            values[key] = min(high, max(low, number)) if number == number else getattr(cls, key)
        values['umbral_fin'] = min(values['umbral_fin'], values['umbral_inicio'])
        values['pausa_larga'] = max(values['pausa_larga'], values['pausa_corta'])
        values['revisar_tras'] = min(values['revisar_tras'], values['pausa_corta'])
        return cls(**values)


class SileroVAD:
    def __init__(self, path=VAD_PATH):
        if not Path(path).exists():
            raise ModelMissing('Falta el detector de voz Silero VAD. ' + DOWNLOAD_HINT)
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.inter_op_num_threads = options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        self.rate = np.array(RATE, dtype=np.int64)
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros(64, dtype=np.float32)

    def __call__(self, frame):
        x = np.concatenate([self.context, frame])[None, :].astype(np.float32)
        output, self.state = self.session.run(None, {'input': x, 'state': self.state, 'sr': self.rate})
        self.context = x[0, -64:]
        return float(output[0, 0])


class InlineExecutor:
    """Ejecuta en el mismo hilo; útil para pruebas y uso por archivo."""

    def submit(self, fn, *args):
        future = Future()
        try:
            future.set_result(fn(*args))
        except Exception as error:
            future.set_exception(error)
        return future


class TurnDetector:
    def __init__(self, vad, transcribe, config=None, executor=None, analysis=None, analysis_executor=None):
        self.vad, self.transcribe = vad, transcribe
        self.analysis, self.analysis_executor = analysis, analysis_executor
        self.config = config or TurnConfig()
        self.executor = executor or InlineExecutor()
        self.dt = FRAME / RATE
        self.reset()

    def frames(self, seconds):
        return max(1, int(round(seconds / self.dt)))

    def reset(self):
        self.vad.reset()
        self.leftover = np.zeros(0, dtype=np.float32)
        self.prob = 0.0
        self._clear()

    def _clear(self, cancel_analysis=True):
        if getattr(self, 'job', None):
            self.job['future'].cancel()
            analysis_future = self.job.get('analysis_future')
            if cancel_analysis and analysis_future:
                analysis_future.cancel()
        self.pre = deque(maxlen=self.frames(self.config.prebuffer))
        self.audio = []
        self.active = self.speaking = self.forced = False
        self.onset = self.resume = self.speech_frames = self.last_speech = self.silence = 0
        self.job = None

    def cancel(self):
        """Descarta el turno especulativo y cancela trabajos pendientes."""
        self._clear()

    @property
    def state(self):
        if not self.active:
            return 'escuchando'
        if self.speaking or not self.silence:
            return 'voz'
        if self.job and not self.job['future'].done() and self.silence * self.dt >= self.config.pausa_corta:
            return 'transcribiendo'
        return 'pausa'

    def feed(self, samples):
        data = np.concatenate([self.leftover, np.asarray(samples, dtype=np.float32)])
        count = len(data) // FRAME
        self.leftover = data[count * FRAME:]
        event = {'turn': None, 'discarded': None, 'error': None}
        for i in range(count):
            result = self._step(data[i * FRAME:(i + 1) * FRAME])
            if result:
                event.update(result)
                if not result.get('discarded'):
                    self.leftover = np.zeros(0, dtype=np.float32)
                    break
        if not count and self.active:
            # Revisa transcripciones terminadas aunque llegue menos de un bloque.
            event.update(self._decide() or {})
        event.update(state=self.state, prob=round(self.prob, 3), silence=round(self.silence * self.dt, 2),
                     incomplete=bool(self.job and self.job['future'].done() and self.job.get('incomplete')))
        return event

    def _step(self, frame):
        config = self.config
        if self.forced:
            self.silence += 1
            return self._decide()
        p = self.prob = self.vad(frame)
        if not self.active:
            self.pre.append(frame)
            self.onset = self.onset + 1 if p >= config.umbral_inicio else 0
            if self.onset >= self.frames(config.inicio_minimo):
                self.active = self.speaking = True
                self.audio = list(self.pre)
                self.pre.clear()
                self.speech_frames = self.onset
                self.last_speech = len(self.audio)
            return None
        self.audio.append(frame)
        if len(self.audio) >= self.frames(config.turno_maximo):
            self.forced, self.speaking = True, False
            return self._decide()
        if self.speaking:
            voice = p >= config.umbral_fin
        else:
            self.resume = self.resume + 1 if p >= config.umbral_inicio else 0
            voice = self.resume >= RESUME_FRAMES
        if voice:
            self.speaking, self.resume, self.silence = True, 0, 0
            self.speech_frames += 1
            self.last_speech = len(self.audio)
            return None
        self.speaking = False
        self.silence += 1
        return self._decide()

    def _timed(self, clip):
        start = time.perf_counter()
        transcript = self.transcribe(clip)
        return transcript, time.perf_counter() - start

    def _decide(self):
        config = self.config
        silence = self.silence * self.dt
        if self.speech_frames * self.dt < config.voz_minima:
            if silence >= config.pausa_corta:
                self._clear()
                return {'discarded': 'ruido breve'}
            return None
        version = self.last_speech
        if (silence >= config.revisar_tras or self.forced) and (self.job is None or self.job['version'] != version):
            if self.job:
                self.job['future'].cancel()
                analysis_future = self.job.get('analysis_future')
                if analysis_future:
                    analysis_future.cancel()
            end = min(len(self.audio), version + self.frames(config.cola))
            clip = np.concatenate(self.audio[:end])
            self.job = {'version': version, 'end': end, 'future': self.executor.submit(self._timed, clip),
                        'analysis_future': None}
            if self.analysis and self.analysis_executor:
                try:
                    self.job['analysis_future'] = self.analysis_executor.submit(self.analysis, clip)
                except Exception:
                    pass
        if not self.job or self.job['version'] != version or not self.job['future'].done():
            return None
        try:
            transcript, seconds = self.job['future'].result()
        except Exception as error:
            self._clear()
            return {'error': str(error) or error.__class__.__name__}
        if transcript.discarded or not transcript.text:
            reason = transcript.discarded or 'sin palabras'
            self._clear()
            return {'discarded': reason}
        incomplete = self.job['incomplete'] = looks_incomplete(transcript)
        if not self.forced and silence < (config.pausa_larga if incomplete else config.pausa_corta):
            return None
        voice_analysis = self.job.get('analysis_future')
        turn = {
            'id': uuid.uuid4().hex,
            'transcript': transcript,
            'incomplete': incomplete,
            'audio': self.job['end'] * self.dt,
            'timings': {'voz': round(self.speech_frames * self.dt, 2), 'fin_de_turno': round(silence, 2),
                        'transcripcion': round(seconds, 2)},
            'voice_analysis': voice_analysis,
        }
        self._clear(cancel_analysis=False)
        return {'turn': turn}
