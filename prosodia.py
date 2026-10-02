"""Señales prosódicas locales y temporales; no clasifica ni almacena emociones."""
from collections import deque
import threading
import time

import numpy as np

RATE = 16000
FRAME = int(RATE * 0.03)
HOP = int(RATE * 0.01)
MIN_DURATION = 0.8
MIN_SPEECH_SECONDS = 0.55
MAX_PITCH_FRAMES = 160
BASELINE_TURNS = 3
BASELINE_WINDOW = 12


def _pitch_features(windows, active):
    selected = active[::3][:MAX_PITCH_FRAMES]
    if len(selected) < 4:
        return None, None, 0.0
    frames = windows[selected].astype(np.float32, copy=True)
    frames -= np.mean(frames, axis=1, keepdims=True)
    frames *= np.hanning(frames.shape[1]).astype(np.float32)
    spectrum = np.fft.rfft(frames, n=1024, axis=1)
    correlation = np.fft.irfft(spectrum * np.conjugate(spectrum), n=1024, axis=1)
    zero_lag = np.maximum(correlation[:, 0], 1e-9)
    normalized = correlation / zero_lag[:, None]
    low_lag, high_lag = int(RATE / 350), int(RATE / 75)
    region = normalized[:, low_lag:high_lag + 1]
    offsets = np.argmax(region, axis=1)
    strength = region[np.arange(len(region)), offsets]
    reliable = strength >= 0.3
    if np.count_nonzero(reliable) < 3:
        return None, None, float(np.mean(reliable))
    pitches = RATE / (low_lag + offsets[reliable])
    pitch_median = float(np.median(pitches))
    pitch_variation = float(np.std(pitches) / max(pitch_median, 1.0))
    return pitch_median, pitch_variation, float(np.mean(reliable))


def _energy_peaks(envelope, active):
    if len(active) < 9:
        return 0
    smooth = np.convolve(envelope, np.ones(5, dtype=np.float32) / 5, mode='same')
    voice = smooth[active]
    spread = float(np.percentile(voice, 90) - np.percentile(voice, 10))
    threshold = float(np.median(voice)) + spread * 0.2
    candidates = np.flatnonzero(
        (smooth[1:-1] >= smooth[:-2])
        & (smooth[1:-1] > smooth[2:])
        & (smooth[1:-1] >= threshold)
    ) + 1
    selected = []
    for index in candidates:
        if active[0] <= index <= active[-1] and (not selected or index - selected[-1] >= 8):
            selected.append(int(index))
    return len(selected)


class ProsodyAnalyzer:
    """Compara cada turno con una baseline robusta de esta ejecución, sin persistirla."""

    def __init__(self):
        self._lock = threading.Lock()
        self._baseline = {
            'energy': deque(maxlen=BASELINE_WINDOW),
            'rate': deque(maxlen=BASELINE_WINDOW),
            'pitch': deque(maxlen=BASELINE_WINDOW),
        }

    def analyze(self, audio):
        started = time.perf_counter()
        signal = np.asarray(audio, dtype=np.float32).reshape(-1)
        duration = len(signal) / RATE
        result = {
            'usable': False,
            'duration': round(duration, 3),
            'candidate': None,
            'confidence': 0.0,
            'level': 'baja',
            'context': None,
        }
        if duration < MIN_DURATION or not np.all(np.isfinite(signal)):
            result['analysis_ms'] = round((time.perf_counter() - started) * 1000, 3)
            return result

        windows = np.lib.stride_tricks.sliding_window_view(signal, FRAME)[::HOP]
        envelope = np.sqrt(np.mean(windows * windows, axis=1, dtype=np.float64)).astype(np.float32)
        floor = float(np.percentile(envelope, 20))
        upper = float(np.percentile(envelope, 75))
        noise_threshold = floor * 2.5 if floor < upper * 0.6 else upper * 0.25
        active_threshold = max(noise_threshold, upper * 0.12, 0.0005)
        active_mask = envelope >= active_threshold
        active = np.flatnonzero(active_mask)
        speech_seconds = len(active) * HOP / RATE
        active_rms = float(np.median(envelope[active])) if len(active) else 0.0
        result.update({
            'speech_seconds': round(speech_seconds, 3),
            'silence_ratio': round(1.0 - len(active) / max(len(envelope), 1), 3),
            'pauses': self._pause_count(active_mask),
        })
        if speech_seconds < MIN_SPEECH_SECONDS or active_rms < 0.001:
            result['analysis_ms'] = round((time.perf_counter() - started) * 1000, 3)
            return result

        pitch, pitch_variation, voiced_pitch_ratio = _pitch_features(windows, active)
        peak_count = _energy_peaks(envelope, active)
        speech_rate = peak_count / max(speech_seconds, 0.1)
        result.update({
            'usable': True,
            'energy_rms': active_rms,
            'pitch_hz': pitch,
            'pitch_variation': pitch_variation,
            'pitch_confidence': round(voiced_pitch_ratio, 3),
            'speech_rate': round(speech_rate, 2),
        })
        result['analysis_ms'] = round((time.perf_counter() - started) * 1000, 3)
        return result

    def assess(self, features):
        """Compara y actualiza solo cuando el turno ya fue aceptado por Whisper."""
        result = dict(features)
        if not result.get('usable') or result.get('pitch_hz') is None or result.get('pitch_confidence', 0) < 0.2:
            return result
        with self._lock:
            self._compare_and_update(result['energy_rms'], result['speech_rate'], result['pitch_hz'], result)
        return result

    @staticmethod
    def _pause_count(active_mask):
        inactive = ~active_mask
        edges = np.diff(np.pad(inactive.astype(np.int8), (1, 1)))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        return int(np.count_nonzero((ends - starts) * HOP / RATE >= 0.12))

    def _compare_and_update(self, energy, rate, pitch, result):
        observations = self._baseline
        ready = min(len(values) for values in observations.values()) >= BASELINE_TURNS
        candidate = None
        confidence = 0.0
        if ready:
            base_energy = float(np.median(observations['energy']))
            base_rate = float(np.median(observations['rate']))
            base_pitch = float(np.median(observations['pitch']))
            ratios = (energy / max(base_energy, 1e-6),
                      rate / max(base_rate, 0.5),
                      pitch / max(base_pitch, 1.0))
            elevated = sum((ratios[0] >= 1.35, ratios[1] >= 1.2, ratios[2] >= 1.12))
            reduced = sum((ratios[0] <= 0.72, ratios[1] <= 0.78, ratios[2] <= 0.9))
            if elevated >= 2:
                candidate = 'activación vocal superior a la habitual'
                confidence = 0.78 if elevated == 3 else 0.6
            elif reduced >= 2:
                candidate = 'activación vocal inferior a la habitual'
                confidence = 0.72 if reduced == 3 else 0.58
            if candidate:
                result['candidate'] = candidate
                result['confidence'] = confidence
                result['level'] = 'alta' if confidence >= 0.75 else 'media'
                result['context'] = (
                    'CONTEXTO VOCAL DEL TURNO (inferencia acústica incierta):\n'
                    f'- Señal relativa: {candidate}.\n'
                    f'- Confianza: {result["level"]}.\n'
                    'Úsala solo para ajustar con sutileza el tono, la brevedad y la insistencia. '
                    'No diagnostiques ni menciones una emoción salvo que sea claramente útil. '
                    'No conviertas esta señal temporal en un dato del usuario.'
                )

                # La ventana móvil adapta el patrón incluso ante cambios sostenidos.
        observations['energy'].append(float(energy))
        observations['rate'].append(float(rate))
        observations['pitch'].append(float(pitch))