"""Mide por separado: espera de fin de turno, transcripción, generación de Qwen y síntesis de voz.

Usa frases sintetizadas con la voz de macOS y las reproduce a velocidad real (no usa tu micrófono
ni guarda nada en la memoria).  Uso: .venv/bin/python medir.py [--sin-lm]
"""
import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from tests.test_audio_real import silence, speech
from transcripcion import Transcriber
from turnos import SileroVAD, TurnConfig, TurnDetector

CHUNK = 1536
CASES = [
    ('frase completa', ['¿Me recuerdas qué tengo que comprar para la cena de mañana?']),
    ('pausa dentro de la frase', ['Quiero llamar a mi hermana y', 'contarle lo del viaje.']),
    ('nombre propio', ['Mi amiga Ximena Castañeda llega el jueves a Bogotá.']),
]


def simulate(detector, parts):
    audio, speech_end = [], 0
    for i, text in enumerate(parts):
        if i:
            audio.append(silence(1.0))
        audio.append(speech(text))
        speech_end = sum(len(a) for a in audio)
    audio = np.concatenate(audio + [silence(4)])
    start = time.perf_counter()
    for i in range(0, len(audio), CHUNK):
        # Reproduce a velocidad real para que la transcripción compita con el tiempo como en vivo.
        delay = start + i / 16000 - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        event = detector.feed(audio[i:i + CHUNK])
        if event['turn']:
            wall = time.perf_counter() - (start + speech_end / 16000)
            return event['turn'], wall
    return None, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sin-lm', action='store_true', help='No consultar LM Studio')
    parser.add_argument('--repeticiones', type=int, default=2)
    args = parser.parse_args()
    asr = Transcriber()
    start = time.perf_counter()
    asr.load()
    print(f'Carga y calentamiento de {asr.name}: {time.perf_counter() - start:.2f} s')
    vad = SileroVAD()
    frame = np.zeros(512, np.float32)
    start = time.perf_counter()
    for _ in range(300):
        vad(frame)
    print(f'Silero VAD: {(time.perf_counter() - start) / 300 * 1000:.2f} ms por ventana de 32 ms')
    executor = ThreadPoolExecutor(max_workers=1)
    config = TurnConfig.create()
    print(f'Pausa corta {config.pausa_corta} s · pausa larga {config.pausa_larga} s\n')
    for name, parts in CASES:
        waits, asr_times = [], []
        for _ in range(args.repeticiones):
            detector = TurnDetector(vad, lambda clip: asr.transcribe(clip, ['Nexo']), config, executor)
            turn, wall = simulate(detector, parts)
            if not turn:
                print(f'{name}: no se detectó turno')
                break
            waits.append(wall)
            asr_times.append(turn['timings']['transcripcion'])
        if waits:
            print(f'{name}: «{turn["transcript"].text}» (confianza {turn["transcript"].confidence})')
            print(f'  fin de turno tras la última voz: {statistics.median(waits):.2f} s · '
                  f'transcripción: {statistics.median(asr_times):.2f} s (solapada con la pausa)')
    if not args.sin_lm:
        from servidor import LMStudioError, make_audio
        from main import DEFAULT_MODEL, request_json
        from personalidad import MAX_TOKENS, TEMPERATURE, system_prompt
        try:
            for question in ['¿Qué hora es en Bogotá si en Madrid son las nueve?', 'Estoy aprendiendo Python.']:
                start = time.perf_counter()
                result = request_json('/chat/completions', {'model': DEFAULT_MODEL, 'messages': [
                    {'role': 'system', 'content': system_prompt({})}, {'role': 'user', 'content': question}],
                    'max_tokens': MAX_TOKENS, 'temperature': TEMPERATURE})
                generation = time.perf_counter() - start
                text = result['choices'][0]['message']['content']
                tokens = result.get('usage', {}).get('completion_tokens', '?')
                print(f'\nGeneración Qwen «{question}»: {generation:.2f} s, {tokens} tokens')
                start = time.perf_counter()
                make_audio(text)
                print(f'Síntesis de voz: {time.perf_counter() - start:.2f} s')
        except (OSError, LMStudioError, KeyError) as error:
            print('\nLM Studio no disponible:', error)
    executor.shutdown()


if __name__ == '__main__':
    main()
