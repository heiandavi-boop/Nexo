"""Mide la latencia de una respuesta hablada, etapa por etapa, sin tocar la memoria real.

Uso:
  .venv/bin/python medir_latencia.py                       # voz Piper Claude
  .venv/bin/python medir_latencia.py --voz qwen3-serena    # otra voz
  .venv/bin/python medir_latencia.py --audio               # incluye transcripción de la pregunta (voz de macOS)

Métricas (segundos, medidas en el servidor; «t0» = inicio de la respuesta, justo tras recibir el turno):
  transcripcion   Whisper sobre la pregunta sintetizada (solo con --audio)
  first_token     t0 → primer texto útil del modelo
  first_sentence  t0 → primera frase lista para la voz
  first_audio     t0 → primer audio listo para enviar al navegador
  generacion      t0 → respuesta completa del modelo
  sintesis        suma de tiempos de síntesis de todas las frases
El navegador añade su propio tramo (recepción → sonido); ver «audio_play_start» en la página.
"""
import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

import servidor
from conversacion import ConversationState
from memoria import Memoria

QUERIES = [
    ('A', '¿Cuántos años tengo?'),
    ('B', '¿Cuántos días tiene un año?'),
    ('C', 'Explícame cómo se forman las nubes.'),
    ('D', 'Háblame sobre inteligencia artificial.'),
]


def measure(question, voice, transcription=None):
    marks, events, tokens = {}, [], [0]
    real_stream, real_speak = servidor.stream_json, servidor.make_audio
    start = time.perf_counter()

    def stream(path, payload, timeout=120):
        for event in real_stream(path, payload, timeout):
            delta = (event.get('choices') or [{}])[0].get('delta', {}).get('content')
            if delta:
                tokens[0] += 1
                marks.setdefault('first_token', time.perf_counter() - start)
            yield event

    def speak(text, voice_id=None, strict=False):
        marks.setdefault('first_sentence', time.perf_counter() - start)
        return real_speak(text, voice_id, strict)

    def emit(event):
        if event['type'] == 'audio':
            marks.setdefault('first_audio', time.perf_counter() - start)
        events.append(event)

    with mock.patch.object(servidor, 'stream_json', stream):
        servidor.talk_stream({'text': question, 'voice': voice}, emit, speak=speak)
    done = events[-1]
    timings = done.get('timings', {})
    return {
        'transcripcion': transcription,
        'first_token': marks.get('first_token'), 'first_sentence': marks.get('first_sentence'),
        'first_audio': marks.get('first_audio'), 'generacion': timings.get('generacion'),
        'sintesis': timings.get('sintesis_voz'), 'caracteres': len(done.get('answer', '')),
        'fragmentos_stream': tokens[0], 'frases': sum(e['type'] == 'audio' for e in events),
        'respuesta': done.get('answer') or done.get('error', ''),
    }


def transcribe_spoken(question):
    import subprocess
    from tests.test_audio_real import speech
    from transcripcion import Transcriber
    asr = getattr(transcribe_spoken, 'asr', None) or Transcriber()
    transcribe_spoken.asr = asr
    audio = speech(question)
    t = time.perf_counter()
    text = asr.transcribe(audio).text
    return text, time.perf_counter() - t


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--voz', default='es_MX-claude-high')
    parser.add_argument('--audio', action='store_true', help='Sintetiza la pregunta y la transcribe con Whisper')
    parser.add_argument('--repeticiones', type=int, default=2)
    parser.add_argument('--json', help='Guarda los resultados en este archivo')
    parser.add_argument('--mostrar', action='store_true', help='Muestra el inicio de cada respuesta')
    parser.add_argument('--sin-precalentar', action='store_true', help='No precalienta LM Studio antes de cada pregunta')
    args = parser.parse_args()
    copy = Path(tempfile.mkdtemp()) / 'memoria.sqlite3'
    shutil.copy(servidor.memory.path, copy)  # copia: la memoria real no se modifica
    memory = Memoria(copy)
    results = []
    real_warm = servidor.warm_llm
    with mock.patch.object(servidor, 'memory', memory), mock.patch.object(servidor, 'conversation', ConversationState()), \
            mock.patch.object(servidor, 'warm_llm', lambda: None):
        servidor.voz.synthesize('Hola.', args.voz)  # carga la voz fuera de la medición
        for run in range(args.repeticiones):
            for label, question in QUERIES:
                # En uso real el precalentamiento ocurre mientras la persona escucha y habla; aquí se espera a que
                # termine para no medir la cola de LM Studio (atiende una petición cada vez).
                warm = None
                if not args.sin_precalentar:
                    t = time.perf_counter()
                    real_warm()
                    warm = time.perf_counter() - t
                transcription = None
                if args.audio:
                    question, transcription = transcribe_spoken(question)
                row = dict(measure(question, args.voz, transcription), consulta=label, ronda=run + 1, precalentamiento=warm)
                results.append(row)
                fmt = lambda v: '  —  ' if v is None else f'{v:5.2f}'
                print(f"{label} r{run + 1} · token {fmt(row['first_token'])} · frase {fmt(row['first_sentence'])} · "
                      f"audio {fmt(row['first_audio'])} · completa {fmt(row['generacion'])} · síntesis {fmt(row['sintesis'])} · "
                      f"{row['caracteres']:4d} car · {row['frases']} frases"
                      + (f" · precalentar {warm:.2f}" if warm is not None else '')
                      + (f" · ASR {row['transcripcion']:.2f}" if row['transcripcion'] else ''), flush=True)
                if args.mostrar:
                    print('   →', row['respuesta'][:140].replace('\n', ' '))
    if args.json:
        Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=1))
    servidor.voz.QWEN.unload()


if __name__ == '__main__':
    sys.exit(main())
