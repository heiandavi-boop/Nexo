"""Proceso persistente de Qwen3-TTS (se ejecuta con .venv-tts, Python 3.12 + mlx-audio).

Protocolo por líneas JSON: cada solicitud en stdin, cada respuesta en stdout.
Solo trabaja con archivos locales; no descarga nada.
"""
import base64
import io
import json
import os
import sys
import time
import wave

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
protocol = sys.stdout
# Cualquier mensaje de las bibliotecas va a stderr para no romper el protocolo.
sys.stdout = sys.stderr


def send(message):
    protocol.write(json.dumps(message) + '\n')
    protocol.flush()


def to_wav(samples, rate):
    import numpy as np
    pcm = (np.clip(samples, -1, 1) * 32767).astype('<i2')
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


def main(model_dir):
    import mlx.core as mx
    import numpy as np
    from mlx_audio.tts.utils import load_model
    start = time.perf_counter()
    try:
        model = load_model(model_dir)
    except Exception as error:
        send({'ready': False, 'error': f'No pude cargar Qwen3-TTS: {error}'})
        return
    send({'ready': True, 'load_seconds': round(time.perf_counter() - start, 2),
          'speakers': model.get_supported_speakers(), 'languages': model.get_supported_languages()})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            start = time.perf_counter()
            mx.reset_peak_memory()
            # Por fragmentos de 1 s: mismo tiempo total, pero el pico de memoria baja de ~6 GB a ~3 GB.
            results = list(model.generate_custom_voice(
                text=request['text'], speaker=request['speaker'], language=request.get('language', 'spanish'),
                stream=True, streaming_interval=1.0))
            audio = np.concatenate([np.array(r.audio, dtype=np.float32) for r in results])
            rate = results[0].sample_rate
            data = to_wav(audio, rate)
            send({'id': request.get('id'), 'ok': True, 'wav': base64.b64encode(data).decode('ascii'),
                  'seconds': round(time.perf_counter() - start, 2), 'audio_seconds': round(len(audio) / rate, 2),
                  'peak_gb': round(mx.get_peak_memory() / 1e9, 2)})
        except Exception as error:
            send({'id': None, 'ok': False, 'error': str(error) or error.__class__.__name__})
        finally:
            mx.clear_cache()


if __name__ == '__main__':
    main(sys.argv[1])
