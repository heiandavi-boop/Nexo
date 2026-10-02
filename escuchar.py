"""Descarga única de modelos y micrófono para la versión de terminal. Sin servicios externos al conversar."""
import argparse
import hashlib
import urllib.request

import numpy as np

from transcripcion import MODEL_DIR, MODEL_REPO, RATE, VAD_PATH, VAD_SHA256, VAD_URL, Transcriber, load_wav


def descargar():
    from huggingface_hub import snapshot_download
    print('Descargando Whisper large-v3-turbo para MLX (~1,6 GB)…', flush=True)
    snapshot_download(MODEL_REPO, local_dir=str(MODEL_DIR), allow_patterns=['*.json', '*.safetensors'])
    if not VAD_PATH.exists() or hashlib.sha256(VAD_PATH.read_bytes()).hexdigest() != VAD_SHA256:
        print('Descargando Silero VAD v6.2 (~2 MB)…', flush=True)
        with urllib.request.urlopen(VAD_URL, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != VAD_SHA256:
            raise SystemExit('La suma de verificación de Silero VAD no coincide; no se guardó el archivo.')
        VAD_PATH.write_bytes(data)
    import voz
    voz.download()
    print('Modelos listos en modelos/. Las conversaciones posteriores funcionan sin internet.')


class Microfono:
    def __init__(self):
        import sounddevice as sd
        self.sd = sd
        self.transcriber = Transcriber()
        self.transcriber.load()

    def transcribir(self, audio):
        return self.transcriber.transcribe(audio).text

    def escuchar(self):
        chunks, statuses = [], []
        limit, count = RATE * 60, 0

        def callback(data, frames, time_info, status):
            nonlocal count
            if status:
                statuses.append(str(status))
            remaining = limit - count
            if remaining > 0:
                chunks.append(bytes(data)[:remaining * 2])
                count += min(frames, remaining)
            if count >= limit:
                raise self.sd.CallbackStop

        print('Habla ahora. Pulsa Enter para terminar (máximo 60 segundos).', flush=True)
        # CoreAudio entrega directamente 16 kHz; el audio solo vive en memoria.
        with self.sd.RawInputStream(samplerate=RATE, channels=1, dtype='int16', callback=callback):
            input()
        if statuses:
            print('Aviso de audio:', statuses[0])
        if count < RATE // 4:
            return ''
        print('Transcribiendo localmente…', flush=True)
        return self.transcribir(np.frombuffer(b''.join(chunks), dtype='<i2').astype(np.float32) / 32768)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--descargar', action='store_true', help='Descarga Whisper large-v3-turbo (MLX) y Silero VAD')
    parser.add_argument('--archivo', help='Transcribir un WAV existente para verificar el modelo')
    args = parser.parse_args()
    if args.descargar:
        descargar()
    elif args.archivo:
        result = Transcriber().transcribe(load_wav(args.archivo))
        print(result.text or f'(descartado: {result.discarded})')
        print(f'confianza {result.confidence} · dudosas {result.doubtful} · {result.seconds:.2f} s')
    else:
        parser.print_help()
