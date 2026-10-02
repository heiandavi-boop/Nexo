"""Asistente local: LM Studio + voz de macOS. Sin dependencias externas."""

import argparse
import json
import os
import re
import subprocess
import urllib.error
import urllib.request

from personalidad import MAX_TOKENS, TEMPERATURE, system_prompt


BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_MODEL = "qwen3-vl-8b-instruct-mlx"
SYSTEM = system_prompt(memory=False)


def request_json(path, payload=None, timeout=120):
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("LM_STUDIO_API_TOKEN")
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(
        BASE_URL + path,
        data=json.dumps(payload).encode("utf-8") if payload is not None else None,
        headers=headers,
    )
    # Evita enviar conexiones locales a un proxy configurado en el sistema.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return json.load(response)


def answer(model, history, question):
    result = request_json("/chat/completions", {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}]
        + history[-8:] + [{"role": "user", "content": question}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "stream": False,
    })
    content = result["choices"][0]["message"].get("content") or ""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S)
    content = re.sub(r"<think>.*$", "", content, flags=re.S).strip()
    if not content:
        raise ValueError("El modelo no devolvió una respuesta visible. Revisa su configuración.")
    return content


def speak(text, voice):
    from voz import preparar_texto_para_voz
    text = preparar_texto_para_voz(text)
    if not text:
        return
    # El texto viaja por stdin, nunca se interpreta como un comando.
    result = subprocess.run(
        ["/usr/bin/say", "-v", voice], input=text, text=True, check=False
    )
    if result.returncode:
        print("No se pudo reproducir la voz. Puedes continuar escribiendo o usar --sin-voz.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelo", default=DEFAULT_MODEL)
    parser.add_argument("--voz", default="Paulina")
    parser.add_argument("--sin-voz", action="store_true")
    parser.add_argument("--microfono", action="store_true", help="Enter para grabar y Enter para terminar")
    parser.add_argument("--probar", action="store_true", help="Prueba conexión y una respuesta sin sonido")
    args = parser.parse_args()
    history = []
    voice_enabled = not args.sin_voz
    try:
        models = [item["id"] for item in request_json("/models")["data"]]
        if args.modelo not in models:
            print("No encuentro el modelo:", args.modelo)
            print("Modelos disponibles:\n  " + "\n  ".join(models))
            return 1
        if args.probar:
            print(answer(args.modelo, [], "Di hola en una frase breve."))
            return 0
        microphone = None
        if args.microfono:
            try:
                from escuchar import Microfono
                print("Cargando reconocimiento de voz local…", flush=True)
                microphone = Microfono()
            except Exception as error:
                print("No se pudo iniciar el micrófono:", error)
                print("Usa el entorno .venv y sigue la sección Micrófono del README.")
                return 1
        print("\nAsistente local · " + args.modelo)
        print("Escribe y pulsa Enter. /salir termina · /nuevo borra el contexto · /voz activa o silencia")
        print("Los mensajes se mantienen solo en memoria durante esta sesión.\n")
        if microphone:
            print("Pulsa Enter sin escribir para empezar a grabar. También puedes escribir mensajes.")
        while True:
            question = input("Tú: ").strip()
            if not question and microphone:
                try:
                    question = microphone.escuchar()
                except Exception as error:
                    print("No pude grabar o transcribir:", error)
                    print("Revisa el permiso de micrófono de VS Code en Ajustes del Sistema → Privacidad y seguridad → Micrófono.")
                    continue
                if question:
                    print("Escuché:", question)
                else:
                    print("No detecté palabras. Inténtalo de nuevo.")
            if not question:
                continue
            if question == "/salir":
                break
            if question == "/nuevo":
                history.clear()
                print("Conversación reiniciada.\n")
                continue
            if question == "/voz":
                voice_enabled = not voice_enabled
                print("Voz " + ("activada." if voice_enabled else "silenciada."))
                continue
            # Limita también mensajes individuales para evitar contextos enormes.
            if len(question) > 2000:
                print("Para esta primera versión, escribe menos de 2000 caracteres por mensaje.")
                continue
            try:
                print("Procesando…", flush=True)
                response = answer(args.modelo, history, question)
                print("\nIA: " + response + "\n")
                history.extend([
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": response},
                ])
                history = history[-8:]
                if voice_enabled:
                    speak(response, args.voz)
            except (urllib.error.URLError, TimeoutError, ValueError, KeyError, IndexError) as error:
                print("No se pudo completar la respuesta:", error)
                print("Revisa LM Studio y prueba de nuevo.\n")
    except urllib.error.HTTPError as error:
        print("LM Studio devolvió HTTP", error.code)
        print("Si requiere autenticación, configura LM_STUDIO_API_TOKEN en tu terminal.")
        return 1
    except (urllib.error.URLError, TimeoutError):
        print("No puedo conectar con LM Studio en " + BASE_URL)
        print("Abre Developer, carga Qwen y activa Start Server en el puerto 1234.")
        return 1
    except (EOFError, KeyboardInterrupt):
        print("\nHasta luego.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
