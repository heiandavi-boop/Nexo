# Mi asistente local

## Instalación (una sola vez, requiere internet)

Desde la carpeta del proyecto, en la terminal de VS Code:

```sh
python3 -m venv .venv                      # solo si .venv no existe
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python escuchar.py --descargar
```

`--descargar` guarda en `modelos/` Whisper large-v3-turbo para MLX (~1,6 GB, `mlx-community/whisper-large-v3-turbo`), Silero VAD v6.2 (~2 MB, verificado con SHA-256) y cuatro voces Piper en español (~300 MB en `modelos/voces`). Después todo funciona sin internet: el servidor activa `HF_HUB_OFFLINE=1` y solo lee archivos locales. Todo se instala dentro de `.venv`; no se modifica el Python del sistema.

## Aplicación web con manos libres y memoria

Con LM Studio encendido (servidor en el puerto 1234 y `qwen3-vl-8b-instruct-mlx` cargado):

```sh
.venv/bin/python servidor.py
```

Abre siempre **http://localhost:8765** en Safari o Chrome (no `127.0.0.1` ni otro puerto: el navegador guarda el permiso del micrófono por origen, y si entras por `http://127.0.0.1:8765` el servidor te redirige a `localhost`). La primera carga del modelo de voz tarda unos segundos (se hace en segundo plano al arrancar).

### Inicio automático y permisos

Con **Activar conversación al abrir** marcado (predeterminado; la elección se guarda en este navegador), al abrir la página Nexo:

1. Consulta el estado del permiso con la Permissions API, si el navegador la ofrece. Si no la ofrece, no lo trata como denegado.
2. Si está **denegado**, no vuelve a pedirlo: muestra cómo habilitarlo.
3. En otro caso hace **un único intento** de activar el micrófono por carga de página (nunca en bucle). Si nunca lo autorizaste, el navegador te lo preguntará en ese momento.
4. Intenta reanudar el audio sin quedarse esperando. Si el navegador exige un gesto, libera el micrófono y muestra **Activar micrófono y audio**: un toque y queda escuchando.
5. «Escuchando» solo aparece cuando la captura y el procesamiento funcionan; «Hablando» solo cuando la voz empezó a sonar. Si el navegador bloquea la voz, aparece **Toca una vez para habilitar la voz**; la respuesta bloqueada no se reproduce después.

Bajo el botón se ve el estado real: *Permiso del micrófono* (concedido, sin decidir, denegado o desconocido), *Micrófono* (en uso, con punto rojo, o apagado) y *Voz* (habilitada, bloqueada o sin activar). Pausar libera el micrófono pero no revoca el permiso ni cambia la preferencia. Una pausa manual no se reanuda sola; cambiar de pestaña también pausa, como antes. Si la página se abre en segundo plano, el inicio automático espera a que la muestres.

**Autorizar una sola vez (cuando el navegador lo permite):**

- **Chrome**: al aparecer la pregunta elige **Permitir** (no «Permitir esta vez»). Se revisa en el icono a la izquierda de la dirección → Micrófono → Permitir. Chrome suele exigir además **un clic por carga** para habilitar el audio, salvo que ya hayas usado mucho la página; en ese caso verás *Activar micrófono y audio*.
- **Safari**: Safari → Ajustes → Sitios web → Micrófono → `localhost` → **Permitir**. Con «Preguntar» vuelve a consultarte en cada visita. Mientras el micrófono está activo, Safari suele permitir el audio sin otro clic.
- macOS: Ajustes del Sistema → Privacidad y seguridad → Micrófono debe incluir el navegador.

El navegador decide; no se puede garantizar que nunca vuelva a preguntar. Pueden pedir autorización otra vez: modo privado, permisos de «solo esta vez», borrar datos o permisos del sitio, cambiar de navegador o de perfil, o abrir otro origen (`127.0.0.1`, otro puerto). Reiniciar el servidor manteniendo `http://localhost:8765` no debería por sí solo exigir una nueva autorización persistente, aunque depende del navegador. La interacción inicial inevitable es aceptar la primera pregunta de permiso y, en Chrome, normalmente un clic para el audio en cada carga.

### Personalidad y parámetros de generación

Todo está en [personalidad.py](personalidad.py): `PERSONALITY` (carácter y reglas), `STYLE_GUIDE` (ejemplos de intención, no frases a copiar), `MEMORY_RULES`, `REMINDER` (lista breve al final del prompt, porque el modelo de 8B atiende más a lo último) y los parámetros `TEMPERATURE = 0.6` y `MAX_TOKENS = 512`. Los usan tanto la web como `main.py`.

`MAX_TOKENS` es un techo, no una longitud objetivo; respuestas más largas tardan más en generarse y en leerse en voz alta. La temperatura no garantiza naturalidad ni precisión: con 0.6 hay más variedad que con 0.3 y, a veces, más errores de estilo. El servidor pregunta a LM Studio el contexto cargado (`loaded_context_length`; si no responde, supone 4096) y ajusta cuánta conversación antigua adjunta, reservando siempre `MAX_TOKENS` para la respuesta. Los datos confirmados del perfil van siempre en el prompt; los recuerdos antiguos relacionados van como notas de referencia (no como diálogo) para que sus errores o su estilo antiguo no se imiten. Se corta un bucle de líneas repetidas; la respuesta se muestra y se guarda tal como la escribió el modelo.

### Cómo decide que terminaste de hablar

Ya no hay espera fija de 3 segundos:

1. **Silero VAD** (local, en el servidor) calcula cada 32 ms la probabilidad de que haya voz. El ruido constante, golpes o teclas no abren un turno; los sonidos de menos de 0,25 s de voz se descartan sin transcribir.
2. Se guardan **0,5 s previos** al inicio de la voz (primeras sílabas) y **0,3 s posteriores** a la última voz (últimas sílabas). Un umbral más bajo para seguir hablando que para empezar evita cortar finales suaves.
3. Tras 0,25 s de silencio, Whisper transcribe **mientras sigue escuchando**. Si el texto parece una frase terminada, responde tras la **pausa corta (0,7 s)**. Si termina en palabras como «y», «que», «de la», «porque», «eh», en coma o puntos suspensivos, espera la **pausa larga (1,6 s)**. Si vuelves a hablar, esa transcripción se descarta y el turno continúa.

Límites de la pista lingüística: solo mira las últimas palabras; no entiende la oración. Puede esperar de más («Yo no como») o responder antes de tiempo si haces una pausa larga tras una frase gramaticalmente completa («Quiero ir. … a la playa»). Nunca descarta lo que dijiste: solo decide cuánto esperar. La espera real es aproximadamente el mayor entre la pausa y el tiempo de transcripción.

Ajustes en **Ajustar detección de voz** (se guardan en el navegador y se aplican desde el siguiente turno): exigencia para detectar voz, pausa al terminar una frase y pausa dentro de una frase. También por variables de entorno al iniciar, por ejemplo:

```sh
NEXO_PAUSA_CORTA=0.8 NEXO_PAUSA_LARGA=2.0 NEXO_UMBRAL_INICIO=0.6 .venv/bin/python servidor.py
```

Otras variables: `NEXO_PREBUFFER`, `NEXO_COLA`, `NEXO_REVISAR_TRAS`, `NEXO_VOZ_MINIMA`, `NEXO_TURNO_MAXIMO` (segundos), `NEXO_CONFIANZA_PALABRAS=0` (ahorra ~0,3 s por turno a cambio de no marcar palabras dudosas) y `NEXO_PUERTO`.

### Transcripción y palabras dudosas

- Whisper large-v3-turbo se ejecuta en la GPU con MLX y escribe la puntuación. La aplicación solo corrige espacios, mayúsculas iniciales y añade «¿»/«¡» de apertura o el punto final; nunca cambia, añade ni quita palabras.
- El nombre confirmado en tu perfil (y «Nexo») se pasa a Whisper como pista de ortografía. No se insertan en el texto; solo ayudan cuando realmente los dices.
- Las palabras reconocidas con poca seguridad se resaltan en tu mensaje y se avisa a Qwen para que pida aclaración si importan. Si un dato personal («Me llamo…») contiene una palabra dudosa, **no se guarda** en el perfil.
- Si casi toda la frase es dudosa, Nexo responde «No estoy seguro de haberte entendido…» sin consultar a Qwen ni guardar ese intercambio.
- Se descartan transcripciones vacías y frases típicas que Whisper inventa sobre ruido («Subtítulos realizados por…», «Gracias por ver el video», «[Música]»).

### Estados y errores

La página muestra **Escuchando**, **Transcribiendo**, **Pensando** y **Hablando**, y bajo el medidor los tiempos del último turno: fin de turno, transcripción, generación y voz. La terminal del servidor imprime solo esos tiempos, nunca el contenido. Se avisa si el permiso de micrófono se deniega, no hay micrófono, el modelo de voz falta o LM Studio no responde.

La escucha se pausa mientras Nexo piensa y habla (y 0,35 s después), con cancelación de eco del navegador: su voz no se toma como un turno nuevo. Por eso no puedes interrumpirlo hablando encima; usa **Pausar conversación**, que detiene la voz y libera el micrófono. Cambiar de pestaña también pausa. Cada intervención se limita a 60 segundos.

No usa reconocimiento de voz del navegador ni servicios externos: el audio viaja solo de la pestaña a `127.0.0.1` y no se guarda en disco.

### Por qué Whisper large-v3-turbo con MLX

- **Precisión**: mismo codificador que large-v3 (el mejor Whisper multilingüe); mucho más preciso que base con nombres propios y habla rápida en español.
- **Latencia**: el decodificador de 4 capas y la GPU de Apple Silicon dan ~1 s para una frase de 5–7 s con confianza por palabra, en paralelo a la pausa. large-v3 completo sería varias veces más lento; faster-whisper no usa la GPU de Mac.
- **Memoria**: ~1,6 GB de pesos (unos 2–3 GB en uso), compatible con Qwen 8B en 24 GB.
- `distil-large-v3` es solo inglés, por eso no se eligió.

Si falta el modelo turbo pero existe `modelos/whisper-base`, se usa ese de respaldo con un aviso en la página.

### Voz de Nexo

La voz se genera en el Mac con **Piper** (voces neuronales, sin internet) y se reproduce en el navegador; la boca del personaje sigue su volumen. En **Voz de Nexo** puedes elegir Claude o Ald (México), Daniela (Argentina), Davefx (España) o Paulina de macOS, y pulsar **Escuchar muestra**. La elección se guarda en el navegador. La predeterminada es `es_MX-claude-high`; cámbiala con `NEXO_VOZ=es_AR-daniela-high`. Si una voz Piper falla, se usa Paulina automáticamente. Antes de cualquier motor, `preparar_texto_para_voz()` en [voz.py](voz.py) quita emojis (incluidos los compuestos, banderas y tonos de piel), emoticonos y marcas de Markdown, convierte enlaces en su texto visible y añade pausas a títulos y viñetas, sin cambiar palabras; si no queda nada pronunciable, no se genera audio. En pantalla y en la memoria queda la respuesta original. Piper tarda ~0,15 s por frase (Paulina ~0,8 s). El motor Piper es GPL-3.0 y cada voz tiene su propia licencia (ver su `.onnx.json` y el repositorio `rhasspy/piper-voices`).

### Memoria

- Las conversaciones web se guardan en `datos/memoria.sqlite3`, incluso después de reiniciar.
- Los datos explícitos de nombre, edad declarada, profesión y la aclaración de si eres emprendedor se guardan también en un perfil persistente. Las correcciones posteriores reemplazan esos campos y las respuestas del asistente nunca alimentan ese perfil. Revisa **Datos que recuerdo de ti** en la página. Esta extracción es conservadora y reconoce determinadas frases, no cualquier hecho posible; para profesión utiliza «Mi profesión es…».
- Al actualizar desde la versión anterior, el perfil se reconstruye en orden a partir de tus mensajes guardados, sin borrar el historial. Los errores de transcripción deben corregirse explícitamente, preferiblemente por escrito si afectan nombres propios.
- El contexto incluye intercambios recientes y otros relacionados por coincidencia de palabras, dentro de un presupuesto de tamaño. No carga el historial completo ni garantiza recordar cada detalle; usa nombres o palabras del tema para recuperar conversaciones antiguas.
- La pantalla muestra hasta los últimos 100 intercambios; el archivo conserva los anteriores.
- **Borrar memoria** elimina el historial de la aplicación después de una confirmación. No modifica registros externos de LM Studio ni copias de seguridad del Mac.
- La memoria corresponde a una sola persona en este Mac y se comparte entre pestañas. Se procesa un mensaje a la vez.
- La versión de terminal anterior sigue funcionando y no comparte esta memoria web.

Si el puerto aparece ocupado, es posible que la aplicación ya esté abierta: prueba la dirección antes de iniciarla otra vez. Para detener el servidor, usa Ctrl+C en su terminal.

### Verificación

```sh
.venv/bin/python -m unittest discover -s tests   # turnos, ruido, sílabas, duplicados, memoria y audio sintetizado
node tests/vad.mjs                               # remuestreo a 16 kHz y envío ordenado del navegador
.venv/bin/python medir.py                        # tiempos por etapa (añade --sin-lm si LM Studio está cerrado)
```

`tests/test_audio_real.py` y `medir.py` usan frases generadas con la voz Paulina, no tu micrófono; con un micrófono real, acento, distancia y ruido los resultados pueden variar. `medir.py` no escribe en la memoria.

`tests/test_personalidad.py` y `tests/test_servidor.py` comprueban las instrucciones y el payload enviado a LM Studio (parámetros, perfil, presupuesto de contexto), no frases exactas del modelo.

**Comprobación manual de permisos** (en tu navegador, con el servidor en marcha):

1. Primera visita: abre `http://localhost:8765` → debe preguntar por el micrófono una vez; elige Permitir. Debe quedar «Escuchando» o mostrar *Activar micrófono y audio*.
2. Recarga (Cmd+R): no debe volver a preguntar; queda escuchando o pide solo el toque de audio.
3. Reinicia el servidor (Ctrl+C y `.venv/bin/python servidor.py`) y recarga: igual que el paso 2.
4. Pulsa **Pausar conversación**: el indicador de micrófono del navegador se apaga y no se reanuda solo.
5. Desmarca *Activar conversación al abrir* y recarga: no debe activar el micrófono.
6. Bloquea el micrófono en los ajustes del sitio y recarga: debe mostrar instrucciones sin volver a preguntar.
7. Desconecta el micrófono USB durante la escucha: debe pausar y avisar.

Antes de esta actualización se guardó una copia en `datos/memoria.respaldo-antes-de-voz.sqlite3`; la base actual no cambia de formato.

## Micrófono en la versión de terminal

Tras la instalación:

```sh
.venv/bin/python main.py --microfono
```

1. Acepta el permiso de micrófono de macOS para VS Code o la terminal desde la que ejecutes el programa.
2. Pulsa Enter sin escribir para empezar a grabar.
3. Habla y pulsa Enter otra vez para terminar.
4. Verás la transcripción y escucharás la respuesta de Qwen.

Usa el mismo Whisper large-v3-turbo local. La captura se limita a 60 segundos y solo vive en memoria. Para probar un archivo: `.venv/bin/python escuchar.py --archivo grabacion.wav` (WAV PCM de 16 bits).

## Abrir y ejecutar en VS Code

1. Abre esta carpeta mediante **Archivo → Abrir carpeta**.
2. Deja LM Studio abierto, con el servidor iniciado en el puerto **1234** y el modelo `qwen3-vl-8b-instruct-mlx` disponible.
3. En VS Code, abre **Terminal → Nueva terminal**.
4. Ejecuta:

```sh
python3 main.py
```

No necesitas instalar paquetes de Python. Se utiliza la biblioteca estándar y el comando `say` de macOS. La voz seleccionada es Paulina: debe estar instalada en el Mac. Los modelos y voces descargados permiten usar esta aplicación sin internet.

## Comandos

- `/voz`: activar o silenciar la lectura.
- `/nuevo`: limpiar el contexto de esta aplicación.
- `/salir`: terminar.

La aplicación conserva hasta cuatro intercambios recientes en memoria y no escribe conversaciones en archivos. LM Studio administra sus propios registros. Usa la misma personalidad y límite de tokens que la web (ver [personalidad.py](personalidad.py)).

## Opciones

```sh
python3 main.py --sin-voz
python3 main.py --probar
python3 main.py --voz 'Mónica'
python3 main.py --modelo 'qwen3-vl-8b-instruct-mlx'
```

Para listar voces instaladas: `say -v '?'`.

Si activaste autenticación en LM Studio, define la variable de entorno `LM_STUDIO_API_TOKEN` antes de iniciar; no guardes tu token en el código ni lo compartas.

## Archivos

- `main.py`: conexión con el modelo, conversación y salida de voz.
- `personalidad.py`: personalidad de Nexo, temperatura, límite de tokens y presupuesto de contexto.
- `servidor.py`: aplicación web, sesión de escucha y respuestas.
- `transcripcion.py`: Whisper large-v3-turbo (MLX), puntuación conservadora y confianza por palabra.
- `turnos.py`: Silero VAD y detección adaptable del fin de turno.
- `escuchar.py`: descarga de modelos y micrófono de la versión de terminal.
- `medir.py`: medición de tiempos por etapa.
- `web/capture.js`: captura y remuestreo a 16 kHz; `web/vad.js`: envío ordenado del audio al servidor local.
- `.vscode/tasks.json`: tarea **Iniciar asistente local**, disponible en Terminal → Ejecutar tarea.

La conexión utiliza exclusivamente `127.0.0.1:1234`. No necesita servicios de IA en la nube.
