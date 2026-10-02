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

### Contexto vocal opcional

En paralelo a Whisper, Nexo analiza en memoria el mismo array del turno con NumPy: energía RMS por ventanas, pitch y su variación mediante autocorrelación, ritmo aproximado a partir de picos de energía, pausas y proporción de silencio. No modifica ni vuelve a transcribir el audio. Una baseline móvil de hasta 12 turnos aceptados se mantiene solo en RAM y se adapta gradualmente; necesita 3 turnos utilizables antes de comparar. Clips menores de 0,8 s, menos de 0,55 s de voz, reconocimientos de baja confianza, respuestas de hasta 2 palabras, ruido o pitch poco fiable no producen contexto vocal.

Solo si varias señales se apartan de la baseline se añade una nota breve al turno actual, con confianza heurística media/alta. Describe activación relativa, no clasifica enojo, tristeza ni estados clínicos. La señal es una inferencia imperfecta: debe modular con sutileza tono y brevedad, no mencionarse por rutina ni guardarse como dato, recuerdo o preferencia. El análisis no se envía a servicios externos y no se guarda audio.

La extracción usa un pool local de un worker y tiene un presupuesto máximo de espera de **50 ms** después de que Whisper acepta el turno. Si vence, falla o se cancela, Nexo responde solo con el texto. El análisis de cinco segundos de señal sintética midió ~9,1 ms en la primera llamada (inicialización FFT), luego ~0,7 ms; 20 llamadas adicionales dieron p95 ~1,3 ms. El array de entrada ocupa 320 KB por cinco segundos y se comparte con Whisper; el extractor solo crea buffers temporales pequeños. Son mediciones sintéticas, no una garantía para todos los micrófonos/ruidos. No se instala ni carga clasificador emocional; prosodia sola no permite distinguir con fiabilidad entusiasmo de enojo, ni está calibrada como detector de emociones en español.

Benchmark A/B local reproducible (requiere macOS `say`/`afconvert`, LM Studio con Qwen cargado y Piper instalado): `PYTHONPATH=tests NEXO_PROBAR_PROSODIA=1 .venv/bin/python -m unittest test_audio_real.RealProsodyABTests.test_real_audio_whisper_prosody_and_first_audio_ab -v`. Usa SQLite temporal, sintetiza el audio localmente, compara turnos precalentados en orden alternado y muestra texto real, transcripción, primer token, primer audio, espera adicional y RSS del proceso de prueba. En este Mac, dos rondas dieron ~0,56 s de primer audio sin prosodia y ~0,62 s con ella (diferencia observada ~55 ms, no atribuible con certeza al análisis); Whisper ~0,89 s en ambas, primer token ~0,12–0,14 s, extracción ~1,27 ms, espera adicional ~0,003 ms y RSS incremental ~0,05 MB. No implica que el tono generado sea siempre mejor: en esta frase de prueba las respuestas fueron similares.

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

La voz se genera en el Mac y se reproduce en el navegador; la boca del personaje sigue su volumen. En **Voz de Nexo** eliges entre tres motores locales: **Piper** (Claude y Ald de México, Daniela de Argentina, Davefx de España), **Paulina** de macOS y **Qwen3-TTS** (ver abajo). La elección se guarda en el navegador. La predeterminada es `es_MX-claude-high`; cámbiala con `NEXO_VOZ=es_AR-daniela-high` (o `NEXO_VOZ=qwen3-serena`). Antes de cualquier motor, `preparar_texto_para_voz()` en [voz.py](voz.py) quita emojis (incluidos los compuestos, banderas y tonos de piel), emoticonos y marcas de Markdown, convierte enlaces en su texto visible y añade pausas a títulos y viñetas, sin cambiar palabras; si no queda nada pronunciable, no se genera audio. En pantalla y en la memoria queda la respuesta original. El motor Piper es GPL-3.0 y cada voz tiene su propia licencia (ver su `.onnx.json` y el repositorio `rhasspy/piper-voices`).

**Respaldo visible.** En conversación, si la voz elegida falla, Nexo responde con Paulina y lo avisa en pantalla con el motivo. En **Escuchar muestra** nunca se sustituye: si la voz falla, ves el error.

**Comparar voces.** Abre *Comparar voces*, edita el texto de prueba (el mismo para todas; se guarda en el navegador), elige una voz y pulsa **Escuchar muestra**. Debajo aparecen el motor y la voz realmente usados, el tiempo de carga del modelo (solo la primera vez), el de síntesis, la duración del audio y, en Qwen3-TTS, el pico de memoria. El mismo botón pasa a **Detener muestra**. Mientras suena una muestra la escucha queda en pausa. Después de cada respuesta normal también se muestra qué motor sonó.

### Qwen3-TTS

- **Modelo**: `mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit` (conversión MLX de `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`, Apache-2.0), en `modelos/qwen3-tts-0.6b-customvoice-8bit` (~1,8 GB). Variante *CustomVoice*: voces predefinidas, sin clonar ni usar audio de referencia.
- **Voces**: las 9 que declara el modelo (Serena, Vivian, Uncle Fu, Dylan, Eric, Ryan, Aiden, Ono Anna y Sohee). Según su ficha oficial, ninguna es hispanohablante nativa (chino, inglés, japonés o coreano); todas se usan con `language="spanish"`, que el modelo admite. Pueden sonar con acento extranjero.
- **Entorno**: `.venv-tts` con Python 3.12 (Homebrew) y `mlx-audio==0.5.7` (trae `mlx 0.32`). Va aparte porque mlx-audio exige Python ≥ 3.10 y mlx ≥ 0.31, mientras Whisper usa `.venv` con Python 3.9 y `mlx 0.29`; así no se toca lo que ya funciona.
- **Funcionamiento**: [voz.py](voz.py) arranca [tts_qwen.py](tts_qwen.py) con `.venv-tts` la primera vez que eliges o pruebas una voz Qwen; el proceso mantiene el modelo cargado, atiende una síntesis a la vez y se cierra tras 10 minutos sin uso (`NEXO_QWEN_INACTIVIDAD`, en segundos) o al cerrar el servidor. Trabaja sin internet (`HF_HUB_OFFLINE=1`, solo archivos locales). El audio sale como WAV PCM de 16 bits a 24 kHz, compatible con la reproducción y la boca animada.

Instalación (una vez, con internet):

```sh
/opt/homebrew/bin/python3.12 -m venv .venv-tts
.venv-tts/bin/python -m pip install --upgrade pip
.venv-tts/bin/python -m pip install -r requirements-tts.txt
.venv/bin/python escuchar.py --descargar-qwen
```

Si falta `.venv-tts` o el modelo, las voces Qwen simplemente no aparecen en la lista.

**Medido en este Mac (M5 Pro, 24 GB)** con el texto de prueba (~170 caracteres), con el servidor (Whisper cargado) y LM Studio con Qwen3-VL 8B en marcha:

| Voz | Síntesis | Audio generado |
|---|---|---|
| Piper Claude | 0,28 s | 10,9 s |
| Piper Daniela | 1,41 s | 8,4 s |
| Paulina | 0,93 s | 11,3 s |
| Qwen3-TTS Serena, 1.ª vez | 1,25 s de carga + 3,76 s | 13,3 s |
| Qwen3-TTS Serena, Vivian, Ryan, Aiden (ya cargado) | 2,5–3,2 s | 11,8–14,9 s |

Qwen3-TTS no reproduce en streaming: la voz empieza cuando termina toda la síntesis, así que en conversación añade ~2,5–3 s por respuesta de este tamaño (más en respuestas largas). Su memoria: ~2 GB activos con el modelo cargado y picos de ~3,1 GB de memoria unificada durante cada síntesis (se genera por fragmentos de 1 s; sin fragmentar el pico era de ~6–6,6 GB, medido con `mx.get_peak_memory`). Con LM Studio, Whisper y Qwen3-TTS a la vez el Mac puede quedarse sin memoria y recurrir a swap: entonces todo se vuelve más lento (transcripción, generación y voz). Si ocurre, usa una voz Piper para conversar y Qwen3-TTS solo para comparar, o reduce en LM Studio el contexto cargado del modelo. El muestreo es aleatorio, así que la duración y la entonación varían entre ejecuciones con el mismo texto.

Prueba real (sintetiza en español, comprueba con Whisper que se entiende y que el proceso se reutiliza): `NEXO_PROBAR_QWEN=1 .venv/bin/python -m unittest tests.test_voz.QwenRealTests`.

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

### Tres capas: datos, preferencias y estado de la conversación

- **Memoria factual** (tabla `profile`): nombre, edad, profesión, emprendimiento. Igual que antes.
- **Preferencias conversacionales** (tabla `conversation_preferences`, persistente): cómo quieres conversar. Hoy reconoce «no insistas tanto en el mismo tema» / «no quiero que persistas en lo mismo» (y «ya puedes volver a profundizar en los temas» para revertirlo), y los temas rechazados de forma explícita y duradera («no vuelvas a hablar del café», «no menciones más la taza»). Se ven en el prompt, no en «Datos que recuerdo de ti». **Borrar memoria** también las elimina.
- **Estado temporal** (solo mientras el servidor está abierto): tema en curso y temas abandonados en esta sesión. «Cambiemos de tema», «hablemos de otra cosa», «dejemos ese tema», «no insistas con el café»… cierran el tema y el diálogo anterior al pedido deja de enviarse al modelo como conversación reciente (no se borra del historial).

Los temas abandonados se envían al modelo en una sección *ESTADO CONVERSACIONAL ACTUAL* con la regla de no reintroducirlos por su cuenta, y los recuerdos antiguos que los mencionan no se adjuntan como referencia. **No es una prohibición eterna:** si tú vuelves a mencionarlo («¿Qué tipos de café produce Colombia?»), el tema se reabre y Nexo responde con normalidad. No se filtran palabras en la respuesta: todo se resuelve con estado, intención y prompt.

La detección es conservadora: solo instrucciones explícitas tuyas; nunca preguntas, ejemplos, hipótesis, citas, texto del asistente ni frases con palabras reconocidas con poca seguridad. No hay llamadas extra a LM Studio para clasificar, así que no añade latencia.

### Transcripciones imperfectas

Nexo intenta resolver errores de pronunciación o transcripción cuando la interpretación es suficientemente evidente, pero pide aclaración cuando existen varias posibilidades o el dato no es confiable. Por ejemplo, tras «¿Cuántos departamentos tiene Colombia?», ante «¿Cuál es la capital del Baje del Cauca?» responde algo como «Creo que hablas del Valle del Cauca; su capital es Cali», en vez de decir que no existe. Si la frase casi no se entiende (confianza baja en la transcripción), responde sin consultar al modelo pidiendo que la repitas, como antes.

**Inferir para responder no es guardar un dato.** La transcripción se guarda tal cual; la interpretación solo sirve para esa respuesta y nunca entra en el perfil ni en las preferencias. Un nombre, apellido u otro dato personal dudoso no se deduce ni se guarda: Nexo pregunta.

### Fecha y hora

Cada petición incluye la fecha, la hora y la zona horaria del reloj de este Mac (*CONTEXTO LOCAL ACTUAL*), leídas en ese momento y nunca guardadas en memoria; no dependen del conocimiento interno del modelo ni de servicios externos.

### Rapidez de respuesta

- **Voz por frases:** Qwen responde en streaming; el texto se trocea y cada fragmento se sintetiza (en otro hilo) y se envía mientras el resto se sigue generando. El navegador los reproduce en orden, sin solaparse.
- **Troceo:** la primera frase sale en cuanto hay ≥12 caracteres y un `.`, `!`, `?`, `…`, `;` o salto de línea («Tienes 40 años.» ya sirve). Las siguientes, con ≥60 caracteres, para no partir la entonación. Si una frase larga no tiene punto, se corta en la última coma, dos puntos o raya (a partir de 90 caracteres la primera, 220 las demás).
- **Prefijo estable para LM Studio:** las instrucciones fijas van idénticas en cada petición; lo que cambia (datos, recuerdos, fecha, estado) va con el mensaje actual. El diálogo reciente se adjunta como una ventana *anclada*: crece turno a turno sin mover su inicio (hasta 6 turnos, luego se reinicia conservando 2; tras un cambio de tema empieza en el límite). Así LM Studio reutiliza lo ya calculado y solo procesa el mensaje nuevo. Recuerdos antiguos: máx. ~1500 caracteres; diálogo: máx. ~6500.
- **Precalentamiento:** al arrancar el servidor y al terminar cada respuesta se pide a LM Studio 1 token con ese mismo prefijo (no se guarda nada), mientras escuchas y hablas. Piper se carga al arrancar y se mantiene en memoria; Qwen3-TTS queda en su proceso hasta 10 min sin uso.
- **Respuestas cortas:** una pregunta de dato puntual («¿Cuántos años tengo?») se responde en una frase.
- **Cancelación:** al pausar, la página cierra la conexión; el servidor deja de leer a Qwen, descarta los fragmentos pendientes de voz y no guarda esa respuesta. Un error a mitad de la respuesta tampoco guarda nada truncado.

**Métricas** (debajo del avatar y en el registro del servidor; nunca con el contenido de la conversación):

| Métrica | Desde → hasta |
|---|---|
| fin de turno | dejas de hablar → turno cerrado y transcrito |
| primer token | la petición llega al servidor → primer texto de Qwen |
| primera frase | → primer fragmento listo para la voz |
| audio listo | → primer audio sintetizado y enviado |
| empezó a sonar | el navegador recibe el turno → el audio suena de verdad (evento `playing`) |
| respuesta completa | → Qwen terminó de escribir |
| total hasta oír | fin de turno + empezó a sonar: lo que esperaste en silencio |

`.venv/bin/python medir_latencia.py --mostrar` mide las consultas A–D (dos rondas) sobre una copia temporal de la memoria, precalentando antes de cada una como en uso real. Opciones: `--voz qwen3-serena`, `--audio` (añade transcripción de la pregunta con Whisper), `--sin-precalentar`, `--json archivo`.

Resultados en este Mac (M5 Pro 24 GB, qwen3-vl-8b-instruct-mlx, voz Piper es_MX-claude-high; media de 2 rondas, segundos desde que el servidor recibe el turno):

| Consulta | primer token antes → ahora | audio listo antes → ahora | respuesta completa antes → ahora |
|---|---|---|---|
| A ¿Cuántos años tengo? | 3,04 → 2,01 | 3,16 → 2,10 (−34 %) | 4,53 → 2,05 |
| B ¿Cuántos días tiene un año? | 2,69 → 1,77 | 3,06 → 2,27 (−26 %) | 5,77 → 2,61 |
| C Explícame cómo se forman las nubes. | 2,71 → 1,17 | 3,04 → 1,59 (−48 %) | 9,09 → 5,57 |
| D Háblame sobre inteligencia artificial. | 2,51 → 0,74 | 2,90 → 1,18 (−59 %) | 8,18 → 5,92 |

A tarda más que D porque su contexto (perfil y recuerdos sobre tu edad) cambia la parte no cacheada. El «antes» ya tenía voz por frases; antes de eso la voz empezaba al terminar toda la respuesta. Hay que sumar el fin de turno (~1 s tras dejar de hablar) y la reproducción en el navegador. La primera petición tras arrancar o tras mucho tiempo sin uso es más lenta (~3–5 s), igual que con Qwen3-TTS como voz o si macOS usa mucho swap.
- `NEXO_MEMORIA=/ruta/prueba.sqlite3` permite iniciar el servidor con otro archivo de memoria para pruebas.

**Limitaciones** (comprobadas con LM Studio y una copia temporal de la memoria): el modelo de 8B no siempre obedece. Puede seguir terminando con preguntas u opciones, usar emojis en pantalla (la voz no los lee), contar anécdotas inventadas pese a la regla, o adivinar un significado para una palabra suelta sin contexto («Háblame de Baje» → «¿Baja California?») e incluso dar datos geográficos equivocados. La detección de temas usa reglas: entiende formulaciones habituales, no cualquier manera de decirlo; si no reconoce tu frase, repítela de forma más directa («No vuelvas a hablar de…», «Cambiemos de tema»). El estado temporal se pierde al reiniciar el servidor; los rechazos duraderos y las preferencias, no.

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
- `conversacion.py`: preferencias conversacionales, temas abandonados, estado temporal y fecha/hora local.
- `voz.py`: motores de voz (Piper, Paulina, Qwen3-TTS), limpieza de texto para voz y respaldo visible.
- `tts_qwen.py` y `requirements-tts.txt`: proceso y dependencias de Qwen3-TTS en `.venv-tts`.
- `servidor.py`: aplicación web, sesión de escucha y respuestas.
- `transcripcion.py`: Whisper large-v3-turbo (MLX), puntuación conservadora y confianza por palabra.
- `turnos.py`: Silero VAD y detección adaptable del fin de turno.
- `escuchar.py`: descarga de modelos y micrófono de la versión de terminal.
- `medir.py`: medición de tiempos por etapa.
- `web/capture.js`: captura y remuestreo a 16 kHz; `web/vad.js`: envío ordenado del audio al servidor local.
- `.vscode/tasks.json`: tarea **Iniciar asistente local**, disponible en Terminal → Ejecutar tarea.

La conexión utiliza exclusivamente `127.0.0.1:1234`. No necesita servicios de IA en la nube.
