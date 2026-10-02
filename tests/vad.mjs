import assert from 'node:assert/strict';
import {Resampler, CHUNK} from '../web/capture.js';
import {ChunkSender} from '../web/vad.js';

// Remuestreo a 16 kHz desde las frecuencias habituales del Mac.
for (const rate of [48000, 44100, 16000]) {
  const chunks = [], resampler = new Resampler(rate);
  const input = new Float32Array(rate); // 1 s
  for (let i = 0; i < input.length; i++) input[i] = 0.5 * Math.sin(2 * Math.PI * 220 * i / rate);
  for (let i = 0; i < input.length; i += 128) resampler.push(input.subarray(i, i + 128), c => chunks.push(c));
  const total = chunks.length * CHUNK + resampler.offset;
  assert.ok(Math.abs(total - 16000) <= 2, `${rate} Hz → ${total} muestras`);
  assert.ok(chunks.every(c => c instanceof Int16Array && c.length === CHUNK));
  const peak = Math.max(...chunks.flatMap(c => [...c]));
  assert.ok(peak > 15000 && peak <= 32767, `Amplitud conservada en ${rate} Hz`);
}

// Envío secuencial: nunca dos peticiones simultáneas, sin duplicar ni perder fragmentos.
const sent = [], events = [];
let inFlight = 0, maxInFlight = 0, release;
const sender = new ChunkSender(async (body, token) => {
  inFlight++; maxInFlight = Math.max(maxInFlight, inFlight);
  sent.push([...body]);
  await new Promise(r => { release = r; });
  inFlight--;
  return {token};
}, e => events.push(e), e => { throw e; });
sender.push(new Int16Array([1])); // cerrado: se ignora
sender.start('a');
sender.push(new Int16Array([1, 2]));
sender.push(new Int16Array([3]));
sender.push(new Int16Array([4]));
await new Promise(r => setTimeout(r, 0));
release(); await new Promise(r => setTimeout(r, 0));
release(); await new Promise(r => setTimeout(r, 0));
assert.deepEqual(sent, [[1, 2], [3, 4]], 'Agrupa lo pendiente en orden y no repite');
assert.equal(maxInFlight, 1);
assert.equal(events.length, 2);

// Al cerrar (turno detectado o pausa) se descarta lo pendiente y las respuestas antiguas se ignoran.
sender.push(new Int16Array([5]));
sender.push(new Int16Array([6]));
sender.close();
release(); await new Promise(r => setTimeout(r, 0));
assert.deepEqual(sent.at(-1), [5]);
assert.equal(events.length, 2, 'Respuesta de una sesión cerrada ignorada');
sender.push(new Int16Array([7]));
assert.equal(sent.length, 3, 'No envía tras cerrar');

// Si el servidor se atasca, avisa en vez de acumular audio indefinidamente.
let failed = null;
const stuck = new ChunkSender(() => new Promise(() => {}), () => {}, e => { failed = e; }, 3);
stuck.start('b');
for (let i = 0; i < 5; i++) stuck.push(new Int16Array([i]));
assert.ok(failed && !stuck.open);
console.log('OK: remuestreo 48/44,1/16 kHz, envío ordenado, sin duplicados, cierre y saturación.');
