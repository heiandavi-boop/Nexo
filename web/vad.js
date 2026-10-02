// Envío ordenado de audio al servidor local: una petición a la vez, sin duplicados ni reordenamientos.
export class ChunkSender {
  constructor(post, onEvent, onError, maxQueued = 64) {
    this.post = post; this.onEvent = onEvent; this.onError = onError; this.maxQueued = maxQueued;
    this.queue = []; this.sending = false; this.token = null;
  }
  get open() { return this.token !== null; }
  start(token) { this.token = token; this.queue = []; }
  close() { this.token = null; this.queue = []; }
  push(chunk) {
    if (!this.open) return;
    this.queue.push(chunk);
    // ~6 s acumulados sin respuesta: el servidor no da abasto, mejor avisar que perder audio en silencio.
    if (this.queue.length > this.maxQueued) { this.close(); this.onError(new Error('El servidor local no responde a tiempo.')); return; }
    this.flush();
  }
  async flush() {
    if (this.sending || !this.open || !this.queue.length) return;
    this.sending = true;
    const token = this.token, parts = this.queue.splice(0);
    const body = new Int16Array(parts.reduce((n, p) => n + p.length, 0));
    let offset = 0; for (const part of parts) { body.set(part, offset); offset += part.length; }
    try {
      const event = await this.post(body, token);
      if (event && token === this.token) this.onEvent(event);
    } catch (error) {
      if (token === this.token) { this.close(); this.onError(error); }
    } finally { this.sending = false; }
    this.flush();
  }
}
