// Captura del micrófono: reduce a 16 kHz mono PCM de 16 bits. La detección de voz ocurre en el servidor local.
export const TARGET_RATE = 16000;
export const CHUNK = 1536; // 96 ms: tres ventanas de Silero VAD

export class Resampler {
  constructor(inputRate, size = CHUNK) {
    this.ratio = inputRate / TARGET_RATE; this.size = size;
    this.sum = 0; this.count = 0; this.need = this.ratio;
    this.out = new Int16Array(size); this.offset = 0;
  }
  // Promedia cada ventana de entrada (filtro paso bajo simple) antes de diezmar.
  push(samples, emit) {
    for (const sample of samples) {
      this.sum += sample; this.count++;
      if (this.count < this.need) continue;
      const value = Math.max(-1, Math.min(1, this.sum / this.count));
      this.out[this.offset++] = value < 0 ? value * 32768 : value * 32767;
      this.need = this.ratio - (this.count - this.need); this.sum = 0; this.count = 0;
      if (this.offset === this.size) { emit(this.out); this.out = new Int16Array(this.size); this.offset = 0; }
    }
  }
}

if (typeof AudioWorkletProcessor !== 'undefined') {
  class Capture extends AudioWorkletProcessor {
    constructor() { super(); this.resampler = new Resampler(sampleRate); }
    process(inputs) {
      const channel = inputs[0]?.[0];
      if (channel) this.resampler.push(channel, chunk => this.port.postMessage(chunk, [chunk.buffer]));
      return true;
    }
  }
  registerProcessor('capture', Capture);
}
