// Audio encoding and level measurement, shared with the protocol tests.
export function rms(samples) {
  let sum = 0;
  for (const value of samples) sum += value * value;
  return Math.sqrt(sum / Math.max(1, samples.length));
}

export function joinFrames(frames) {
  const result = new Float32Array(frames.reduce((sum, frame) => sum + frame.length, 0));
  let offset = 0;
  for (const frame of frames) { result.set(frame, offset); offset += frame.length; }
  return result;
}

export const INPUT_SAMPLE_RATE = 16000;

export class Pcm16Encoder {
  constructor(sourceRate, gain = 1) {
    this.sourceRate = sourceRate;
    this.gain = gain;
    this.ratio = sourceRate / INPUT_SAMPLE_RATE;
    this.sourceSamples = 0;
    this.outputSamples = 0;
    this.carry = new Float32Array(0);
  }

  encode(frame) {
    const offset = this.sourceSamples - this.carry.length;
    const samples = this.carry.length ? joinFrames([this.carry, frame]) : frame;
    this.sourceSamples += frame.length;
    const endSample = Math.floor(this.sourceSamples * INPUT_SAMPLE_RATE / this.sourceRate);
    const pcm = new Uint8Array((endSample - this.outputSamples) * 2);
    const view = new DataView(pcm.buffer);
    for (let i = this.outputSamples, byte = 0; i < endSample; i++, byte += 2) {
      const start = Math.floor(i * this.ratio) - offset;
      const end = Math.min(samples.length, Math.max(start + 1, Math.floor((i + 1) * this.ratio) - offset));
      let value = 0;
      for (let j = start; j < end; j++) value += samples[j];
      value = Math.max(-1, Math.min(1, this.gain * value / (end - start)));
      view.setInt16(byte, Math.round(value * (value < 0 ? 32768 : 32767)), true);
    }
    // Keep the incomplete resampling window for the next capture batch.
    this.carry = samples.slice(Math.floor(endSample * this.ratio) - offset);
    this.outputSamples = endSample;
    return pcm;
  }
}

export function pcm16Wav(chunks, maxBytes = Infinity, sampleRate = INPUT_SAMPLE_RATE) {
  const size = Math.min(maxBytes, chunks.reduce((total, pcm) => total + pcm.byteLength, 0));
  const buffer = new ArrayBuffer(44 + size);
  const view = new DataView(buffer);
  const text = (offset, value) => [...value].forEach((char, i) => view.setUint8(offset + i, char.charCodeAt(0)));
  text(0, "RIFF"); view.setUint32(4, 36 + size, true); text(8, "WAVE");
  text(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  text(36, "data"); view.setUint32(40, size, true);
  const bytes = new Uint8Array(buffer);
  let offset = 44;
  for (const pcm of chunks) {
    const part = pcm.subarray(0, bytes.length - offset);
    bytes.set(part, offset);
    offset += part.byteLength;
    if (offset === bytes.length) break;
  }
  return buffer;
}
