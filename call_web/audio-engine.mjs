// Pure audio/turn-detection functions, shared with the protocol tests.
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
const GREETING_ONSET_MS = 120;
const GREETING_SYLLABLE_GAP_MS = 90;
const STEADY_NOISE_MS = 200;
const MAX_NOISE_LEVEL = 0.03;
const PRE_ROLL_MS = 700;

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

export function pcm16Wav(chunks, maxBytes = Infinity) {
  const size = Math.min(maxBytes, chunks.reduce((total, pcm) => total + pcm.byteLength, 0));
  const buffer = new ArrayBuffer(44 + size);
  const view = new DataView(buffer);
  const text = (offset, value) => [...value].forEach((char, i) => view.setUint8(offset + i, char.charCodeAt(0)));
  text(0, "RIFF"); view.setUint32(4, 36 + size, true); text(8, "WAVE");
  text(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, 1, true); view.setUint32(24, INPUT_SAMPLE_RATE, true);
  view.setUint32(28, INPUT_SAMPLE_RATE * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
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

export class VoiceActivityDetector {
  constructor(rate, { silenceMs = 850, minSpeechMs = 260, maxSeconds = 30, semantic = false, modelEndpoint = false } = {}) {
    this.rate = rate; this.silenceMs = silenceMs; this.minSpeechMs = minSpeechMs;
    this.maxSeconds = maxSeconds;
    this.semantic = semantic;
    this.modelEndpoint = modelEndpoint;
    this.noise = 0.001;
    this.calibrationMs = 0;
    this.calibrationMinimum = Infinity;
    this.reset();
  }
  reset({ keepPreRoll = false } = {}) {
    this.active = false;
    if (!keepPreRoll) {
      this.preRoll = [];
      this.preSamples = 0;
    }
    this.candidateMs = 0; this.voicedMs = 0; this.quietMs = 0; this.samples = 0;
    this.candidateGapMs = 0;
    this.candidatePeak = 0;
    this.weakPeak = 0;
    this.weakFloor = Infinity;
    this.strongMs = 0;
    this.strongGapMs = 0;
    this.waitForQuiet = false;
    this.paused = false;
    this.voicedRunMs = 0;
    this.stableFloor = 0;
    this.stablePeak = 0;
    this.stableMs = 0;
  }

  bufferPreRoll(frame) {
    this.preRoll.push(frame);
    this.preSamples += frame.length;
    const maxSamples = this.rate * PRE_ROLL_MS / 1000;
    while (this.preSamples > maxSamples && this.preRoll.length > 1) {
      this.preSamples -= this.preRoll.shift().length;
    }
  }

  finish() {
    if (!this.active) return null;
    const event = { type: this.voicedMs >= this.minSpeechMs ? "utterance" : "discard" };
    this.reset();
    return event;
  }

  get hasGreetingOnset() {
    return this.candidateGapMs < GREETING_SYLLABLE_GAP_MS && (this.strongMs >= 60
      || (this.candidateMs >= 60 && this.weakPeak >= this.weakFloor * 1.5));
  }

  feed(frame, playing = false, greeting = false) {
    const level = rms(frame), ms = frame.length * 1000 / this.rate, events = [];
    const calibrating = this.calibrationMs < 180;
    // Keep listening during duplex playback and avoid learning the caller's
    // first words or playback echo as the room noise level.
    const calibrationOnly = calibrating && (!this.semantic || (!playing && !greeting && !this.active && this.candidateMs === 0));
    if (calibrating) this.calibrationMs += ms;
    if (calibrationOnly) {
      this.calibrationMinimum = Math.min(this.calibrationMinimum, level);
      this.noise = Math.max(0.001, Math.min(0.02, this.calibrationMinimum));
    }
    // Track the full range so small fluctuations can count as steady, without
    // absorbing a syllable whose level changes substantially over the run.
    if (level <= MAX_NOISE_LEVEL) {
      const floor = Math.min(level, this.stableFloor);
      const peak = Math.max(level, this.stablePeak);
      if (this.stableMs && peak <= Math.max(floor + 0.0005, floor * 1.6)) {
        this.stableFloor = floor;
        this.stablePeak = peak;
        this.stableMs += ms;
      } else {
        this.stableFloor = this.stablePeak = level;
        this.stableMs = ms;
      }
    } else {
      this.stableMs = 0;
    }
    // Playback can contain echo and must not raise the room noise estimate.
    if (!playing && this.stableMs >= STEADY_NOISE_MS) {
      this.noise = Math.max(this.noise, (this.stableFloor + this.stablePeak) / 2);
    }
    const roomThreshold = Math.max(this.semantic ? 0.003 : 0.005, this.noise * 1.5 + 0.001);
    const strongThreshold = playing ? Math.max(0.025, roomThreshold * 2) : Math.max(0.008, roomThreshold);
    const threshold = playing ? strongThreshold : roomThreshold;
    const voiced = level > threshold;
    if (this.waitForQuiet) {
      this.quietMs = voiced ? 0 : this.quietMs + ms;
      if (this.quietMs > 400) this.reset();
      return events;
    }
    if (!this.active) {
      if (!playing && !calibrating && !voiced) {
        this.noise = this.noise * 0.97 + Math.min(level, MAX_NOISE_LEVEL) * 0.03;
      }
      this.bufferPreRoll(frame);
      if (calibrationOnly) return events;
      const greetingPlayback = this.semantic && playing && greeting;
      const steady = this.stableMs >= STEADY_NOISE_MS / 2;
      const onsetMs = playing ? (greetingPlayback ? GREETING_ONSET_MS : 240) : (steady ? STEADY_NOISE_MS : 110);
      const candidateGapLimit = greetingPlayback ? GREETING_SYLLABLE_GAP_MS : 50;
      let started;
      if (this.semantic) {
        const candidateThreshold = playing ? roomThreshold * (greetingPlayback ? 1.5 : 2) : roomThreshold;
        const onsetLevel = playing && steady ? 0 : level;
        if (onsetLevel > strongThreshold) {
          this.strongMs += ms;
          this.strongGapMs = 0;
        } else {
          this.strongGapMs += ms;
          if (!greetingPlayback || this.strongGapMs >= candidateGapLimit) this.strongMs = 0;
        }
        if (onsetLevel > candidateThreshold) {
          this.candidateMs += ms;
          this.candidateGapMs = 0;
          this.candidatePeak = Math.max(this.candidatePeak, level);
          if (level <= strongThreshold) {
            this.weakPeak = Math.max(this.weakPeak, level);
            this.weakFloor = Math.min(this.weakFloor, level);
          }
        } else {
          this.candidateGapMs += ms;
          if (this.candidateGapMs >= candidateGapLimit) {
            this.candidateMs = 0;
            this.candidatePeak = 0;
            this.weakPeak = 0;
            this.weakFloor = Infinity;
          }
        }
        // Give quieter speech more time to show an amplitude envelope distinct from steady noise.
        started = (onsetLevel > strongThreshold && this.strongMs >= onsetMs)
          || (this.candidateGapMs === 0 && this.candidateMs >= (playing ? onsetMs : Math.max(onsetMs, 150))
          && this.weakPeak >= this.weakFloor * 1.5);
      } else {
        this.candidateMs = voiced ? this.candidateMs + ms : 0;
        started = this.candidateMs >= onsetMs;
      }
      if (started) {
        this.active = true; this.samples = this.preSamples;
        this.voicedMs = this.candidateMs; this.quietMs = 0;
        if (this.semantic) this.voicedRunMs = this.candidateMs;
        events.push({ type: "start", frames: this.preRoll, peakLevel: this.candidatePeak });
        this.preRoll = []; this.preSamples = 0;
      }
      return events;
    }
    events.push({ type: "audio", frame });
    this.samples += frame.length;
    const full = this.samples >= this.rate * (this.maxSeconds - 0.1);
    if (this.modelEndpoint) {
      if (full) {
        events.push({ type: "limit" });
        this.reset();
        this.waitForQuiet = true;
      }
      return events;
    }
    this.quietMs += ms;
    this.voicedRunMs = voiced ? this.voicedRunMs + ms : 0;
    if (voiced) this.voicedMs += ms;
    // Brief clicks should not restart the end-of-speech timer.
    const continuationMs = this.paused && playing ? 240 : 65;
    if (this.voicedRunMs >= continuationMs) this.quietMs = 0;
    if (this.semantic) {
      if (this.paused && this.voicedRunMs >= continuationMs) {
        this.paused = false;
        events.push({ type: "resume" });
      }
      if (full) {
        events.push({ type: "limit" });
        this.reset();
        this.waitForQuiet = true;
      } else if (!this.paused && this.quietMs >= this.silenceMs) {
        if (this.voicedMs < this.minSpeechMs) events.push(this.finish());
        else {
          this.paused = true;
          events.push({ type: "pause" });
        }
      }
      return events;
    }
    if (this.quietMs >= this.silenceMs || full) {
      events.push(this.finish());
      if (full) this.waitForQuiet = true;
    }
    return events;
  }
}
