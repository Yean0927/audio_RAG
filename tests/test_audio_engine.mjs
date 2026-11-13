import assert from "node:assert/strict";
import { Pcm16Encoder, VoiceActivityDetector, pcm16Wav } from "../call_web/audio-engine.mjs";
import { VoiceCall } from "../call_web/call.js";

assert.equal(typeof VoiceCall.prototype.start, "function");
assert.equal(typeof VoiceCall.prototype.stop, "function");
const frame = amplitude => new Float32Array(480).fill(amplitude);
const turnEvents = events => events.filter(event => event.type !== "audio").map(event => event.type);
const vad = new VoiceActivityDetector(48000);
const events = [];
for (let i = 0; i < 50; i++) events.push(...vad.feed(frame(0.0001)));
for (let i = 0; i < 60; i++) events.push(...vad.feed(frame(0.05)));
for (let i = 0; i < 100; i++) events.push(...vad.feed(frame(0.0001)));
assert.deepEqual(turnEvents(events), ["start", "utterance"]);
const encoder = new Pcm16Encoder(48000);
const recordedFrames = [...events[0].frames, ...events.filter(event => event.type === "audio").map(event => event.frame)];
const wav = pcm16Wav(recordedFrames.map(samples => encoder.encode(samples)));
const header = new DataView(wav);
assert.equal(header.getUint32(24, true), 16000);
assert.equal(header.getUint16(22, true), 1);
assert.equal(header.getUint16(34, true), 16);
assert.equal(header.getUint32(40, true), wav.byteLength - 44);

// A fractional resampling window may cross any WebSocket message boundary.
const samples = Float32Array.from({ length: 44100 }, (_, i) => Math.sin(i * 0.1));
const whole = pcm16Wav([new Pcm16Encoder(44100).encode(samples)]);
const partial = new Pcm16Encoder(44100);
const parts = [];
for (let offset = 0; offset < samples.length; offset += 1024) {
  parts.push(partial.encode(samples.subarray(offset, offset + 1024)));
}
assert.equal(whole.byteLength, 44 + 32000);
assert.deepEqual(pcm16Wav(parts), whole);

const clicks = new VoiceActivityDetector(48000);
const noise = clicks.feed(frame(0.1));
for (let i = 0; i < 120; i++) noise.push(...clicks.feed(frame(0.0001)));
assert.equal(noise.length, 0);

// Room noise above the old fixed floor must not start or hold a turn open.
const noisyRoom = new VoiceActivityDetector(48000);
const ambientEvents = [];
for (let i = 0; i < 50; i++) ambientEvents.push(...noisyRoom.feed(frame(0.008)));
assert.deepEqual(ambientEvents, []);
for (let i = 0; i < 60; i++) ambientEvents.push(...noisyRoom.feed(frame(0.05)));
for (let i = 0; i < 100; i++) ambientEvents.push(...noisyRoom.feed(frame(0.008)));
assert.deepEqual(turnEvents(ambientEvents), ["start", "utterance"]);

// A fan that starts during speech, plus short clicks during the pause.
const changingRoom = new VoiceActivityDetector(48000);
const changedEvents = [];
for (let i = 0; i < 40; i++) changedEvents.push(...changingRoom.feed(frame(0.001)));
for (let i = 0; i < 60; i++) changedEvents.push(...changingRoom.feed(frame(0.05)));
for (let i = 0; i < 115; i++) changedEvents.push(...changingRoom.feed(frame(i % 30 === 0 ? 0.05 : 0.008)));
assert.deepEqual(turnEvents(changedEvents), ["start", "utterance"]);

const quietRoom = new VoiceActivityDetector(48000, { silenceMs: 350, minSpeechMs: 120, semantic: true });
const quietEvents = [];
for (let i = 0; i < 50; i++) quietEvents.push(...quietRoom.feed(frame(0.001)));
for (let i = 0; i < 26; i++) quietEvents.push(...quietRoom.feed(frame(i % 2 ? 0.004 : 0.007)));
for (let i = 0; i < 40; i++) quietEvents.push(...quietRoom.feed(frame(0.001)));
assert.deepEqual(turnEvents(quietEvents), ["start", "pause"]);

const greetingVad = new VoiceActivityDetector(48000, { semantic: true, minSpeechMs: 120 });
const greetingEvents = [];
for (const [amplitude, count] of [[0.05, 8], [0.001, 3], [0.05, 8]]) {
  for (let i = 0; i < count; i++) greetingEvents.push(...greetingVad.feed(frame(amplitude), true, true));
}
assert.deepEqual(turnEvents(greetingEvents), ["start"]);

const manual = new VoiceActivityDetector(48000);
for (let i = 0; i < 40; i++) manual.feed(frame(0.001));
for (let i = 0; i < 60; i++) manual.feed(frame(0.05));
assert.equal(manual.finish()?.type, "utterance");
assert.equal(manual.finish(), null);

const bounded = new VoiceActivityDetector(48000, { maxSeconds: 1 });
const longEvents = [];
for (let i = 0; i < 300; i++) longEvents.push(...bounded.feed(frame(0.05)));
assert.equal(longEvents.filter(event => event.type === "utterance").length, 1);
for (let i = 0; i < 100; i++) bounded.feed(frame(0.0001));
const next = [];
for (let i = 0; i < 40; i++) next.push(...bounded.feed(frame(0.05)));
for (let i = 0; i < 100; i++) next.push(...bounded.feed(frame(0.0001)));
assert.equal(next.filter(event => event.type === "utterance").length, 1);

// Marking a newer utterance must discard a late reply before any playback.
const call = Object.create(VoiceCall.prototype);
call.active = true; call.epoch = 1; call.turn = 2;
call.showReply = () => { throw new Error("Stale reply should be discarded"); };
call.onMessage({ type: "reply", id: 1, text: "old", audio: "old" }, 1);
call.active = false;
call.onMessage({ type: "reply", id: 2, text: "closed", audio: "closed" }, 1);

// Hanging up must release the actual capture/playback handles, not just the UI.
const elements = new Map();
const makeElement = () => ({
  children: [],
  append(...children) { this.children.push(...children); },
  replaceChildren() { this.children = []; },
});
globalThis.document = {
  body: { dataset: {} }, documentElement: { style: { setProperty() {} } },
  createElement: makeElement,
  getElementById(id) { if (!elements.has(id)) elements.set(id, makeElement()); return elements.get(id); },
};
globalThis.window = {};
globalThis.WebSocket = { OPEN: 1 };

function callAwaitingReply() {
  const call = Object.create(VoiceCall.prototype);
  const messages = [], packets = [], phases = [];
  Object.assign(call, {
    active: true, epoch: 1, turn: 0, turnSequence: 0, playToken: 0, awaitingReply: false, speaking: false, pauseUntil: 0,
    streamPlayers: new Map(),
    autoListen: true,
    vad: new VoiceActivityDetector(48000), context: { sampleRate: 48000 },
    socket: { readyState: 1, bufferedAmount: 0, send(data) {
      if (typeof data === "string") {
        data = JSON.parse(data);
        messages.push(data);
      }
      packets.push(data);
    } },
    phase(name) { phases.push(name); },
    showReply() {},
    showInputClip(wav, trigger) {
      assert.equal(new DataView(wav).getUint32(24, true), 16000);
      assert.equal(trigger, "自动发送");
      const previousEnd = packets.slice(0, -1).findLastIndex(packet => packet.type === "utterance_end");
      const sent = packets.slice(previousEnd + 1, -1).filter(packet => packet instanceof Uint8Array);
      assert.deepEqual(wav, pcm16Wav(sent));
    },
  });
  const feed = (amplitude, count) => {
    for (let i = 0; i < count; i++) call.onAudio(frame(amplitude), 1);
  };
  feed(0.001, 50);
  assert.equal(packets.length, 0);
  feed(0.05, 60);
  assert.deepEqual(messages, [{ type: "utterance_start", id: 1 }]);
  assert.ok(packets[1] instanceof Uint8Array);
  assert.equal(call.awaitingReply, false);
  feed(0.001, 90);
  assert.deepEqual(packets.at(-1), { type: "utterance_end", id: 1 });
  return { call, feed, messages, packets, phases };
}

const turnMessages = id => [{ type: "utterance_start", id }, { type: "utterance_end", id }];
// Even speech-length noise must not supersede a reply, including during decoding.
const guarded = callAwaitingReply();
const feedNoiseTurn = () => {
  guarded.feed(0.05, 60);
  guarded.feed(0.001, 90);
  assert.deepEqual(guarded.messages, turnMessages(1));
};
guarded.call.onMessage({ type: "thinking", id: 1 }, 1);
feedNoiseTurn();
assert.equal(guarded.phases.at(-1), "thinking");

let finishDecode, playback;
const decoded = new Promise(resolve => { finishDecode = resolve; });
const player = { connect() {}, disconnect() {}, start() {} };
Object.assign(guarded.call.context, {
  decodeAudioData() { return decoded; },
  async resume() {},
  createBufferSource() { return player; },
});
guarded.call.playAudio = (...args) => { playback = VoiceCall.prototype.playAudio.apply(guarded.call, args); };
const retrieval = {
  selected_metric: "cosine",
  cosine: [{ rank: 1, score: 0.875, passage: "<FAQ>\n完整问答" }],
  euclidean: [{ rank: 1, score: 1.75, passage: "对照片段" }],
};
guarded.call.onMessage({ type: "reply", id: 1, text: "answer", audio: "AA==", retrieval }, 1);
assert.equal(elements.get("retrieval-check").hidden, false);
assert.equal(elements.get("retrieval-summary").textContent, "第 1 轮检索结果 · Top-1");
const [selected, hits, alternate] = elements.get("retrieval-results").children;
assert.match(selected.textContent, /^用于生成回答.*余弦相似度/);
assert.match(alternate.textContent, /^对照排序.*平方欧氏距离/);
assert.equal(hits.children[0].children[1].textContent, retrieval.cosine[0].passage);
assert.equal(guarded.call.speaking, true);
feedNoiseTurn();
finishDecode({ duration: 0.5 });
await playback;
assert.equal(guarded.phases.at(-1), "speaking");
feedNoiseTurn();

player.onended();
assert.equal(guarded.phases.at(-1), "listening");
guarded.call.pauseUntil = 0; // Advance past the playback echo guard.
guarded.feed(0.05, 60);
guarded.feed(0.001, 90);
assert.deepEqual(guarded.messages, [...turnMessages(1), ...turnMessages(2)]);
guarded.call.onMessage({ type: "reply", id: 1, retrieval }, 1);
assert.equal(elements.get("retrieval-check").hidden, true);
assert.equal(elements.get("retrieval-results").children.length, 0);

const streamed = callAwaitingReply();
const chunks = [];
Object.assign(streamed.call, {
  autoListen: false, maxRecordingSamples: 48000 * 30,
  showInputTokens() {},
  showInputClip(wav, trigger) {
    assert.equal(trigger, "手动发送");
    assert.equal(new DataView(wav).getUint32(40, true), 32000 * 2);
  },
});
Object.assign(streamed.call.context, {
  currentTime: 0,
  createBuffer(channels, length, rate) {
    assert.equal(channels, 1);
    assert.equal(rate, 22050);
    return { duration: length / rate, getChannelData() { return new Float32Array(length); } };
  },
  createBufferSource() {
    return {
      connect() {}, disconnect() {},
      start(at) { this.at = at; chunks.push(this); },
      stop() { this.stopped = true; },
    };
  },
});
const chunk = { type: "audio_chunk", id: 1, pcm: "AQACAA==" };
streamed.call.queryBuffer = { duration: 1.1 };
const acknowledgement = { type: "utterance_received", id: 1 };
streamed.call.onMessage(acknowledgement, 1);
streamed.call.onMessage(acknowledgement, 1);
assert.equal(chunks.length, 1);
assert.equal(chunks[0].buffer, streamed.call.queryBuffer);
assert.equal(streamed.call.receivedAudio, false);
assert.equal(streamed.phases.at(-1), "thinking");
streamed.call.onMessage(chunk, 1);
// A packet arriving just before the scheduled end should still play contiguously.
streamed.call.context.currentTime = streamed.call.nextPlayAt - 0.005;
streamed.call.onMessage(chunk, 1);
assert.equal(streamed.phases.at(-1), "speaking");
assert.equal(chunks[1].at, chunks[0].at + chunks[0].buffer.duration);
assert.equal(chunks[2].at, chunks[1].at + chunks[1].buffer.duration);
assert.equal(streamed.call.awaitingReply, true);
streamed.call.startSpeaking();
assert.deepEqual(streamed.messages, [...turnMessages(1), { type: "interrupt" }, { type: "utterance_start", id: 2 }]);
assert.ok(chunks.every(player => player.stopped));
streamed.call.onMessage(chunk, 1);
streamed.call.onMessage(acknowledgement, 1);
streamed.call.onMessage({ type: "reply", id: 1, text: "stale" }, 1);
assert.equal(chunks.length, 3);
assert.equal(streamed.phases.at(-1), "recording");
streamed.feed(0.0001, 200);
assert.equal(streamed.phases.at(-1), "recording");
assert.ok(streamed.packets.at(-1) instanceof Uint8Array);
streamed.call.finishSpeaking();
assert.deepEqual(streamed.packets.at(-1), { type: "utterance_end", id: 2 });
streamed.call.onMessage({ ...chunk, id: 2 }, 1);
streamed.call.onMessage({ type: "reply", id: 2, text: "finished" }, 1);
assert.equal(streamed.phases.at(-1), "speaking");
chunks.at(-1).onended();
assert.equal(streamed.phases.at(-1), "listening");
streamed.feed(0.05, 100);
assert.equal(streamed.phases.at(-1), "listening");

// Cancelling a short recording drops any PCM that has already been uploaded.
streamed.call.startSpeaking();
streamed.feed(0.05, 9);
assert.ok(streamed.packets.at(-1) instanceof Uint8Array);
streamed.call.finishSpeaking();
assert.deepEqual(streamed.packets.at(-1), { type: "utterance_cancel", id: 3 });
assert.equal(streamed.call.turn, 2);
assert.equal(streamed.call.awaitingReply, false);
streamed.call.startSpeaking();
assert.deepEqual(streamed.packets.at(-1), { type: "utterance_start", id: 4 });

// Full-duplex capture keeps a paused sentence open and ignores obsolete model decisions.
const duplex = Object.create(VoiceCall.prototype);
const duplexPackets = [];
const duplexPlayers = [];
let duplexWav;
Object.assign(duplex, {
  duplex: true, active: true, epoch: 1, turn: 0, turnSequence: 0, playToken: 0,
  context: {
    sampleRate: 48000, currentTime: 0,
    async decodeAudioData() { return { duration: 3 }; },
    async resume() {},
    createBuffer(channels, length, rate) {
      return { duration: length / rate, getChannelData() { return new Float32Array(length); } };
    },
    createBufferSource() {
      const player = {
        connect() {}, disconnect() {},
        start(at, offset) { Object.assign(this, { at, offset }); },
        stop() { this.stopped = true; },
      };
      duplexPlayers.push(player);
      return player;
    },
  },
  maxRecordingSamples: 48000 * 30,
  vad: new VoiceActivityDetector(48000, { silenceMs: 350, minSpeechMs: 120, semantic: true }),
  streamPlayers: new Map(),
  socket: { readyState: 1, bufferedAmount: 0, send(data) {
    duplexPackets.push(typeof data === "string" ? JSON.parse(data) : data);
  } },
  showInputClip(wav, trigger) {
    assert.equal(trigger, "模型确认说完");
    duplexWav = wav;
  },
});
const feedDuplex = (amplitude, count) => {
  for (let i = 0; i < count; i++) duplex.onAudio(frame(amplitude), 1);
};
const duplexMessages = () => duplexPackets.filter(packet => !(packet instanceof Uint8Array));
const probe = (number, id = 2) => ({ type: "utterance_probe", id, probe: number });
const decision = (number, value, id = 2) => ({ type: "turn_decision", id, probe: number, decision: value });
duplex.listen();
assert.equal(elements.get("start-speaking").hidden, true);
assert.equal(elements.get("send-now").hidden, true);
await duplex.playAudio("AA==", 0, 1);
const greeting = duplexPlayers[0];
assert.equal(elements.get("start-speaking").hidden, true);
// Sustained moderate echo and a brief louder noise must leave playback alone.
feedDuplex(0.02, 60);
feedDuplex(0.05, 10);
feedDuplex(0.001, 6);
assert.equal(greeting.stopped, undefined);
assert.deepEqual(duplexMessages(), []);
assert.equal(duplex.vad.noise, 0.001);
// A noise long enough to trigger VAD pauses the greeting at its current offset.
duplex.context.currentTime = greeting.at + 0.5;
for (let i = 0; i < 24; i++) feedDuplex(i % 2 ? 0.035 : 0.05, 1);
assert.equal(greeting.stopped, true);
feedDuplex(0.001, 40);
assert.deepEqual(duplexMessages(), [{ type: "utterance_start", id: 1 }, probe(1, 1)]);
feedDuplex(0.001, 90);
assert.equal(duplexPlayers.length, 1);
duplex.onMessage(decision(1, "discard", 1), 1);
assert.deepEqual(duplexMessages().at(-1), { type: "utterance_cancel", id: 1 });
assert.equal(document.body.dataset.phase, "speaking");
const resumedGreeting = duplexPlayers.at(-1);
assert.equal(resumedGreeting.buffer, greeting.buffer);
assert.equal(resumedGreeting.offset, 0.5);

// A quiet two-syllable onset with a 64 ms gap can interrupt the greeting.
for (const level of [0.008, 0.014, 0.009, 0.001, 0.001, 0.001, 0.009, 0.016, 0.01]) {
  duplex.onAudio(new Float32Array(1024).fill(level), 1);
}
assert.equal(resumedGreeting.stopped, true);
feedDuplex(0.001, 40);
assert.deepEqual(duplexMessages().slice(-2), [{ type: "utterance_start", id: 2 }, probe(1)]);
duplex.onMessage(decision(1, "wait"), 1);
const firstProbePackets = duplexPackets.length;
feedDuplex(0.001, 70);
assert.equal(duplexPackets.length, firstProbePackets);
feedDuplex(0.06, 40);
feedDuplex(0.001, 40);
assert.deepEqual(duplexMessages().slice(-2), [{ type: "utterance_resume", id: 2, probe: 1 }, probe(2)]);
feedDuplex(0.06, 3);
duplex.onMessage(decision(2, "discard"), 1);
assert.equal(duplex.awaitingReply, false);
assert.equal(duplexPlayers.length, 2);
feedDuplex(0.06, 37);
feedDuplex(0.001, 40);
assert.deepEqual(duplexMessages().slice(-2), [{ type: "utterance_resume", id: 2, probe: 2 }, probe(3)]);
duplex.onMessage(decision(2, "reply"), 1);
assert.equal(duplex.awaitingReply, false);
duplex.onMessage(decision(3, "reply"), 1);
assert.equal(duplex.awaitingReply, true);
assert.deepEqual(duplexMessages().at(-1), { type: "utterance_end", id: 2, probe: 3 });
const secondStart = duplexPackets.findIndex(packet => packet.type === "utterance_start" && packet.id === 2);
assert.deepEqual(duplexWav, pcm16Wav(duplexPackets.slice(secondStart).filter(packet => packet instanceof Uint8Array)));
duplex.onMessage({ type: "retrieval", id: 2, retrieval }, 1);
assert.equal(elements.get("retrieval-check").hidden, false);
assert.equal(elements.get("retrieval-summary").textContent, "第 2 轮检索结果 · Top-1");

// Keep queued chunks and accept new chunks while classifying the interruption.
const duplexChunk = { type: "audio_chunk", id: 2, pcm: btoa("\x01\x00".repeat(22050)) };
duplex.onMessage(duplexChunk, 1);
duplex.onMessage(duplexChunk, 1);
duplex.onMessage(duplexChunk, 1);
const [playedChunk, firstChunk, queuedChunk] = duplexPlayers.slice(-3);
duplex.context.currentTime = firstChunk.at + 0.25;
const replyPackets = duplexPackets.length;
feedDuplex(0.022, 60);
feedDuplex(0.001, 100);
assert.equal(duplexPackets.length, replyPackets);
assert.equal(firstChunk.stopped, undefined);
feedDuplex(0.06, 30);
feedDuplex(0.001, 40);
assert.ok(playedChunk.stopped && firstChunk.stopped && queuedChunk.stopped);
assert.deepEqual(duplexMessages().slice(-2), [{ type: "utterance_start", id: 3 }, probe(1, 3)]);
duplex.onMessage(duplexChunk, 1);
duplex.onMessage({ type: "reply", id: 2, text: "answer" }, 1);
const playersBeforeResume = duplexPlayers.length;
assert.equal(playersBeforeResume, 5);
assert.equal(document.body.dataset.phase, "recording");
// Silence alone must not restart playback while the probe still awaits a decision.
feedDuplex(0.001, 160);
assert.equal(duplexPlayers.length, playersBeforeResume);
assert.equal(document.body.dataset.phase, "recording");
duplex.onMessage(decision(1, "discard", 3), 1);
const resumedChunks = duplexPlayers.slice(playersBeforeResume);
assert.equal(resumedChunks.length, 3);
assert.equal(resumedChunks[0].buffer, firstChunk.buffer);
assert.equal(resumedChunks[1].buffer, queuedChunk.buffer);
assert.deepEqual(resumedChunks.map(player => player.offset), [0.25, 0, 0]);
assert.equal(resumedChunks[1].at, resumedChunks[0].at + 0.75);
assert.equal(resumedChunks[2].at, resumedChunks[1].at + 1);
assert.equal(duplex.turn, 2);
assert.equal(duplex.recording, null);
assert.equal(document.body.dataset.phase, "speaking");
assert.deepEqual(duplexMessages().slice(-3), [
  { type: "utterance_start", id: 3 }, probe(1, 3), { type: "utterance_cancel", id: 3 },
]);
assert.equal(elements.get("retrieval-check").hidden, false);

// A quiet utterance can take over during playback, and the uploaded PCM is amplified.
const fourthStart = duplexPackets.length;
for (let i = 0; i < 26; i++) feedDuplex(i % 2 ? 0.01 : 0.018, 1);
feedDuplex(0.001, 40);
assert.deepEqual(duplexMessages().at(-1), probe(1, 4));
const speechPcm = new DataView(duplexPackets.slice(fourthStart).find(packet => packet instanceof Uint8Array).buffer);
assert.ok(speechPcm.getInt16(0, true) > Math.round(0.018 * 32767));
duplex.onMessage(decision(1, "reply", 4), 1);
assert.deepEqual(duplexMessages().at(-1), { type: "utterance_end", id: 4, probe: 1 });
assert.ok(resumedChunks.every(player => player.stopped));
duplex.onMessage(duplexChunk, 1);
const playersBeforeFullReply = playersBeforeResume + resumedChunks.length;
assert.equal(duplexPlayers.length, playersBeforeFullReply);
assert.equal(document.body.dataset.phase, "thinking");
assert.equal(elements.get("send-now").hidden, true);

// A full reply that finishes decoding during a probe must wait to start playing.
feedDuplex(0.06, 30);
feedDuplex(0.001, 40);
let fullReplyPlayback;
duplex.playAudio = (...args) => { fullReplyPlayback = VoiceCall.prototype.playAudio.apply(duplex, args); };
duplex.onMessage({ type: "reply", id: 4, text: "完整回复", audio: "AA==" }, 1);
await fullReplyPlayback;
assert.equal(duplexPlayers.length, playersBeforeFullReply);
assert.equal(document.body.dataset.phase, "recording");
duplex.onMessage(decision(1, "discard", 5), 1);
assert.equal(duplexPlayers.length, playersBeforeFullReply + 1);
assert.equal(duplexPlayers.at(-1).offset, 0);
assert.equal(document.body.dataset.phase, "speaking");

// A full reply already in progress also resumes at the interruption point.
const fullReply = duplexPlayers.at(-1);
duplex.context.currentTime = fullReply.at + 0.5;
feedDuplex(0.06, 30);
feedDuplex(0.001, 40);
duplex.onMessage(decision(1, "discard", 6), 1);
assert.equal(fullReply.stopped, true);
assert.equal(duplexPlayers.at(-1).buffer, fullReply.buffer);
assert.equal(duplexPlayers.at(-1).offset, 0.5);

// A recognized phrase gets a bounded grace period when the model predicts more speech.
duplex.showInputClip = (_, trigger) => assert.equal(trigger, "停顿后自动发送");
feedDuplex(0.06, 30);
feedDuplex(0.001, 40);
duplex.onMessage(decision(1, "wait", 7), 1);
assert.equal(duplex.turn, 4);
feedDuplex(0.001, 79);
assert.equal(duplex.turn, 4);
feedDuplex(0.001, 1);
assert.deepEqual(duplexMessages().at(-1), { type: "utterance_end", id: 7, probe: 1, after_wait: true });

// Playback tails must not seed a fresh duplex turn after the speaker stops.
const echoCall = Object.create(VoiceCall.prototype);
const echoMessages = [];
Object.assign(echoCall, {
  duplex: true, active: true, epoch: 1, playToken: 0, speaking: true,
  turnSequence: 0, context: { sampleRate: 48000 },
  maxRecordingSamples: 48000 * 30,
  vad: new VoiceActivityDetector(48000, { semantic: true }),
  streamPlayers: new Map([[{}, {}]]),
  socket: { readyState: 1, bufferedAmount: 0, send(data) {
    if (typeof data === "string") echoMessages.push(JSON.parse(data));
  } },
  phase() {},
});
for (let i = 0; i < 40; i++) echoCall.onAudio(frame(0.001), 1);
for (let i = 0; i < 15; i++) echoCall.onAudio(frame(0.05), 1);
assert.deepEqual(echoMessages, []);
echoCall.streamPlayers.clear();
echoCall.playbackEnded(0, 1);
assert.equal(echoCall.vad.candidateMs, 0);
// Hold the clock inside the guard without relying on test execution speed.
echoCall.pauseUntil = Infinity;
for (let i = 0; i < 60; i++) echoCall.onAudio(frame(0.05), 1);
assert.deepEqual(echoMessages, []);
assert.equal(echoCall.vad.preSamples, 0);
echoCall.pauseUntil = 0;
for (let i = 0; i < 60; i++) echoCall.onAudio(frame(i % 2 ? 0.0055 : 0.0075), 1);
assert.deepEqual(echoMessages, []);
for (let i = 0; i < 30; i++) echoCall.onAudio(frame(0.05), 1);
assert.deepEqual(echoMessages, [{ type: "utterance_start", id: 1 }]);
// An utterance already in progress must survive the end of playback.
const liveRecording = echoCall.recording;
echoCall.playbackEnded(0, 1);
assert.equal(echoCall.recording, liveRecording);
assert.equal(echoCall.vad.active, true);

// Keep speech that begins just before the greeting ends.
echoCall.recording = null;
echoCall.turn = 0;
echoCall.speaking = true;
echoCall.pauseUntil = 0;
echoCall.vad.reset();
echoCall.streamPlayers.set({}, {});
for (let i = 0; i < 7; i++) echoCall.onAudio(frame(0.05), 1);
assert.equal(echoMessages.length, 1);
echoCall.streamPlayers.clear();
echoCall.playbackEnded(0, 1);
for (let i = 0; i < 5; i++) echoCall.onAudio(frame(0.05), 1);
assert.deepEqual(echoMessages.at(-1), { type: "utterance_start", id: 2 });

const ended = { tracks: 0, socket: 0, context: 0, nodes: 0, playback: 0 };
const sent = [];
const connected = Object.create(VoiceCall.prototype);
Object.assign(connected, {
  active: true, epoch: 1, playToken: 0,
  socket: { readyState: 1, send(data) { sent.push(JSON.parse(data)); }, close() { ended.socket++; } },
  streamPlayers: new Map([[{ stop() { ended.playback++; }, disconnect() {} }, {}]]),
  capture: { port: {}, disconnect() { ended.nodes++; } },
  source: { disconnect() { ended.nodes++; } }, silent: { disconnect() { ended.nodes++; } },
  stream: { getTracks() { return [{ stop() { ended.tracks++; } }]; } },
  context: { close() { ended.context++; return Promise.resolve(); } },
});
connected.stop();
assert.deepEqual(ended, { tracks: 1, socket: 1, context: 1, nodes: 3, playback: 1 });
assert.deepEqual(sent, [{ type: "hangup" }]);
assert.equal(connected.active, false);
assert.equal(connected.stream, null);
assert.equal(connected.context, null);
assert.equal(connected.capture, null);
delete globalThis.document; delete globalThis.window; delete globalThis.WebSocket;
