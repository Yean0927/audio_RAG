import { INPUT_SAMPLE_RATE, Pcm16Encoder, joinFrames, pcm16Wav, rms } from "./audio-engine.mjs?v=semantic-vad";
import { CallHistory } from "./call-history.mjs?v=faq-links";

const GREETING_PREROLL_SECONDS = 0.25;
const STREAM_PLAYBACK_LEAD_SECONDS = 0.12;
const INTERRUPT_HINT = "可以直接插话。";
const semanticCommitLabels = {
  semantic_complete: "Smart Turn 确认说完",
  silence_timeout: "静音超时提交",
  recording_limit: "达到录音时限",
};
const $ = id => document.getElementById(id);
const decodeBase64 = encoded => Uint8Array.from(atob(encoded), char => char.charCodeAt(0));
const titles = {
  idle: ["语义 VAD 通话", "接通后直接说话，由 Smart Turn 判断何时回答"],
  connecting: ["正在接通", "正在为您建立语音连接"],
  listening: ["我在听，请讲", "接通后可以直接说话"],
  recording: ["正在聆听", "您可以自然地表达您的需求"],
  thinking: ["正在处理语音", "请稍等片刻"],
  speaking: ["客服正在回复", "可以直接插话"],
  error: ["暂时无法接通", "请稍后重试"],
};

export class VoiceCall {
  constructor() {
    this.active = false; this.epoch = 0; this.turn = 0; this.playToken = 0;
    this.health = null; this.speaking = false;
    this.awaitingReply = false;
    this.streamPlayers = new Map();
    $("call-button").addEventListener("click", () => this.toggle());
    this.history = new CallHistory(player => this.registerPreview(player));
    window.addEventListener("pagehide", () => this.stop());
    this.phase("idle");
    this.refreshHealth();
    this.poll = setInterval(() => { if (!this.active) this.refreshHealth(); }, 2500);
  }

  phase(name, caption = "") {
    document.body.dataset.phase = name;
    const [heading, description] = titles[name] || titles.idle;
    $("heading").textContent = heading; $("description").textContent = description;
    if (caption) $("caption").textContent = caption;
  }

  async refreshHealth() {
    if (this.active) return;
    try {
      const response = await fetch("/api/health", { cache: "no-store" });
      if (!response.ok) throw new Error("连接失败");
      const data = await response.json();
      if (this.active) return;
      this.health = data;
      $("brand").textContent = data.title || "语音客服";
      document.title = data.title || "语音客服";
      const ready = data.state === "ready" && data.available;
      $("service-dot").classList.toggle("ready", ready);
      $("service-label").textContent = ready ? "在线" : data.state === "error" ? "暂未就绪" : data.available ? "正在准备" : "正在通话";
      $("call-button").disabled = data.state === "loading" || (data.state === "ready" && !data.available);
      $("button-label").textContent = ready ? "开始通话" : data.state === "error" ? "重新连接" : data.available ? "客服准备中" : "客服正在通话";
      if (ready && document.body.dataset.phase !== "error") $("caption").textContent = "点击后允许麦克风权限，即可开始交流。";
      else if (data.state === "loading") $("caption").textContent = "客服正在准备，请稍候。";
      else if (data.state === "error") { this.phase("error", "客服暂未就绪，请稍后重新连接。"); }
    } catch (_) {
      if (this.active) return;
      this.health = null;
      $("service-dot").classList.remove("ready"); $("service-label").textContent = "连接中";
      $("call-button").disabled = false; $("button-label").textContent = "重新连接";
      $("caption").textContent = "暂时无法连接客服，请稍后重试。";
    }
  }

  async toggle() {
    if (this.active) {
      this.stop({ playFarewell: true });
      return;
    }
    if (!this.health || this.health.state !== "ready") {
      $("call-button").disabled = true;
      try { await fetch("/api/retry", { method: "POST" }); } catch (_) { /* health shows connection status */ }
      await this.refreshHealth();
      return;
    }
    await this.start();
  }

  async start() {
    this.stopFarewell();
    this.activePreview?.pause();
    const epoch = ++this.epoch;
    this.active = true; this.turn = 0;
    this.awaitingReply = false;
    this.previewPauseUntil = 0;
    this.phase("connecting", "请允许浏览器使用麦克风。");
    $("call-button").disabled = false; $("button-label").textContent = "挂断通话";
    try {
      if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) throw new Error("请通过 localhost 或 HTTPS 打开页面后使用麦克风。");
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      if (!AudioContext) throw new Error("当前浏览器暂不支持语音通话。");
      // Create/resume on the button gesture so later replies can play automatically.
      const context = new AudioContext(); this.context = context;
      if (!context.audioWorklet || !window.AudioWorkletNode) throw new Error("当前浏览器暂不支持语音通话。");
      await context.resume();
      const hello = await this.connectSocket(epoch);
      if (!this.active || this.epoch !== epoch) return;
      if (!hello.semantic_vad) throw new Error("服务器尚未启用语义 VAD。");
      const farewellUrl = hello.farewell_audio_url;
      this.farewellAudio = farewellUrl ? this.decodeFarewell(farewellUrl, context) : null;
      this.outputSampleRate = hello.output_sample_rate || 22050;
      this.queryBuffer = hello.query_pcm ? this.pcmBuffer(decodeBase64(hello.query_pcm)) : null;
      this.semanticCommitReason = null;
      const stream = await navigator.mediaDevices.getUserMedia({ audio: {
        channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: false,
      } });
      if (!this.active || this.epoch !== epoch) { stream.getTracks().forEach(track => track.stop()); return; }
      this.stream = stream;
      await context.audioWorklet.addModule("/assets/capture-worklet.js");
      if (!this.active || this.epoch !== epoch) return;
      this.micEncoder = new Pcm16Encoder(context.sampleRate);
      this.micFrames = [];
      this.micSamples = 0;
      this.source = context.createMediaStreamSource(stream);
      this.capture = new AudioWorkletNode(context, "microphone-capture");
      this.silent = context.createGain(); this.silent.gain.value = 0;
      this.source.connect(this.capture).connect(this.silent).connect(context.destination);
      this.capture.port.onmessage = event => this.onAudio(new Float32Array(event.data), epoch);
      stream.getAudioTracks().forEach(track => track.addEventListener("ended", () => {
        if (this.active && this.epoch === epoch) this.fail("麦克风连接已断开，请重新开始通话。");
      }));
      this.startedAt = Date.now(); this.timer = setInterval(() => this.updateTimer(), 1000);
      this.updateTimer();
      this.activePreview?.pause();
      this.activePreview = null;
      this.history.clear();
      this.listen();
      if (hello.greeting_audio) this.playAudio(decodeBase64(hello.greeting_audio), 0, epoch);
    } catch (error) {
      if (this.epoch !== epoch) return;
      const message = error.name === "NotAllowedError" ? "麦克风权限未开启，请允许后重新开始通话。"
        : error.name === "NotFoundError" ? "没有找到麦克风，请连接麦克风后重试。"
        : error.message || "通话连接失败，请重试。";
      this.fail(message);
    }
  }

  connectSocket(epoch) {
    return new Promise((resolve, reject) => {
      const url = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/api/call?stream=1&ack=1&vad=semantic`;
      const socket = new WebSocket(url); this.socket = socket;
      let ready = false;
      const timeout = setTimeout(() => { reject(new Error("客服连接超时，请重试。")); socket.close(); }, 15000);
      socket.onmessage = event => {
        if (this.epoch !== epoch) return;
        let message;
        try { message = JSON.parse(event.data); } catch (_) { return; }
        if (message.type === "ready") { ready = true; clearTimeout(timeout); resolve(message); return; }
        if (message.type === "error" && !ready) { clearTimeout(timeout); reject(new Error(message.message)); return; }
        this.onMessage(message, epoch);
      };
      socket.onerror = () => { clearTimeout(timeout); if (!ready) reject(new Error("无法建立语音连接，请重试。")); };
      socket.onclose = () => {
        clearTimeout(timeout);
        if (!ready) reject(new Error("客服暂时无法接通，请稍后重试。"));
        if (this.active && this.epoch === epoch) this.fail("通话连接已断开，请重新开始通话。");
      };
    });
  }

  onAudio(frame, epoch) {
    if (!this.active || this.epoch !== epoch || !this.micEncoder) return;
    this.uploadSemanticAudio(frame);
  }

  uploadSemanticAudio(frame) {
    if (this.previewPlaying || performance.now() < this.previewPauseUntil) {
      this.micFrames = [];
      this.micSamples = 0;
      return;
    }
    const volume = Math.min(1, rms(frame) * 15);
    document.documentElement.style.setProperty("--level", volume.toFixed(3));
    this.micFrames.push(frame);
    this.micSamples += frame.length;
    if (this.micSamples < this.context.sampleRate * 0.08) return;
    if (this.socket?.readyState !== WebSocket.OPEN || this.socket.bufferedAmount >= 64 * 1024) {
      this.fail("语音连接不稳定，请重新开始通话。");
      return;
    }
    const pcm = this.micEncoder.encode(joinFrames(this.micFrames));
    this.micFrames = [];
    this.micSamples = 0;
    if (pcm.byteLength) this.socket.send(pcm);
  }

  listen(message) {
    this.phase("listening", message || "您可以直接说话。");
  }

  markSubmitted(turn) {
    this.recording = null;
    this.stopPlayback();
    this.turn = turn;
    this.awaitingReply = true;
    this.receivedAudio = false;
    this.history?.begin(turn, this.outputSampleRate);
  }

  onMessage(message, epoch) {
    if (!this.active || this.epoch !== epoch) return;
    if (message.type === "speech_started") {
      this.pauseReply();
      this.recording = { id: message.id };
      this.phase("recording", "我在听，请继续说。");
      return;
    }
    if (message.type === "speech_discarded") {
      if (this.recording?.id !== message.id) return;
      this.recording = null;
      this.resumeReply(message.message);
      return;
    }
    if (message.type === "speech_stopped") {
      if (this.recording?.id !== message.id) return;
      this.markSubmitted(message.id);
      this.semanticCommitReason = message.reason;
      this.phase("thinking", "正在确认这段语音。");
      return;
    }
    if (message.id != null && message.id !== this.turn) return;
    if (message.type === "turn_ignored") {
      this.history?.finish({ status: "本轮已跳过" });
      this.awaitingReply = false;
      if (!this.recording) this.listen();
    } else if (message.type === "utterance_received") {
      if (message.input_pcm) {
        const label = semanticCommitLabels[this.semanticCommitReason];
        this.showInputClip(pcm16Wav([decodeBase64(message.input_pcm)]), label || "送入模型的录音");
      }
      if (!message.clarify && !message.courtesy && this.awaitingReply && this.queryBuffer && !this.speaking) {
        this.queueBuffer(this.queryBuffer, epoch);
        if (!this.recording) this.phase("thinking", `正在为您查询，${INTERRUPT_HINT}`);
      }
    } else if (message.type === "thinking") {
      if (!this.speaking && !this.recording) this.phase("thinking", `正在为您整理回复，${INTERRUPT_HINT}`);
    } else if (message.type === "audio_chunk" && this.awaitingReply) {
      this.queueAudio(message.pcm, epoch);
    } else if (message.type === "retrieval") {
      this.history?.setRetrieval(message.retrieval, message.links);
    } else if (message.type === "reply") {
      this.history?.setRetrieval(message.retrieval, message.links);
      const wav = message.audio && !this.receivedAudio ? decodeBase64(message.audio) : null;
      this.history?.finish({ text: message.text, wav });
      if (wav) {
        this.playAudio(wav, message.id, epoch);
      } else {
        this.awaitingReply = false;
        if (this.speaking && !this.streamPlayers.size) this.playbackEnded(this.playToken, epoch);
        else if (!this.speaking && !this.recording) this.listen();
      }
    } else if (message.type === "error") {
      this.history?.finish({ status: message.message || "回复失败" });
      this.awaitingReply = false;
      this.stopPlayback();
      if (!this.recording) this.listen(message.message || "这次没有听清，请再说一次。");
    }
  }

  registerPreview(player) {
    player.addEventListener("play", () => {
      if (this.speaking || this.recording || this.farewellContext) { player.pause(); return; }
      if (this.activePreview !== player) this.activePreview?.pause();
      this.activePreview = player;
    });
    const resumeAfterEcho = () => {
      if (this.activePreview === player) this.previewPauseUntil = performance.now() + 180;
    };
    player.addEventListener("pause", resumeAfterEcho);
    player.addEventListener("ended", resumeAfterEcho);
  }

  get previewPlaying() { return this.activePreview?.paused === false; }

  showInputClip(wav, trigger) {
    const clip = new Blob([wav], { type: "audio/wav" });
    const meta = `${((wav.byteLength - 44) / (2 * INPUT_SAMPLE_RATE)).toFixed(1)} 秒 · ${trigger}`;
    this.history?.setInput(this.turn, clip, meta);
  }

  send(message) { if (this.socket?.readyState === WebSocket.OPEN) this.socket.send(JSON.stringify(message)); }

  pcmBuffer(bytes) {
    if (!bytes.length || bytes.length % 2) return null;
    const samples = new DataView(bytes.buffer);
    const buffer = this.context.createBuffer(1, bytes.length / 2, this.outputSampleRate || 22050);
    const channel = buffer.getChannelData(0);
    for (let i = 0; i < channel.length; i++) channel[i] = samples.getInt16(i * 2, true) / 32768;
    return buffer;
  }

  queueAudio(encoded, epoch) {
    const pcm = decodeBase64(encoded);
    const buffer = this.pcmBuffer(pcm);
    if (!buffer) return;
    // Bank a little playback time at the start to absorb late arrivals.
    this.queueBuffer(buffer, epoch, 0, this.receivedAudio ? undefined : STREAM_PLAYBACK_LEAD_SECONDS);
    this.history?.addAudioChunk(pcm);
    if (!this.receivedAudio) {
      this.receivedAudio = true;
      if (!this.pausedAudio) this.phase("speaking", `正在播放回复，${INTERRUPT_HINT}`);
    }
  }

  queueBuffer(buffer, epoch, offset = 0, lead = 0.02) {
    if (this.pausedAudio) {
      this.pausedAudio.push({ buffer, offset });
      return;
    }
    this.activePreview?.pause();
    const context = this.context;
    const token = this.playToken;
    const player = context.createBufferSource();
    player.buffer = buffer;
    player.connect(context.destination);
    player.onended = () => {
      if (!this.streamPlayers.delete(player)) return;
      player.disconnect();
      if (!this.streamPlayers.size && !this.awaitingReply) this.playbackEnded(token, epoch);
    };
    const now = context.currentTime;
    const start = this.nextPlayAt > now ? this.nextPlayAt : now + lead;
    this.streamPlayers.set(player, { start, offset });
    player.start(start, offset);
    this.nextPlayAt = start + (buffer.duration - offset);
    this.speaking = true;
  }

  pauseReply() {
    this.activePreview?.pause();
    const now = this.context.currentTime;
    this.pausedAudio = [];
    // Buffer sources are single-use. Save their unplayed audio before stopping them.
    for (const [player, { start, offset }] of this.streamPlayers) {
      const resumeOffset = offset + Math.max(0, now - start);
      if (resumeOffset < player.buffer.duration) {
        this.pausedAudio.push({ buffer: player.buffer, offset: resumeOffset });
      }
    }
    this.clearPlaybackSources();
  }

  resumeReply(message) {
    const pausedAudio = this.pausedAudio;
    this.pausedAudio = null;
    for (const { buffer, offset } of pausedAudio ?? []) this.queueBuffer(buffer, this.epoch, offset);
    if (this.speaking && (this.receivedAudio || !this.awaitingReply)) {
      this.phase("speaking", `正在继续播放回复，${INTERRUPT_HINT}`);
    } else if (this.awaitingReply) {
      this.phase("thinking", `正在为您整理回复，${INTERRUPT_HINT}`);
    } else {
      this.listen(message);
    }
  }

  async playAudio(wav, turn, epoch) {
    this.activePreview?.pause();
    const token = this.playToken;
    if (!this.pausedAudio) this.speaking = true;
    try {
      let buffer = await this.context.decodeAudioData(wav.buffer);
      if (!this.active || this.epoch !== epoch || this.turn !== turn || this.playToken !== token) return;
      await this.context.resume();
      if (!this.active || this.epoch !== epoch || this.turn !== turn || this.playToken !== token) return;
      if (turn === 0) {
        // Opening the microphone can reconfigure the audio route. Render a short
        // silent lead-in on that route before sending it the first syllable.
        const lead = Math.ceil(buffer.sampleRate * GREETING_PREROLL_SECONDS);
        const padded = this.context.createBuffer(buffer.numberOfChannels, lead + buffer.length, buffer.sampleRate);
        for (let channel = 0; channel < buffer.numberOfChannels; channel++) {
          padded.copyToChannel(buffer.getChannelData(channel), channel, lead);
        }
        buffer = padded;
      }
      this.queueBuffer(buffer, epoch);
      this.awaitingReply = false;
      if (!this.pausedAudio) this.phase("speaking", `正在播放回复，${INTERRUPT_HINT}`);
    } catch (_) {
      if (this.active && this.epoch === epoch && this.turn === turn && this.playToken === token) {
        this.awaitingReply = false;
        if (!this.streamPlayers.size && !this.recording) {
          this.playbackEnded(token, epoch);
          this.listen("语音暂时无法播放，可以继续通话。");
        }
      }
    }
  }

  playbackEnded(token, epoch) {
    if (!this.active || this.epoch !== epoch || this.playToken !== token) return;
    this.speaking = false; this.nextPlayAt = 0;
    if (!this.recording) this.listen();
  }

  stopPlayback() {
    this.playToken++;
    this.pausedAudio = null;
    this.clearPlaybackSources();
  }

  clearPlaybackSources() {
    this.speaking = false;
    this.nextPlayAt = 0;
    for (const player of this.streamPlayers.keys()) {
      player.onended = null;
      player.stop();
      player.disconnect();
    }
    this.streamPlayers.clear();
  }

  updateTimer() {
    const seconds = Math.floor((Date.now() - this.startedAt) / 1000);
    $("timer").textContent = `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
  }

  fail(message) {
    this.stop(); this.phase("error", message);
    $("button-label").textContent = "重新开始";
  }

  async decodeFarewell(url, context) {
    try {
      const response = await fetch(url);
      if (!response.ok) return null;
      return await context.decodeAudioData(await response.arrayBuffer());
    } catch (_) {
      return null;
    }
  }

  async playFarewell(context, audio) {
    this.farewellContext = context;
    try {
      if (context.state !== "running") await context.resume();
      const buffer = await audio;
      if (this.farewellContext !== context) return;
      if (buffer) {
        const player = context.createBufferSource();
        player.buffer = buffer;
        player.connect(context.destination);
        player.onended = () => {
          player.disconnect();
          if (this.farewellContext === context) this.stopFarewell();
        };
        player.start();
        return;
      }
    } catch (_) { /* Release the context even if playback fails. */ }
    if (this.farewellContext === context) this.stopFarewell();
  }

  stopFarewell() {
    const context = this.farewellContext;
    this.farewellContext = null;
    context?.close().catch(() => {});
  }

  stop({ playFarewell = false } = {}) {
    this.active = false; this.epoch++;
    this.awaitingReply = false;
    this.recording = null;
    this.queryBuffer = null;
    this.stopFarewell();
    this.activePreview?.pause();
    this.history?.finish({ status: "通话已结束" });
    this.stopPlayback(); this.send({ type: "hangup" });
    if (this.socket) { this.socket.onclose = null; this.socket.close(); this.socket = null; }
    if (this.capture) { this.capture.port.onmessage = null; this.capture.disconnect(); this.capture = null; }
    if (this.source) { this.source.disconnect(); this.source = null; }
    if (this.silent) { this.silent.disconnect(); this.silent = null; }
    if (this.stream) { this.stream.getTracks().forEach(track => track.stop()); this.stream = null; }
    const context = this.context;
    this.context = null;
    const farewellAudio = this.farewellAudio;
    this.farewellAudio = null;
    if (context) {
      if (playFarewell && farewellAudio) this.playFarewell(context, farewellAudio);
      else context.close().catch(() => {});
    }
    this.micEncoder = null;
    this.micFrames = [];
    clearInterval(this.timer);
    document.documentElement.style.setProperty("--level", "0");
    $("call-button").disabled = false; $("button-label").textContent = "开始通话";
    this.phase("idle", "通话已结束，您可以再次开始。");
  }
}

if (typeof document !== "undefined") new VoiceCall();
