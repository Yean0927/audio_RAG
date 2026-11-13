import { INPUT_SAMPLE_RATE, Pcm16Encoder, VoiceActivityDetector, joinFrames, pcm16Wav, rms } from "./audio-engine.mjs?v=vad-onset";

const TURN_SILENCE_TIMEOUT_MS = 1200;
const FIRST_MODEL_PROBE_MS = 750;
const FIRST_PROBE_AFTER_START_MS = 290;
const MODEL_PROBE_INTERVAL_MS = 500;
const DUPLEX_TARGET_LEVEL = 0.06;
const semanticCommitLabels = {
  semantic_complete: "Smart Turn 确认说完",
  silence_timeout: "静音超时提交",
  recording_limit: "达到录音时限",
};
const $ = id => document.getElementById(id);
const decodeBase64 = encoded => Uint8Array.from(atob(encoded), char => char.charCodeAt(0));
const titles = {
  idle: ["随时为您解答", "点击开始，像打电话一样交流"],
  connecting: ["正在接通", "正在为您建立语音连接"],
  listening: ["可以开始说话", "点击按钮开始录音"],
  recording: ["正在聆听", "您可以自然地表达您的需求"],
  thinking: ["正在为您解答", "稍等片刻，我正在整理回复"],
  speaking: ["客服正在回复", "需要插话时，点击「开始说话」"],
  error: ["暂时无法接通", "请稍后重试"],
};

export class VoiceCall {
  constructor(duplex = false) {
    this.duplex = duplex;
    this.requestSemanticVAD = duplex && new URLSearchParams(location.search).get("vad") === "semantic";
    this.active = false; this.epoch = 0; this.turn = 0; this.playToken = 0;
    this.health = null; this.speaking = false; this.pauseUntil = 0;
    this.awaitingReply = false;
    this.streamPlayers = new Map();
    $("call-button").addEventListener("click", () => this.toggle());
    if (!duplex) {
      $("start-speaking").addEventListener("click", () => this.startSpeaking());
      $("send-now").addEventListener("click", () => this.finishSpeaking());
    }
    $("decode-input").addEventListener("click", () => {
      if (!this.active || this.socket?.readyState !== WebSocket.OPEN) return;
      $("decode-input").disabled = true;
      $("decode-status").textContent = "正在解码…";
      this.send({ type: "inspect_input", id: this.turn });
    });
    this.farewell = $("farewell");
    this.previews = [$("input-recording"), $("decoded-input")];
    for (const player of this.previews) {
      player.addEventListener("play", () => {
        if (this.speaking || this.recording || !this.farewell.paused) { player.pause(); return; }
        for (const other of this.previews) if (other !== player) other.pause();
        this.vad?.reset();
      });
      player.addEventListener("pause", () => { this.previewPauseUntil = performance.now() + 180; });
      player.addEventListener("ended", () => { this.previewPauseUntil = performance.now() + 180; });
    }
    window.addEventListener("pagehide", () => this.stop());
    this.phase("idle");
    this.refreshHealth();
    this.poll = setInterval(() => { if (!this.active) this.refreshHealth(); }, 2500);
  }

  phase(name, caption = "") {
    document.body.dataset.phase = name;
    $("start-speaking").hidden = this.duplex || !["listening", "thinking", "speaking"].includes(name);
    $("send-now").hidden = this.duplex || name !== "recording";
    let [heading, description] = titles[name] || titles.idle;
    if (name === "idle" && this.requestSemanticVAD) [heading, description] = ["语义 VAD 通话", "接通后直接说话，由 Smart Turn 判断何时回答"];
    else if (name === "idle" && this.duplex) [heading, description] = ["全双工语音通话", "接通后直接说话，由模型判断何时回答"];
    else if (name === "listening" && this.duplex) [heading, description] = ["我在听，请讲", "接通后可以直接说话"];
    else if (name === "listening" && this.autoListen) {
      [heading, description] = ["我在听，请讲", "说完停顿片刻，或点击「说完了，发送」"];
    } else if (name === "speaking" && this.duplex) description = "可以直接插话";
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
      this.stop();
      if (this.farewellText) {
        this.showReply(this.farewellText);
        if (this.farewell.hasAttribute("src")) {
          this.farewell.currentTime = 0;
          this.farewell.play().catch(() => {});
        }
      }
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
    this.farewell.pause();
    this.farewellText = "";
    const epoch = ++this.epoch;
    this.active = true; this.turn = 0; this.turnSequence = 0;
    this.awaitingReply = false;
    this.ignoredTurn = null;
    this.pauseUntil = 0;
    this.previewPauseUntil = 0;
    this.clearInput();
    this.showRetrieval();
    this.phase("connecting", "请允许浏览器使用麦克风。");
    $("call-button").disabled = false; $("button-label").textContent = "挂断通话";
    $("reply").hidden = true;
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
      if (this.duplex && !hello.duplex) throw new Error("服务器尚未启用全双工，请更新服务后重试。");
      if (this.requestSemanticVAD && !hello.semantic_vad) throw new Error("服务器尚未启用语义 VAD。");
      this.farewellText = hello.farewell;
      const farewellUrl = hello.farewell_audio_url || null;
      if (this.farewell.getAttribute("src") !== farewellUrl) {
        if (farewellUrl) this.farewell.src = farewellUrl;
        else {
          this.farewell.removeAttribute("src");
          this.farewell.load();
        }
      }
      this.outputSampleRate = hello.output_sample_rate || 22050;
      this.queryBuffer = hello.query_pcm ? this.pcmBuffer(hello.query_pcm) : null;
      this.autoListen = hello.auto_listen === true;
      this.modelEndpoint = this.duplex && hello.model_endpoint === true;
      this.filterInput = this.duplex && hello.speech_filter === true;
      this.semanticVAD = this.requestSemanticVAD && hello.semantic_vad === true;
      this.semanticCommitReason = null;
      const stream = await navigator.mediaDevices.getUserMedia({ audio: {
        channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: !(this.duplex || this.autoListen),
      } });
      if (!this.active || this.epoch !== epoch) { stream.getTracks().forEach(track => track.stop()); return; }
      this.stream = stream;
      await context.audioWorklet.addModule("/assets/capture-worklet.js");
      if (!this.active || this.epoch !== epoch) return;
      this.maxRecordingSamples = Math.floor(context.sampleRate * hello.max_audio_seconds);
      if (this.semanticVAD) {
        this.micEncoder = new Pcm16Encoder(context.sampleRate);
        this.micFrames = [];
        this.micSamples = 0;
      } else {
        this.vad = new VoiceActivityDetector(context.sampleRate, {
          silenceMs: hello.silence_ms,
          minSpeechMs: this.duplex ? Math.min(hello.min_speech_ms, 120) : hello.min_speech_ms,
          maxSeconds: hello.max_audio_seconds, semantic: this.duplex, modelEndpoint: this.modelEndpoint,
        });
      }
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
      this.listen();
      if (hello.greeting) {
        this.showReply(hello.greeting);
        if (hello.greeting_audio) this.playAudio(hello.greeting_audio, 0, epoch);
      }
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
      let mode = "";
      if (this.duplex) mode = this.requestSemanticVAD ? "&duplex=1&vad=semantic" : "&duplex=1&model_endpoint=1";
      const url = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/api/call?stream=1&ack=1${mode}`;
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
    if (!this.active || this.epoch !== epoch) return;
    if (this.semanticVAD) {
      if (this.micEncoder) this.uploadSemanticAudio(frame);
      return;
    }
    if (!this.vad) return;
    const recording = this.recording;
    if (recording?.limited) return;
    if (!recording?.manual) {
      if ((!this.duplex && (!this.autoListen || this.awaitingReply || this.speaking))
          || this.previews?.some(player => !player.paused)) return;
      if (!recording) {
        const now = performance.now();
        if (now < this.previewPauseUntil) return;
        if (now < this.pauseUntil) {
          // Keep the onset through the playback tail without letting echo start a turn.
          this.vad.bufferPreRoll(frame);
          return;
        }
      }
    }
    const volume = Math.min(1, rms(frame) * 15);
    document.documentElement.style.setProperty("--level", volume.toFixed(3));
    if (recording?.manual) {
      const remaining = this.maxRecordingSamples - recording.samples;
      this.captureFrames([frame.subarray(0, remaining)]);
      if (this.recording && recording.samples >= this.maxRecordingSamples) this.finishSpeaking("达到录音时限，自动发送");
      return;
    }
    if (this.duplex) {
      const playing = this.streamPlayers.size > 0;
      const greeting = this.turn === 0 && this.speaking;
      for (const event of this.vad.feed(frame, playing, greeting)) this.handleDuplexAudioEvent(event);
      if (this.modelEndpoint) this.maybeProbeModel();
      else this.acceptTurnDecision();
    } else {
      for (const event of this.vad.feed(frame)) this.handleAudioEvent(event, "自动发送");
    }
  }

  uploadSemanticAudio(frame) {
    if (this.previews?.some(player => !player.paused) || performance.now() < this.previewPauseUntil) {
      this.micFrames = [];
      this.micSamples = 0;
      return;
    }
    const volume = Math.min(1, rms(frame) * 15);
    document.documentElement.style.setProperty("--level", volume.toFixed(3));
    this.micFrames.push(frame);
    this.micSamples += frame.length;
    if (this.micSamples < this.context.sampleRate * 0.08 || !this.canUploadAudio()) return;
    const pcm = this.micEncoder.encode(joinFrames(this.micFrames));
    this.micFrames = [];
    this.micSamples = 0;
    if (pcm.byteLength) this.socket.send(pcm);
  }

  listen(message) {
    this.phase("listening", message || (this.duplex ? "您可以直接说话。" : this.autoListen
      ? "您可以直接说话，也可以点击「开始说话」。" : "点击「开始说话」，说完后点击发送。"));
  }

  interruptReply() {
    if (this.awaitingReply || this.speaking) {
      this.ignoredTurn = this.turn;
      this.send({ type: "interrupt" });
    }
    this.awaitingReply = false;
    this.stopPlayback();
    for (const preview of this.previews ?? []) preview.pause();
  }

  startSpeaking() {
    if (!this.active || !this.vad || this.recording) return;
    this.interruptReply();
    this.vad.reset();
    if (this.beginRecording(true)) this.phase("recording", "正在录音，说完请点击「说完了，发送」。");
  }

  finishSpeaking(trigger = "手动发送") {
    if (!this.active || !this.recording) return;
    if (this.recording.manual) {
      this.finishRecording(this.recording.samples >= this.context.sampleRate * 0.12, trigger);
    } else {
      this.handleAudioEvent(this.vad.finish(), trigger);
    }
  }

  handleAudioEvent(event, trigger) {
    if (!event || !this.active) return;
    if (event.type === "start") {
      if (!this.beginRecording(false) || !this.captureFrames(event.frames)) return;
      this.phase("recording", "说完会自动发送，也可以点「说完了，发送」。");
    } else if (event.type === "audio") {
      this.captureFrames([event.frame]);
    } else {
      this.finishRecording(event.type === "utterance", trigger);
    }
  }

  handleDuplexAudioEvent(event) {
    if (!this.active) return;
    if (event.type === "start") {
      this.pauseReply();
      const gain = this.filterInput ? 1 : Math.max(1, Math.min(6, DUPLEX_TARGET_LEVEL / event.peakLevel));
      if (this.beginRecording(false, gain) && this.captureFrames(event.frames)) {
        this.recording.preRollSamples = this.recording.samples;
        this.phase("recording", "我在听，请继续说。");
      }
      return;
    }
    const recording = this.recording;
    if (!recording) return;
    if (event.type === "audio") {
      this.captureFrames([event.frame.subarray(0, this.maxRecordingSamples - recording.samples)]);
    } else if (event.type === "resume") {
      this.send({ type: "utterance_resume", id: recording.id, probe: recording.probe.id });
      recording.probe = null;
      if (this.uploadFrames()) this.phase("recording", "我在听，请继续说。");
    } else if (event.type === "discard") {
      this.finishRecording(false);
    } else {
      if (event.type === "limit") recording.limited = true;
      if (!this.modelEndpoint && this.probeTurn(recording)) {
        this.phase("recording", "正在判断这句话是否说完…");
      }
    }
  }

  probeTurn(recording) {
    if (recording.probe || !this.uploadFrames()) return false;
    const id = (recording.lastProbe || 0) + 1;
    recording.lastProbe = id;
    recording.lastProbeSamples = recording.samples;
    recording.probe = { id };
    this.send({ type: "utterance_probe", id: recording.id, probe: id });
    return true;
  }

  maybeProbeModel() {
    const recording = this.recording;
    if (!recording || recording.probe) return;
    const lastSamples = recording.lastProbeSamples || 0;
    const intervalMs = lastSamples ? MODEL_PROBE_INTERVAL_MS : FIRST_MODEL_PROBE_MS;
    const newSamples = recording.samples - lastSamples;
    const samplesPerMs = this.context.sampleRate / 1000;
    // The retained audio can already span most of the first probe's window.
    const ready = newSamples >= samplesPerMs * intervalMs
      && (lastSamples > 0 || newSamples - recording.preRollSamples >= samplesPerMs * FIRST_PROBE_AFTER_START_MS);
    if (recording.limited && !newSamples) {
      this.finishRecording(false, "这段录音已达时限，请分成短句再说一次。");
    } else if (recording.limited || ready) {
      this.probeTurn(recording);
    }
  }

  acceptTurnDecision() {
    const recording = this.recording;
    const probe = recording?.probe;
    // Keep playback paused while the model checks for caller speech.
    // A voice onset may be in progress while the result arrives.
    if (!probe?.decision || (!recording.limited && this.vad.voicedRunMs)) return;
    const silenceTimeout = Math.max(TURN_SILENCE_TIMEOUT_MS, this.vad.silenceMs + 150);
    const silenceExpired = recording.limited || this.vad.quietMs >= silenceTimeout;
    if (probe.decision === "wait" && !silenceExpired) return;
    if (!recording.limited) this.vad.reset();
    if (probe.decision === "discard") {
      this.finishRecording(false, probe.message);
    } else {
      const afterWait = probe.decision === "wait";
      this.finishRecording(true, afterWait ? "停顿后自动发送" : "模型确认说完", probe.id, afterWait);
    }
  }

  canUploadAudio() {
    if (this.socket?.readyState === WebSocket.OPEN && this.socket.bufferedAmount < 64 * 1024) return true;
    this.fail("语音连接不稳定，请重新开始通话。");
    return false;
  }

  beginRecording(manual, gain = 1) {
    if (!this.canUploadAudio()) return false;
    const id = ++this.turnSequence;
    this.recording = {
      id, manual, samples: 0, pendingSamples: 0, frames: [], chunks: [],
      encoder: new Pcm16Encoder(this.context.sampleRate, gain),
    };
    this.send({ type: "utterance_start", id });
    return true;
  }

  captureFrames(frames) {
    const recording = this.recording;
    for (const frame of frames) {
      recording.frames.push(frame);
      recording.samples += frame.length;
      recording.pendingSamples += frame.length;
    }
    if (recording.probe && !this.modelEndpoint) {
      // The server already has the pause. Keep a short local tail so a resumed
      // sentence keeps its onset without uploading the entire waiting period.
      const tail = this.context.sampleRate * 0.2;
      while (recording.pendingSamples > tail && recording.frames.length > 1) {
        recording.pendingSamples -= recording.frames.shift().length;
      }
      return true;
    }
    return recording.pendingSamples < this.context.sampleRate * 0.08 || this.uploadFrames();
  }

  uploadFrames() {
    if (!this.canUploadAudio()) return false;
    const recording = this.recording;
    if (recording.pendingSamples) {
      const pcm = recording.encoder.encode(joinFrames(recording.frames));
      recording.frames = [];
      recording.pendingSamples = 0;
      if (pcm.byteLength) {
        this.socket.send(pcm);
        recording.chunks.push(pcm);
      }
    }
    return true;
  }

  finishRecording(commit, trigger, probe = null, afterWait = false) {
    const recording = this.recording;
    if (!recording) return;
    document.documentElement.style.setProperty("--level", "0");
    if (!commit) {
      this.recording = null;
      this.send({ type: "utterance_cancel", id: recording.id });
      if (this.duplex) this.resumeReply(trigger);
      else this.listen(recording.manual ? "这段录音太短，请重新开始说话。" : undefined);
      return;
    }
    const flushAudio = probe === null || this.modelEndpoint;
    if (flushAudio ? !this.uploadFrames() : !this.canUploadAudio()) return;
    this.markSubmitted(recording.id);
    const message = { type: "utterance_end", id: recording.id };
    if (probe !== null) message.probe = probe;
    if (afterWait) message.after_wait = true;
    this.send(message);
    if (this.filterInput) this.clearInput();
    else this.showInputClip(pcm16Wav(recording.chunks), trigger);
    this.phase("thinking", "已听到您的问题，请稍候。");
  }

  markSubmitted(turn) {
    this.recording = null;
    if (this.duplex) this.stopPlayback();
    this.turn = turn;
    this.awaitingReply = true;
    this.receivedAudio = false;
    if (this.duplex) this.showReply("");
    this.showRetrieval();
  }

  onMessage(message, epoch) {
    if (!this.active || this.epoch !== epoch) return;
    if (this.semanticVAD && message.type === "speech_started") {
      this.pauseReply();
      this.recording = { id: message.id };
      this.phase("recording", "我在听，请继续说。");
      return;
    }
    if (this.semanticVAD && message.type === "speech_stopped") {
      if (this.recording?.id !== message.id) return;
      this.markSubmitted(message.id);
      this.semanticCommitReason = message.reason;
      this.clearInput();
      this.phase("thinking", "已听到您的问题，请稍候。");
      return;
    }
    if (message.type === "turn_decision") {
      const recording = this.recording;
      if (!this.duplex || !recording?.probe || message.id !== recording.id || message.probe !== recording.probe.id) return;
      if (this.modelEndpoint) {
        if (message.decision === "wait") {
          this.send({ type: "utterance_resume", id: recording.id, probe: recording.probe.id });
          recording.probe = null;
          this.maybeProbeModel();
        } else {
          if (!recording.limited) this.vad.reset();
          const reply = message.decision === "reply";
          if (!reply && message.input_pcm) {
            this.showInputClip(pcm16Wav([decodeBase64(message.input_pcm)]), "模型未通过的判定片段", recording.id);
          } else if (!reply && Number.isInteger(message.audio_bytes) && message.audio_bytes > 0 && message.audio_bytes % 2 === 0) {
            const label = message.speech_detected === false ? "未检测到人声的录音" : "模型未通过的判定片段";
            this.showInputClip(pcm16Wav(recording.chunks, message.audio_bytes), label, recording.id);
          }
          this.finishRecording(reply, reply ? "模型确认说完" : message.message, recording.probe.id);
        }
        return;
      }
      recording.probe.decision = message.decision;
      recording.probe.message = message.message;
      if (message.decision === "wait" && !this.speaking) this.phase("recording", "您可以继续说。");
      this.acceptTurnDecision();
      return;
    }
    if (message.id != null && message.id !== this.turn) return;
    if (message.id != null && message.id === this.ignoredTurn && message.type !== "input_audio") return;
    if (message.type === "utterance_received") {
      if (message.input_pcm) {
        const label = this.semanticVAD ? semanticCommitLabels[this.semanticCommitReason] : null;
        this.showInputClip(pcm16Wav([decodeBase64(message.input_pcm)]), label || "送入模型的录音");
      }
      if (this.awaitingReply && this.queryBuffer && !this.speaking) {
        this.queueBuffer(this.queryBuffer, epoch);
        if (!this.recording) this.phase("thinking", `正在为您查询，${this.interruptHint}`);
      }
    } else if (message.type === "thinking") {
      if (!this.speaking && !this.recording) this.phase("thinking", `正在为您整理回复，${this.interruptHint}`);
    } else if (message.type === "audio_chunk" && this.awaitingReply) {
      this.queueAudio(message.pcm, epoch);
    } else if (message.type === "retrieval") {
      this.showRetrieval(message.retrieval);
    } else if (message.type === "reply") {
      this.showInputTokens(message.input_tokens);
      this.showReply(message.text || "");
      if (this.retrievalTurn !== this.turn) this.showRetrieval(message.retrieval);
      if (message.audio && !this.receivedAudio) {
        this.playAudio(message.audio, message.id, epoch);
      } else {
        this.awaitingReply = false;
        if (this.speaking && !this.streamPlayers.size) this.playbackEnded(this.playToken, epoch);
        else if (!this.speaking && !this.recording) this.listen();
      }
    } else if (message.type === "error") {
      this.awaitingReply = false;
      this.stopPlayback();
      if (!this.recording) this.listen(message.message || "这次没有听清，请再说一次。");
    } else if (message.type === "input_audio") {
      if (message.id !== this.inputClipTurn) return;
      $("decode-input").disabled = false;
      $("decode-status").textContent = message.message || "已解码，可以播放。";
      if (message.audio) this.setPreview($("decoded-input"), decodeBase64(message.audio));
    }
  }

  setPreview(player, wav) {
    const previous = player.getAttribute("src");
    player.pause();
    if (previous) URL.revokeObjectURL(previous);
    if (wav) player.src = URL.createObjectURL(new Blob([wav], { type: "audio/wav" }));
    else player.removeAttribute("src");
    player.hidden = !wav;
    player.load();
  }

  clearInput() {
    this.inputClipTurn = null;
    for (const player of this.previews) this.setPreview(player, null);
    $("input-check").hidden = true;
    $("input-token-details").hidden = true;
    $("input-tokens").textContent = "";
    $("decode-status").textContent = "";
    $("decode-input").disabled = true;
  }

  showInputClip(wav, trigger, turn = this.turn) {
    this.clearInput();
    this.inputClipTurn = turn;
    this.setPreview($("input-recording"), wav);
    $("input-meta").textContent = `第 ${turn} 轮 · ${((wav.byteLength - 44) / (2 * INPUT_SAMPLE_RATE)).toFixed(1)} 秒 · ${trigger}`;
    $("input-check").hidden = false;
  }

  showInputTokens(tokens) {
    if (this.inputClipTurn !== this.turn || !Array.isArray(tokens) || !tokens.length) return;
    $("input-token-summary").textContent = `语音 token（${tokens.length} 个）`;
    $("input-tokens").textContent = tokens.join(" ");
    $("input-token-details").hidden = false;
    $("decode-input").disabled = false;
  }

  showReply(text) { $("reply").textContent = text; $("reply").hidden = !text; }

  showRetrieval(retrieval) {
    const panel = $("retrieval-check");
    const results = $("retrieval-results");
    const selected = retrieval?.selected_metric;
    const matches = retrieval?.[selected];
    results.replaceChildren();
    panel.hidden = !matches?.length;
    this.retrievalTurn = panel.hidden ? null : this.turn;
    if (panel.hidden) return;

    $("retrieval-summary").textContent = `第 ${this.turn} 轮检索结果 · Top-${matches.length}`;
    const labels = { euclidean: "平方欧氏距离（越小越靠前）", cosine: "余弦相似度（越大越靠前）" };
    const other = selected === "euclidean" ? "cosine" : "euclidean";
    for (const metric of [selected, other]) {
      if (!retrieval[metric]?.length) continue;
      const heading = document.createElement("h3");
      heading.textContent = `${metric === selected ? "用于生成回答" : "对照排序"} · ${labels[metric]}`;
      const list = document.createElement("ol");
      for (const hit of retrieval[metric]) {
        const item = document.createElement("li");
        const score = document.createElement("strong");
        score.textContent = Number.isFinite(hit.score) ? `分数 ${hit.score.toPrecision(6)}` : "分数不可用";
        const passage = document.createElement("div");
        passage.textContent = hit.passage;
        item.append(score, passage);
        list.append(item);
      }
      results.append(heading, list);
    }
  }

  send(message) { if (this.socket?.readyState === WebSocket.OPEN) this.socket.send(JSON.stringify(message)); }

  get interruptHint() { return this.duplex ? "可以直接插话。" : "点击「开始说话」可打断。"; }

  pcmBuffer(encoded) {
    const bytes = decodeBase64(encoded);
    if (!bytes.length || bytes.length % 2) return null;
    const samples = new DataView(bytes.buffer);
    const buffer = this.context.createBuffer(1, bytes.length / 2, this.outputSampleRate || 22050);
    const channel = buffer.getChannelData(0);
    for (let i = 0; i < channel.length; i++) channel[i] = samples.getInt16(i * 2, true) / 32768;
    return buffer;
  }

  queueAudio(encoded, epoch) {
    const buffer = this.pcmBuffer(encoded);
    if (!buffer) return;
    this.queueBuffer(buffer, epoch);
    if (!this.receivedAudio) {
      this.receivedAudio = true;
      if (!this.pausedAudio) this.phase("speaking", `正在播放回复，${this.interruptHint}`);
    }
  }

  queueBuffer(buffer, epoch, offset = 0) {
    if (this.pausedAudio) {
      this.pausedAudio.push({ buffer, offset });
      return;
    }
    for (const preview of this.previews ?? []) preview.pause();
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
    const start = this.nextPlayAt > now ? this.nextPlayAt : now + 0.02;
    this.streamPlayers.set(player, { start, offset });
    player.start(start, offset);
    this.nextPlayAt = start + (buffer.duration - offset);
    this.speaking = true;
  }

  pauseReply() {
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
      this.phase("speaking", `正在继续播放回复，${this.interruptHint}`);
    } else if (this.awaitingReply) {
      this.phase("thinking", `正在为您整理回复，${this.interruptHint}`);
    } else {
      this.listen(message);
    }
  }

  async playAudio(encoded, turn, epoch) {
    for (const preview of this.previews ?? []) preview.pause();
    const token = this.playToken;
    if (!this.pausedAudio) this.speaking = true;
    try {
      const bytes = decodeBase64(encoded);
      const buffer = await this.context.decodeAudioData(bytes.buffer);
      if (!this.active || this.epoch !== epoch || this.turn !== turn || this.playToken !== token) return;
      await this.context.resume();
      if (!this.active || this.epoch !== epoch || this.turn !== turn || this.playToken !== token) return;
      this.queueBuffer(buffer, epoch);
      this.awaitingReply = false;
      if (!this.pausedAudio) this.phase("speaking", `正在播放回复，${this.interruptHint}`);
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
    const keepOnset = this.duplex && this.turn === 0 && this.vad?.hasGreetingOnset;
    this.pauseUntil = keepOnset ? 0 : performance.now() + 180;
    if (!this.recording) {
      // Clear playback candidates while retaining the microphone history for barge-in.
      if (!keepOnset) this.vad?.reset({ keepPreRoll: this.duplex });
      this.listen();
    }
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

  stop() {
    this.active = false; this.epoch++;
    this.awaitingReply = false;
    this.recording = null;
    this.queryBuffer = null;
    this.farewell.pause();
    for (const preview of this.previews ?? []) preview.pause();
    $("decode-input").disabled = true;
    this.stopPlayback(); this.send({ type: "hangup" });
    if (this.socket) { this.socket.onclose = null; this.socket.close(); this.socket = null; }
    if (this.capture) { this.capture.port.onmessage = null; this.capture.disconnect(); this.capture = null; }
    if (this.source) { this.source.disconnect(); this.source = null; }
    if (this.silent) { this.silent.disconnect(); this.silent = null; }
    if (this.stream) { this.stream.getTracks().forEach(track => track.stop()); this.stream = null; }
    if (this.context) { this.context.close().catch(() => {}); this.context = null; }
    this.vad = null;
    this.micEncoder = null;
    this.micFrames = [];
    clearInterval(this.timer);
    document.documentElement.style.setProperty("--level", "0");
    $("call-button").disabled = false; $("button-label").textContent = "开始通话";
    this.phase("idle", "通话已结束，您可以再次开始。");
  }
}

if (typeof document !== "undefined") new VoiceCall(location.pathname === "/duplex");
