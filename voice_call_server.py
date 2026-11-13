"""One-button voice calls with selectable GLM, Kimi-Audio and Qwen-Omni backends."""
from __future__ import annotations

import asyncio
import base64
from collections import deque
from contextlib import asynccontextmanager, nullcontext
from copy import copy
from dataclasses import dataclass, field
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import re
import threading
import time
import uuid
import wave
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from voice_backend import BACKENDS, DEFAULT_OUTPUT_SAMPLE_RATE, VoiceBackendBase, check_cancelled
from voice_input import CallSpeechFilter
from voice_semantic_vad import SemanticVAD, SmartTurn
from voice_web_demo import ROOT, STREAM_BLOCK_SIZES, InferenceCancelled, create_backend, create_parser, validate_args

LOG = logging.getLogger("voice-call")
ASSETS = ROOT / "call_web"
INPUT_TOKEN = re.compile(r"<\|audio_(\d+)\|>")
INPUT_SAMPLE_RATE = 16000
INPUT_SAMPLE_BYTES = 2
FAREWELL_TEXT = (
    "亲亲，感谢您对雷士的支持，如果您后续有其他任何需要咨询的问题欢迎随时进线哦。"
    "祝您生活愉快！再见。"
)


@dataclass
class TurnProbe:
    sequence: int
    cancel: threading.Event = field(default_factory=threading.Event)
    decision: str | None = None
    input_ids: object | None = None
    pcm_size: int = 0
    model_pcm: bytes = b""


class CallLatencyProfile:
    def __init__(self, session, block_sizes):
        self.started = time.perf_counter()
        self.session = session
        self.block_sizes = block_sizes
        self.events = []
        self.lock = threading.Lock()

    def mark(self, stage, *, at=None, **details):
        elapsed = (time.perf_counter() if at is None else at) - self.started
        with self.lock:
            self.events.append({"stage": stage, "ms": round(elapsed * 1000, 3), **details})

    def snapshot(self):
        with self.lock:
            return {"session": self.session, "block_sizes": self.block_sizes,
                    "events": sorted(self.events, key=lambda item: item["ms"])}


def load_config(path):
    path = Path(path)
    if not path.exists():
        path.write_text((ROOT / "call_config.example.json").read_text(encoding="utf-8"), encoding="utf-8")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("language", "zh") not in ("zh", "en"):
        raise ValueError("call_config.json language must be zh or en")
    greeting = config.get("greeting")
    if greeting is not None and not isinstance(greeting, str):
        raise ValueError("call_config.json greeting must be text")
    config["greeting"] = (greeting or "").strip()
    knowledge = config.get("knowledge_file", "")
    if knowledge:
        file = Path(knowledge)
        if not file.is_absolute():
            file = ROOT / file
        if not file.is_file() or file.stat().st_size > 1_000_000:
            raise ValueError("knowledge_file must be an existing UTF-8 file under 1 MB")
        config["knowledge_text"] = file.read_text(encoding="utf-8-sig").strip()
        if len(config["knowledge_text"]) > 200_000:
            raise ValueError("Knowledge text must be under 200,000 characters")
    else:
        config["knowledge_text"] = ""
    config["silence_ms"] = max(500, min(1800, int(config.get("silence_ms", 850))))
    config["min_speech_ms"] = max(200, min(800, int(config.get("min_speech_ms", 260))))
    if not isinstance(config.get("auto_listen", False), bool):
        raise ValueError("call_config.json auto_listen must be true or false")
    if not isinstance(config.get("query_prompt", False), bool):
        raise ValueError("call_config.json query_prompt must be true or false")
    return config


def missing_decoder(args):
    folder = Path(args.flow_path)
    return [name for name in ("config.yaml", "flow.pt", "hift.pt")
            if not (folder / name).is_file() or (folder / name).stat().st_size < 64]


def prepare_decoder(args):
    if getattr(args, "backend", "glm") != "glm":
        return
    if missing_decoder(args) and args.download_models:
        # This flag explicitly enables downloads for this project process only.
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            os.environ[key] = "0"
        from huggingface_hub import snapshot_download
        LOG.info("Downloading missing decoder files into %s", args.flow_path)
        snapshot_download(
            "zai-org/glm-4-voice-decoder", local_dir=args.flow_path,
            allow_patterns=["config.yaml", "flow.pt", "hift.pt"], max_workers=2,
        )
    missing = missing_decoder(args)
    if missing:
        raise RuntimeError(
            f"Decoder 文件缺失：{', '.join(missing)}。运行 bash start_voice_call.sh 自动下载，"
            "或先 unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE，"
            f"再下载到 {Path(args.flow_path).resolve()}。"
        )
    for name in ("flow.pt", "hift.pt"):
        with (Path(args.flow_path) / name).open("rb") as stream:
            if stream.read(80).startswith(b"version https://git-lfs"):
                raise RuntimeError(f"{name} 是 Git LFS 指针，请下载真实权重。")


def validate_wav(data, max_seconds, *, sample_rate=INPUT_SAMPLE_RATE):
    if len(data) > (max_seconds + 1) * sample_rate * 2 + 4096:
        raise ValueError("这段语音过长，请分成短句。")
    try:
        with wave.open(io.BytesIO(data), "rb") as stream:
            if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate()) != (1, 2, sample_rate):
                raise ValueError("音频格式无效，请重新开始通话。")
            frames = stream.getnframes()
            if not 0 < frames <= (max_seconds + 0.5) * sample_rate:
                raise ValueError("语音为空或过长，请重试。")
            if len(stream.readframes(frames)) != frames * 2:
                raise ValueError("语音传输不完整，请重新说一次。")
    except (wave.Error, EOFError) as error:
        raise ValueError("语音格式无效，请重新说一次。") from error


def cached_call_clip(args, backend, text, language):
    identity = ["voice-greeting-v2", text, language, args.model_path,
                str(Path(args.flow_path).resolve()), args.quantization]
    if args.flow_steps != 10:
        identity.append(args.flow_steps)
    sample_rate = getattr(backend, "output_sample_rate", DEFAULT_OUTPUT_SAMPLE_RATE)
    identity.extend([getattr(args, "backend", "glm"), sample_rate, args.tokenizer_path,
                     getattr(args, "omni_speaker", "Chelsie")])
    key = hashlib.sha256(json.dumps(identity).encode("utf-8")).hexdigest()[:24]
    path = Path(args.output_dir).resolve() / "greetings" / f"{key}.wav"
    try:
        audio = path.read_bytes()
        validate_wav(audio, 45, sample_rate=sample_rate)
        return audio, path
    except FileNotFoundError:
        pass
    except ValueError:
        LOG.warning("Cached call audio is invalid; regenerating %s", path)

    audio = backend.synthesize_greeting(text, language)
    validate_wav(audio, 45, sample_rate=sample_rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(audio)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    LOG.info("Saved call audio to %s", path)
    return audio, path


def build_backends(args):
    if not args.replica_gpus:
        return [create_backend(args)]
    backends = []
    for gpu in args.replica_gpus:
        worker_args = copy(args)
        worker_args.llm_device = worker_args.speech_device = f"cuda:{gpu}"
        backends.append(create_backend(worker_args))
    return backends


def create_app(args, config, backend=None, *, speech_filter=None):
    args.system_prompt = config.get("system_prompt", "")
    greeting = config.get("greeting", "")
    language = config.get("language", "zh")
    knowledge = config.get("knowledge_text", "").strip()
    backends = [backend] if backend is not None else build_backends(args)
    slots = deque(backends)
    phase = {"state": "loading"}
    initialize_task = None
    greeting_audio = None
    farewell_path = None
    query_pcm = None
    smart_turn = None

    def device_scope(worker):
        if args.replica_gpus and isinstance(worker, VoiceBackendBase):
            import torch
            return torch.cuda.device(worker.args.llm_device)
        return nullcontext()

    def prepare_clip(text, clip_language=language):
        if not text:
            return None, None
        worker = backends[0]
        try:
            with worker.lock, device_scope(worker):
                return cached_call_clip(args, worker, text, clip_language)
        except Exception:
            LOG.exception("预录音频准备失败：%s", text)
            return None, None

    def initialize_sync():
        nonlocal speech_filter, smart_turn
        if args.smart_turn_model and smart_turn is None:
            try:
                smart_turn = SmartTurn(args.smart_turn_model)
            except Exception:
                LOG.exception("语义 VAD 模型加载失败：%s", args.smart_turn_model)
        # Injected backends are used by protocol tests, without CUDA or model weights.
        if any(isinstance(worker, VoiceBackendBase) for worker in backends):
            if speech_filter is None:
                speech_filter = CallSpeechFilter(INPUT_SAMPLE_RATE)
            prepare_decoder(args)
        for worker in backends:
            with worker.lock, device_scope(worker):
                worker.load()
                if knowledge:
                    worker.prepare_knowledge(knowledge, language)
                if args.replica_gpus and isinstance(worker, VoiceBackendBase):
                    import torch
                    free, total = torch.cuda.mem_get_info(worker.llm_device)
                    LOG.info("Call replica on %s loaded; device memory free %.2f / %.2f GiB",
                             worker.llm_device, free / 1024**3, total / 1024**3)
        opening, _ = prepare_clip(greeting)
        _, ending_path = prepare_clip(FAREWELL_TEXT, "zh")
        query = None
        if config.get("query_prompt", False):
            query_text = "正在为您查询。" if language == "zh" else "I'm looking into that for you."
            query, _ = prepare_clip(query_text)
        pcm = None
        if query:
            with wave.open(io.BytesIO(query), "rb") as audio:
                pcm = base64.b64encode(audio.readframes(audio.getnframes())).decode("ascii")
        return (base64.b64encode(opening).decode("ascii") if opening else None), ending_path, pcm

    async def initialize():
        nonlocal greeting_audio, farewell_path, query_pcm
        phase["state"] = "loading"
        try:
            greeting_audio, farewell_path, query_pcm = await asyncio.to_thread(initialize_sync)
        except Exception:
            LOG.exception("客服初始化失败；修复后可在页面点击重新连接，或重启服务。")
            phase["state"] = "error"
        else:
            phase["state"] = "ready"
            LOG.info("VOICE CALL READY — %s concurrent call slot(s), http://localhost:%s",
                     len(backends), args.port)

    def start_initialization():
        nonlocal initialize_task
        if initialize_task is None or initialize_task.done():
            initialize_task = asyncio.create_task(initialize())

    @asynccontextmanager
    async def lifespan(app):
        start_initialization()
        yield
        if initialize_task is not None and not initialize_task.done():
            initialize_task.cancel()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.mount("/assets", StaticFiles(directory=ASSETS), name="assets")

    @app.get("/")
    async def home():
        return RedirectResponse("/duplex", headers={"Cache-Control": "no-store"})

    @app.get("/manual")
    @app.get("/duplex")
    async def index():
        return FileResponse(ASSETS / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/api/farewell/{filename}")
    async def farewell(filename: str):
        if farewell_path is None or filename != farewell_path.name or not farewell_path.is_file():
            raise HTTPException(status_code=404)
        return FileResponse(farewell_path, media_type="audio/wav", headers={"Cache-Control": "no-cache"})

    @app.get("/api/health")
    async def health():
        free = len(slots) if phase["state"] == "ready" else 0
        return {"state": phase["state"], "available": free > 0, "available_slots": free,
                "max_calls": len(backends), "title": config.get("title", "语音客服")}

    @app.post("/api/retry")
    async def retry():
        if phase["state"] == "error":
            phase["state"] = "loading"
            start_initialization()
        return {"state": phase["state"]}

    @app.websocket("/api/call")
    async def call(socket: WebSocket):
        origin = socket.headers.get("origin")
        if origin and urlparse(origin).netloc != socket.headers.get("host"):
            await socket.close(code=1008)
            return
        await socket.accept()
        if phase["state"] != "ready" or not slots:
            await socket.send_json({"type": "error", "message": "客服暂未就绪或正在通话，请稍后再试。"})
            await socket.close(code=1013)
            return
        stream_audio = socket.query_params.get("stream") == "1"
        duplex = socket.query_params.get("duplex") == "1"
        use_semantic_vad = duplex and socket.query_params.get("vad") == "semantic"
        if use_semantic_vad and smart_turn is None:
            message = ("语义 VAD 模型加载失败，请检查服务端日志。" if args.smart_turn_model else
                       "语义 VAD 尚未启用，请用 --smart-turn-model 启动服务。")
            await socket.send_json({"type": "error", "message": message})
            await socket.close(code=1008)
            return
        model_endpoint = duplex and not use_semantic_vad and socket.query_params.get("model_endpoint") == "1"
        acknowledge_input = socket.query_params.get("ack") == "1"
        preloaded_query_pcm = query_pcm if acknowledge_input else None
        profile_latency = stream_audio and socket.query_params.get("profile") == "1"
        block_sizes = STREAM_BLOCK_SIZES
        requested_blocks = socket.query_params.get("block_sizes") if profile_latency else None
        if requested_blocks is not None:
            if not re.fullmatch(r"(?:[1-9][0-9]|100)(?:,(?:[1-9][0-9]|100)){2}", requested_blocks):
                await socket.send_json({"type": "error", "message": "分块需要三个 10–100 的整数。"})
                await socket.close(code=1008)
                return
            block_sizes = tuple(map(int, requested_blocks.split(",")))
        backend = slots.popleft()
        inference_lock = asyncio.Lock()
        session = uuid.uuid4().hex
        incoming = Path(args.output_dir).resolve() / "incoming" / session
        history = []
        current_event = None
        current_id = 0
        last_id = 0
        pending_id = None
        pending_pcm = None
        pending_probe = None
        last_probe = 0
        max_pcm_bytes = args.max_audio_seconds * INPUT_SAMPLE_RATE * INPUT_SAMPLE_BYTES
        last_input = None
        inspection = None
        closed = False
        jobs = set()
        semantic_vad = None

        async def send(message):
            if not closed:
                await socket.send_json(message)

        def start_job(coro):
            task = asyncio.create_task(coro)
            jobs.add(task)
            task.add_done_callback(jobs.discard)

        def clear_probe():
            nonlocal pending_probe
            if pending_probe is not None:
                pending_probe.cancel.set()
                pending_probe = None

        def prepare_call_pcm(pcm):
            return speech_filter(pcm) if speech_filter is not None else pcm

        def judge(pcm, event):
            check_cancelled(event)
            model_pcm = prepare_call_pcm(pcm)
            if not model_pcm:
                return "discard", None, model_pcm
            with backend.lock, device_scope(backend):
                check_cancelled(event)
                decision, input_ids = backend.judge_turn(model_pcm, INPUT_SAMPLE_RATE, language, event)
            return decision, input_ids, model_pcm

        async def assess_turn(turn, probe):
            started = time.monotonic()
            retry = False
            try:
                async with inference_lock:
                    if closed or pending_probe is not probe:
                        return
                    pcm = bytes(pending_pcm)
                    probe.pcm_size = len(pcm)
                    decision, input_ids, probe.model_pcm = await asyncio.to_thread(judge, pcm, probe.cancel)
                # The first live probe can contain only the start of the caller's speech.
                retry = model_endpoint and probe.sequence == 1 and decision == "discard"
                message = {"type": "turn_decision", "id": turn, "probe": probe.sequence,
                           "decision": "wait" if retry else decision}
                if decision in ("reply", "wait"):
                    probe.input_ids = input_ids
            except InferenceCancelled:
                return
            except Exception:
                LOG.exception("Call %s turn %s endpoint inference failed", session, turn)
                decision = "discard"
                message = {"type": "turn_decision", "id": turn, "probe": probe.sequence,
                           "decision": decision, "message": "暂时无法判断这段话，请再说一次。"}
            if closed or probe.cancel.is_set():
                return
            probe.decision = message["decision"]
            if model_endpoint and probe.decision == "discard" and probe.pcm_size:
                message["audio_bytes"] = probe.pcm_size
                if speech_filter is not None and "message" not in message:
                    if not probe.model_pcm:
                        message["speech_detected"] = False
                    elif acknowledge_input:
                        message["input_pcm"] = base64.b64encode(probe.model_pcm).decode("ascii")
            LOG.info("Call %s turn %s probe %s decision %s audio_ms=%.0f probe_ms=%.0f retry=%s",
                     session, turn, probe.sequence, decision,
                     probe.pcm_size * 1000 / (INPUT_SAMPLE_RATE * INPUT_SAMPLE_BYTES),
                     (time.monotonic() - started) * 1000, retry)
            try:
                await send(message)
            except (RuntimeError, WebSocketDisconnect):
                probe.cancel.set()

        def infer(path, saved_history, event, on_audio=None, profile=None, input_ids=None, on_retrieval=None):
            options = {}
            if on_audio is not None:
                options["on_audio"] = on_audio
            if on_retrieval is not None:
                options["on_retrieval"] = on_retrieval
            if input_ids is not None:
                options["input_ids"] = input_ids
            if profile:
                options["profile"] = profile.mark
                if requested_blocks is not None:
                    options["stream_block_sizes"] = block_sizes
                profile.mark("worker_started")
            with backend.lock, device_scope(backend):
                if event.is_set():
                    raise InferenceCancelled()
                if profile:
                    profile.mark("backend_lock_acquired")
                return backend.answer(
                    str(path), bool(knowledge), knowledge, None, language, saved_history, cancel_event=event,
                    **options,
                )

        async def stream_reply(turn, path, event, profile, input_ids):
            loop = asyncio.get_running_loop()
            updates = asyncio.Queue()
            started = time.monotonic()

            def publish(kind, payload):
                loop.call_soon_threadsafe(updates.put_nowait, (kind, payload))

            def publish_retrieval(retrieval):
                publish("retrieval", retrieval)

            def run(saved_history):
                try:
                    return infer(
                        path, saved_history, event, on_audio=lambda pcm: publish("audio_chunk", pcm),
                        profile=profile, input_ids=input_ids,
                        on_retrieval=publish_retrieval if duplex and knowledge else None,
                    )
                finally:
                    loop.call_soon_threadsafe(updates.put_nowait, None)

            worker = asyncio.create_task(asyncio.to_thread(run, list(history)))
            streamed = False
            packet = 0
            try:
                while (update := await updates.get()) is not None:
                    if event.is_set() or closed or turn != current_id:
                        break
                    kind, payload = update
                    if kind == "retrieval":
                        await send({"type": "retrieval", "id": turn, "retrieval": payload})
                        continue
                    if profile:
                        packet += 1
                        profile.mark("chunk_dequeued", packet=packet)
                    message = {"type": "audio_chunk", "id": turn, "pcm": base64.b64encode(payload).decode("ascii")}
                    if profile:
                        profile.mark("chunk_send_start", packet=packet)
                    await send(message)
                    if profile:
                        profile.mark("chunk_send_end", packet=packet)
                    if not streamed:
                        streamed = True
                        LOG.info("Call %s turn %s first audio in %.2fs", session, turn, time.monotonic() - started)
            except BaseException:
                event.set()
                raise
            finally:
                result = await worker
            return result, streamed

        async def respond(turn, path, event, profile, input_ids):
            nonlocal history, last_input
            try:
                async with inference_lock:
                    if event.is_set() or closed:
                        return
                    if profile:
                        profile.mark("inference_slot_acquired")
                    await send({"type": "thinking", "id": turn})
                    if stream_audio:
                        result, streamed = await stream_reply(turn, path, event, profile, input_ids)
                    else:
                        result = await asyncio.to_thread(infer, path, list(history), event, input_ids=input_ids)
                        streamed = False
                    if event.is_set() or closed or turn != current_id:
                        return
                    _, audio, _, _, updated = result
                    record = updated[-1]
                    encoded = base64.b64encode(Path(audio).read_bytes()).decode("ascii") if audio and not streamed else None
                    if event.is_set() or closed or turn != current_id:
                        return
                    tokens = record.get("input_tokens")
                    if tokens is None:
                        tokens = [int(value) for value in INPUT_TOKEN.findall(record.get("question_tokens", ""))]
                    last_input = (turn, tokens)
                    reply = {"type": "reply", "id": turn, "text": record["text"],
                             "audio": encoded, "input_tokens": tokens}
                    if "retrieval" in record:
                        reply["retrieval"] = record["retrieval"]
                    if profile:
                        profile.mark("reply_ready")
                        reply["latency_profile"] = profile.snapshot()
                    await send(reply)
                    history = updated
            except InferenceCancelled:
                LOG.info("Call %s turn %s interrupted", session, turn)
            except (WebSocketDisconnect, RuntimeError):
                if closed or event.is_set():
                    return
                LOG.exception("Call inference failed")
                try:
                    await send({"type": "error", "id": turn, "message": "这次回答暂时遇到问题，请重新说一次。"})
                except Exception:
                    pass
            except Exception:
                LOG.exception("Call inference failed")
                if not closed and not event.is_set():
                    await send({"type": "error", "id": turn, "message": "这次回答暂时遇到问题，请重新说一次。"})

        def decode_input(tokens):
            with backend.lock, device_scope(backend):
                return backend.decode_input_wav(tokens)

        async def inspect_input(turn, tokens):
            try:
                async with inference_lock:
                    if closed or turn != current_id:
                        return
                    audio = await asyncio.to_thread(decode_input, tokens)
                if not closed and turn == current_id:
                    await send({"type": "input_audio", "id": turn, "audio": base64.b64encode(audio).decode("ascii")})
            except Exception:
                LOG.exception("Input token playback failed")
                if not closed and turn == current_id:
                    try:
                        await send({"type": "input_audio", "id": turn, "message": "这轮的语音 token 暂时无法解码。"})
                    except (RuntimeError, WebSocketDisconnect):
                        pass

        async def submit_recording(turn, data, *, pcm=False, input_ids=None, filtered=False):
            nonlocal current_id, current_event
            profile = CallLatencyProfile(session, block_sizes) if profile_latency else None
            if pcm:
                if not data:
                    raise ValueError("语音为空，请重新说一次。")
            else:
                validate_wav(data, args.max_audio_seconds)
            if current_event:
                current_event.set()
            current_id = turn
            current_event = threading.Event()
            path = incoming / f"{turn}.wav"
            if pcm:
                with wave.open(str(path), "wb") as output:
                    output.setparams((1, INPUT_SAMPLE_BYTES, INPUT_SAMPLE_RATE, 0, "NONE", "not compressed"))
                    output.writeframes(data)
            else:
                path.write_bytes(data)
            if profile:
                profile.mark("wav_saved")
            if preloaded_query_pcm or (acknowledge_input and filtered):
                message = {"type": "utterance_received", "id": turn}
                if acknowledge_input and filtered:
                    message["input_pcm"] = base64.b64encode(data).decode("ascii")
                await send(message)
            start_job(respond(turn, path, current_event, profile, input_ids))

        try:
            incoming.mkdir(parents=True, exist_ok=True)
            if use_semantic_vad:
                try:
                    semantic_vad = await asyncio.to_thread(SemanticVAD, smart_turn, args.max_audio_seconds)
                except Exception as error:
                    LOG.exception("Call %s failed to initialize streaming Silero", session)
                    raise ValueError("语义 VAD 初始化失败，请检查服务端日志。") from error
            ready = {"type": "ready", "greeting": greeting, "greeting_audio": greeting_audio,
                     "farewell": FAREWELL_TEXT,
                     "farewell_audio_url": f"/api/farewell/{farewell_path.name}" if farewell_path else None,
                     "query_pcm": preloaded_query_pcm,
                     "output_sample_rate": getattr(backend, "output_sample_rate", DEFAULT_OUTPUT_SAMPLE_RATE),
                     "language": language, "silence_ms": config["silence_ms"],
                     "min_speech_ms": config["min_speech_ms"], "max_audio_seconds": args.max_audio_seconds,
                     "auto_listen": config.get("auto_listen", False)}
            if duplex:
                ready["duplex"] = True
                ready["model_endpoint"] = model_endpoint
                ready["speech_filter"] = speech_filter is not None
            if semantic_vad is not None:
                ready["semantic_vad"] = True
            await send(ready)
            while True:
                packet = await socket.receive()
                if packet["type"] == "websocket.disconnect":
                    break
                if packet.get("text") is not None:
                    if len(packet["text"]) > 4096:
                        raise ValueError("通话消息过长。")
                    message = json.loads(packet["text"])
                    if not isinstance(message, dict):
                        raise ValueError("通话消息格式无效。")
                    kind = message.get("type")
                    turn = message.get("id")
                    if semantic_vad is not None and kind in (
                        "utterance", "utterance_start", "utterance_probe", "utterance_resume",
                        "utterance_end", "utterance_cancel",
                    ):
                        raise ValueError("语义 VAD 通话由服务器划分话轮。")
                    if kind in ("interrupt", "hangup"):
                        if current_event:
                            current_event.set()
                        pending_id = None
                        pending_pcm = None
                        clear_probe()
                        if kind == "hangup":
                            break
                    elif kind in ("utterance", "utterance_start"):
                        if duplex and kind == "utterance":
                            raise ValueError("全双工通话需要分片上传语音。")
                        if type(turn) is not int or not last_id < turn < 1_000_000_000:
                            raise ValueError("通话序号无效，请重新连接。")
                        if pending_id is not None:
                            raise ValueError("上一段录音尚未结束。")
                        pending_id = turn
                        last_id = turn
                        last_probe = 0
                        pending_pcm = bytearray() if kind == "utterance_start" else None
                    elif duplex and kind in ("utterance_probe", "utterance_resume"):
                        sequence = message.get("probe")
                        if not pending_pcm or type(turn) is not int or turn != pending_id or type(sequence) is not int:
                            raise ValueError("语音结束判断状态不同步，请重新连接。")
                        if kind == "utterance_probe":
                            if pending_probe is not None or not last_probe < sequence < 1_000_000_000:
                                raise ValueError("语音结束判断序号无效，请重新连接。")
                            pending_probe = TurnProbe(sequence)
                            last_probe = sequence
                            start_job(assess_turn(turn, pending_probe))
                        elif pending_probe is not None and sequence == pending_probe.sequence:
                            clear_probe()
                        else:
                            raise ValueError("语音续说状态不同步，请重新连接。")
                    elif kind in ("utterance_end", "utterance_cancel"):
                        if pending_pcm is None or type(turn) is not int or turn != pending_id:
                            raise ValueError("录音状态不同步，请重新连接。")
                        if kind == "utterance_end":
                            input_ids = None
                            data = pending_pcm
                            if duplex:
                                sequence = message.get("probe")
                                after_wait = not model_endpoint and message.get("after_wait") is True
                                expected = "wait" if after_wait else "reply"
                                if (pending_probe is None or pending_probe.decision != expected
                                        or type(sequence) is not int or sequence != pending_probe.sequence):
                                    raise ValueError("本轮语音尚未确认结束。")
                                data = pending_probe.model_pcm
                                if pending_probe.pcm_size != len(pending_pcm):
                                    # New samples can contain a real continuation or only noise.
                                    prepared = await asyncio.to_thread(prepare_call_pcm, bytes(pending_pcm))
                                    data = prepared or data
                                if data == pending_probe.model_pcm:
                                    input_ids = pending_probe.input_ids
                            await submit_recording(turn, data, pcm=True, input_ids=input_ids,
                                                   filtered=duplex and speech_filter is not None)
                        clear_probe()
                        pending_id = None
                        pending_pcm = None
                    elif kind == "ping":
                        await send({"type": "pong"})
                    elif kind == "inspect_input":
                        if (last_input and turn == current_id == last_input[0] and last_input[1]
                                and (inspection is None or inspection.done())):
                            inspection = asyncio.create_task(inspect_input(*last_input))
                        elif not inspection or inspection.done():
                            await send({"type": "input_audio", "id": turn, "message": "这轮的语音 token 尚未就绪。"})
                elif packet.get("bytes") is not None:
                    data = packet["bytes"]
                    if semantic_vad is not None:
                        if not data or len(data) % INPUT_SAMPLE_BYTES or len(data) > INPUT_SAMPLE_RATE * INPUT_SAMPLE_BYTES:
                            raise ValueError("语义 VAD 需要不超过一秒的 PCM16 音频分片。")
                        try:
                            events = await semantic_vad.append(data)
                        except Exception as error:
                            LOG.exception("Call %s streaming Silero failed", session)
                            raise ValueError("语义 VAD 处理失败，请检查服务端日志。") from error
                        for event in events:
                            if event.reason == "speech_start":
                                last_id += 1
                                await send({"type": "speech_started", "id": last_id})
                                continue
                            LOG.info("Call %s turn %s semantic_vad commit_reason=%s probability=%s audio_ms=%.0f",
                                     session, last_id, event.reason, event.probability,
                                     len(event.pcm) * 1000 / (INPUT_SAMPLE_RATE * INPUT_SAMPLE_BYTES))
                            await send({"type": "speech_stopped", "id": last_id, "reason": event.reason})
                            await submit_recording(last_id, event.pcm, pcm=True, filtered=True)
                        continue
                    if pending_id is None:
                        raise ValueError("语音没有对应的通话消息，请重新连接。")
                    if pending_pcm is None:
                        await submit_recording(pending_id, data)
                        pending_id = None
                    else:
                        if pending_probe is not None and not model_endpoint:
                            raise ValueError("续说前请先恢复语音上传。")
                        if not data or len(data) % INPUT_SAMPLE_BYTES:
                            raise ValueError("语音分片格式无效，请重新说一次。")
                        if len(pending_pcm) + len(data) > max_pcm_bytes:
                            raise ValueError("这段语音过长，请分成短句。")
                        pending_pcm.extend(data)
        except WebSocketDisconnect:
            pass
        except (ValueError, json.JSONDecodeError) as error:
            await send({"type": "error", "message": str(error)})
        finally:
            closed = True
            clear_probe()
            if current_event:
                current_event.set()
            try:
                pending = list(jobs)
                if inspection is not None:
                    pending.append(inspection)
                if semantic_vad is not None:
                    pending.append(semantic_vad.close())
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
            finally:
                slots.append(backend)
            try:
                await socket.close()
            except RuntimeError:
                pass
            LOG.info("Call %s ended", session)

    return app


def parse_args(argv=None):
    parser = create_parser()
    parser.description = __doc__
    parser.set_defaults(output_dir=str(ROOT / "answer_data/calls"), max_new_tokens=None)
    parser.add_argument("--config", default=str(ROOT / "call_config.json"))
    parser.add_argument("--smart-turn-model", type=Path, metavar="ONNX",
                        help="Smart Turn v3.2 CPU ONNX 本地路径，用于启用 /duplex?vad=semantic")
    parser.add_argument("--replica-gpus", default="", metavar="0,1",
                        help="每张指定的 CUDA 卡各加载一套模型，每套独立服务一路通话")
    parser.add_argument("--download-models", action="store_true", help="Download missing decoder weights and enable online model loading in this process")
    parser.add_argument("--check", action="store_true", help="Check decoder files and configuration without loading GPU models")
    args = parser.parse_args(argv)
    if args.smart_turn_model is not None:
        args.smart_turn_model = args.smart_turn_model.expanduser().resolve()
        if not args.smart_turn_model.is_file():
            parser.error(f"找不到 Smart Turn 权重：{args.smart_turn_model}")
    if args.max_new_tokens is None:
        args.max_new_tokens = 1024 if args.backend == "glm" else 512
    args = validate_args(args, parser)
    devices = [value.strip() for value in args.replica_gpus.split(",")] if args.replica_gpus else []
    if any(not re.fullmatch(r"[0-9]+", value) for value in devices):
        parser.error("--replica-gpus 需要逗号分隔的 CUDA 序号，例如 0,1")
    args.replica_gpus = tuple(map(int, devices))
    if len(set(args.replica_gpus)) != len(args.replica_gpus):
        parser.error("--replica-gpus 中每张卡只能指定一次")
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.chdir(ROOT)
    if args.download_models:
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            os.environ[key] = "0"
    config = load_config(args.config)
    if args.check:
        prepare_decoder(args)
        if args.backend == "glm":
            print("Decoder 文件和客服配置检查通过。GPU 模型尚未加载。")
        else:
            print(f"{BACKENDS[args.backend][0]} 参数和客服配置检查通过。模型组件将在启动时从 {args.model_path} 加载。")
        return
    import uvicorn
    app = create_app(args, config)
    uvicorn.run(app, host=args.host, port=args.port, ws_max_size=(args.max_audio_seconds + 1) * 32000 + 8192)


if __name__ == "__main__":
    main()
