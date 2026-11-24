"""One-button voice calls with selectable local and hosted voice backends."""
from __future__ import annotations

import asyncio
import base64
from contextlib import asynccontextmanager, nullcontext
from copy import copy
from datetime import datetime, timezone
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import threading
import time
import uuid
import wave
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, UUID4

from voice_service import ROOT
from voice_service.backend import (
    API_BACKENDS, BACKENDS, DEFAULT_OUTPUT_SAMPLE_RATE, STREAM_BLOCK_SIZES, InferenceCancelled, TurnDecision, VoiceBackendBase,
    check_cancelled,
)
from voice_service.cli import create_backend, create_parser, validate_args
from voice_service.input import CallSpeechFilter
from voice_service.links import selected_faq_links
from voice_service.semantic_vad import SemanticVAD, SmartTurn

LOG = logging.getLogger("voice-call")
ASSETS = ROOT / "call_web"
FEEDBACK_DIR = ROOT / "数据飞轮"
MAX_FEEDBACK_BYTES = 16 * 1024 * 1024
INPUT_TOKEN = re.compile(r"<\|audio_(\d+)\|>")
INPUT_SAMPLE_RATE = 16000
INPUT_SAMPLE_BYTES = 2
FAREWELL_TEXT = (
    "亲亲，感谢您对雷士的支持，如果您后续有其他任何需要咨询的问题欢迎随时进线哦。"
    "祝您生活愉快！再见。"
)


class FeedbackTurn(BaseModel):
    id: UUID4
    turn: int = Field(gt=0, lt=1_000_000_000)
    status: str = Field(default="", max_length=500)
    input_meta: str = Field(default="", max_length=500)
    reply: str = Field(default="", max_length=80_000)
    retrieval: dict | None = None
    question_wav: str | None = None
    reply_wav: str | None = None


def feedback_audio(encoded):
    if encoded is None:
        return None
    data = base64.b64decode(encoded, validate=True)
    if len(data) <= 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("录音格式无效。")
    return data


def save_feedback_turn(turn):
    question_audio = feedback_audio(turn.question_wav)
    reply_audio = feedback_audio(turn.reply_wav)
    if not (question_audio or reply_audio or turn.reply or turn.retrieval):
        raise ValueError("这轮暂无可保存的内容。")
    destination = FEEDBACK_DIR / str(turn.id)
    if (destination / "record.json").is_file():
        return

    record = turn.model_dump(mode="json", exclude={"question_wav", "reply_wav"})
    record["saved_at"] = datetime.now(timezone.utc).isoformat()
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".pending-", dir=FEEDBACK_DIR) as temporary:
        folder = Path(temporary)
        for kind, audio in (("question", question_audio), ("reply", reply_audio)):
            filename = f"{kind}.wav" if audio else None
            record[f"{kind}_audio"] = filename
            if filename:
                (folder / filename).write_bytes(audio)
        (folder / "record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                            encoding="utf-8")
        try:
            folder.rename(destination)
        except OSError:
            if not (destination / "record.json").is_file():
                raise


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
        template = ROOT / "resources/call_config.example.json"
        path.write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("language", "zh") not in ("zh", "en"):
        raise ValueError("call_config.json language must be zh or en")
    greeting = config.get("greeting")
    if greeting is not None and not isinstance(greeting, str):
        raise ValueError("call_config.json greeting must be text")
    config["greeting"] = (greeting or "").strip()
    if not isinstance(config.get("farewell_enabled", True), bool):
        raise ValueError("call_config.json farewell_enabled must be true or false")
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


def resolve_smart_turn_model(args, config):
    if args.smart_turn_model is not None:
        return args.smart_turn_model
    model = config.get("smart_turn_model", "")
    if not isinstance(model, str):
        raise ValueError("call_config.json 的 smart_turn_model 必须是路径字符串。")
    model = model.strip()
    if not model:
        return None
    path = (ROOT / Path(model).expanduser()).resolve()
    if not path.is_file():
        raise ValueError(f"找不到 Smart Turn 权重：{path}。请检查 call_config.json 的 smart_turn_model。")
    return path


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
    if getattr(args, "vllm_base_url", None):
        identity.extend([args.vllm_base_url, args.vllm_model])
    if args.flow_steps != 10:
        identity.append(args.flow_steps)
    sample_rate = getattr(backend, "output_sample_rate", DEFAULT_OUTPUT_SAMPLE_RATE)
    identity.extend([getattr(args, "backend", "glm"), sample_rate, args.tokenizer_path,
                     getattr(args, "omni_speaker", "Chelsie")])
    if args.backend in API_BACKENDS:
        identity.extend(backend.speech_cache_identity())
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
    farewell_text = FAREWELL_TEXT if config.get("farewell_enabled", True) else ""
    language = config.get("language", "zh")
    knowledge = config.get("knowledge_text", "").strip()
    smart_turn_model = resolve_smart_turn_model(args, config)
    backends = [backend] if backend is not None else build_backends(args)
    active_calls = [0] * len(backends)
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
        if smart_turn_model is not None and smart_turn is None:
            try:
                smart_turn = SmartTurn(smart_turn_model)
            except Exception:
                LOG.exception("语义 VAD 模型加载失败：%s", smart_turn_model)
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
        _, ending_path = prepare_clip(farewell_text, "zh")
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
                     args.max_calls, args.port)

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
    @app.get("/manual")
    async def home():
        return RedirectResponse("/omni_vad", headers={"Cache-Control": "no-store"})

    @app.get("/omni_vad")
    @app.get("/omni_vad/vllm")
    async def index(request: Request):
        if request.url.path == "/omni_vad/vllm" and not getattr(args, "vllm_base_url", None):
            raise HTTPException(status_code=404)
        return FileResponse(ASSETS / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/style.css")
    @app.get("/omni_vad/style.css", include_in_schema=False)
    async def stylesheet():
        return FileResponse(ASSETS / "style.css", media_type="text/css")

    @app.post("/api/feedback")
    async def save_feedback(request: Request):
        origin = request.headers.get("origin")
        if origin and urlparse(origin).netloc != request.headers.get("host"):
            raise HTTPException(status_code=403, detail="请从当前语音通话页提交记录。")
        if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
            raise HTTPException(status_code=415, detail="记录格式无效。")
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > MAX_FEEDBACK_BYTES:
                raise HTTPException(status_code=413, detail="这轮的语音文件太大，无法保存。")
            body.extend(chunk)
        try:
            turn = FeedbackTurn.model_validate_json(body)
            await asyncio.to_thread(save_feedback_turn, turn)
        except ValueError as error:
            raise HTTPException(status_code=422, detail="记录或语音格式无效。") from error
        except OSError as error:
            LOG.exception("Failed to save feedback turn")
            raise HTTPException(status_code=500, detail="数据飞轮写入失败，请重试。") from error
        return {"path": f"{FEEDBACK_DIR.name}/{turn.id}"}

    @app.get("/api/farewell/{filename}")
    async def farewell(filename: str):
        if farewell_path is None or filename != farewell_path.name or not farewell_path.is_file():
            raise HTTPException(status_code=404)
        return FileResponse(farewell_path, media_type="audio/wav", headers={"Cache-Control": "no-cache"})

    @app.get("/api/health")
    async def health():
        free = args.max_calls - sum(active_calls) if phase["state"] == "ready" else 0
        return {"state": phase["state"], "available": free > 0, "available_slots": free,
                "max_calls": args.max_calls, "title": config.get("title", "语音客服")}

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
        if phase["state"] != "ready" or sum(active_calls) >= args.max_calls:
            await socket.send_json({"type": "error", "message": "客服暂未就绪或正在通话，请稍后再试。"})
            await socket.close(code=1013)
            return
        stream_audio = socket.query_params.get("stream") == "1"
        use_semantic_vad = socket.query_params.get("vad") == "semantic"
        if not use_semantic_vad and (socket.query_params.get("duplex") == "1"
                                     or socket.query_params.get("model_endpoint") == "1"):
            await socket.send_json({"type": "error", "message": "旧版通话模式已停用，请访问 /omni_vad。"})
            await socket.close(code=1008)
            return
        if use_semantic_vad and smart_turn is None:
            message = ("语义 VAD 模型加载失败，请检查服务端日志。" if smart_turn_model else
                       "请在 call_config.json 中设置 smart_turn_model，或用 --smart-turn-model 指定权重。")
            await socket.send_json({"type": "error", "message": message})
            await socket.close(code=1008)
            return
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
        backend_index = min(range(len(backends)), key=active_calls.__getitem__)
        backend = backends[backend_index]
        concurrent_inference = getattr(backend, "supports_concurrent_inference", False)
        inference_lock = asyncio.Lock()
        session = uuid.uuid4().hex
        incoming = Path(args.output_dir).resolve() / "incoming" / session
        history = []
        current_event = None
        current_id = 0
        last_id = 0
        pending_id = None
        pending_pcm = None
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

        def prepare_call_pcm(pcm):
            return speech_filter(pcm) if speech_filter is not None else pcm

        def model_lock():
            return nullcontext() if concurrent_inference else backend.lock

        def judge_call_turn(pcm, previous_reply, event):
            with model_lock(), device_scope(backend):
                check_cancelled(event)
                decision, input_ids = backend.judge_turn(pcm, INPUT_SAMPLE_RATE, language, event,
                                                         previous_reply=previous_reply)
                if decision.action == "echo":
                    return backend.judge_turn(pcm, INPUT_SAMPLE_RATE, language, event, previous_reply=previous_reply,
                                              include_previous_reply=False)
                return decision, input_ids

        def infer(path, saved_history, event, on_audio=None, profile=None, on_retrieval=None, *, decision, input_ids=None):
            options = {}
            if input_ids is not None:
                options["input_ids"] = input_ids
            if decision.action == "clarify":
                options["clarify"] = True
            if decision.action == "courtesy":
                options["courtesy"] = True
            if decision.faq_request:
                options["faq_request"] = decision.faq_request
            if on_audio is not None:
                options["on_audio"] = on_audio
            if on_retrieval is not None:
                options["on_retrieval"] = on_retrieval
            if profile:
                options["profile"] = profile.mark
                if requested_blocks is not None:
                    options["stream_block_sizes"] = block_sizes
                profile.mark("worker_started")
            with model_lock(), device_scope(backend):
                if event.is_set():
                    raise InferenceCancelled()
                if profile and not concurrent_inference:
                    profile.mark("backend_lock_acquired")
                return backend.answer(
                    str(path), bool(knowledge) and decision.action == "reply", knowledge, None, language, saved_history,
                    cancel_event=event,
                    **options,
                )

        async def stream_reply(turn, path, event, profile, input_ids, decision):
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
                        profile=profile, input_ids=input_ids, decision=decision,
                        on_retrieval=publish_retrieval if use_semantic_vad and knowledge and decision.action == "reply" else None,
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
                        await send({"type": "retrieval", "id": turn, "retrieval": payload,
                                    "links": selected_faq_links(payload)})
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

        async def respond(turn, path, event, profile, semantic_pcm):
            nonlocal history, last_input
            try:
                async with inference_lock:
                    if event.is_set() or closed:
                        return
                    if profile:
                        profile.mark("inference_slot_acquired")
                    input_ids = None
                    decision = TurnDecision("reply")
                    if semantic_pcm is not None:
                        previous_reply = history[-1]["text"][-500:] if history and args.history_turns else ""
                        if profile:
                            profile.mark("turn_check_start")
                        decision, input_ids = await asyncio.to_thread(judge_call_turn, semantic_pcm, previous_reply, event)
                        if event.is_set() or closed or turn != current_id:
                            return
                        if profile:
                            profile.mark("turn_check_end", decision=decision.action)
                        LOG.info("Call %s turn %s turn check: %s", session, turn, decision.action)
                        if decision.action not in ("reply", "clarify", "courtesy"):
                            await send({"type": "turn_ignored", "id": turn})
                            return
                        if acknowledge_input:
                            await send({"type": "utterance_received", "id": turn,
                                        "clarify": decision.action == "clarify",
                                        "courtesy": decision.action == "courtesy",
                                        "input_pcm": base64.b64encode(semantic_pcm).decode("ascii")})
                    await send({"type": "thinking", "id": turn})
                    if stream_audio:
                        result, streamed = await stream_reply(turn, path, event, profile, input_ids, decision)
                    else:
                        result = await asyncio.to_thread(infer, path, list(history), event,
                                                         input_ids=input_ids, decision=decision)
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
                        reply["links"] = selected_faq_links(record["retrieval"])
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
            with model_lock(), device_scope(backend):
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

        async def submit_recording(turn, data, *, pcm=False, filtered=False):
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
            if preloaded_query_pcm and not filtered:
                await send({"type": "utterance_received", "id": turn})
            start_job(respond(turn, path, current_event, profile, data if filtered else None))

        active_calls[backend_index] += 1
        try:
            incoming.mkdir(parents=True, exist_ok=True)
            if use_semantic_vad:
                try:
                    semantic_vad = await asyncio.to_thread(SemanticVAD, smart_turn, args.max_audio_seconds)
                except Exception as error:
                    LOG.exception("Call %s failed to initialize streaming Silero", session)
                    raise ValueError("语义 VAD 初始化失败，请检查服务端日志。") from error
            ready = {"type": "ready", "greeting": greeting, "greeting_audio": greeting_audio,
                     "farewell": farewell_text,
                     "farewell_audio_url": f"/api/farewell/{farewell_path.name}" if farewell_path else None,
                     "query_pcm": preloaded_query_pcm,
                     "output_sample_rate": getattr(backend, "output_sample_rate", DEFAULT_OUTPUT_SAMPLE_RATE),
                     "language": language, "silence_ms": config["silence_ms"],
                     "min_speech_ms": config["min_speech_ms"], "max_audio_seconds": args.max_audio_seconds,
                     "auto_listen": config.get("auto_listen", False)}
            if semantic_vad is not None:
                ready["duplex"] = True
                ready["speech_filter"] = speech_filter is not None
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
                    if kind in ("utterance_probe", "utterance_resume"):
                        raise ValueError("旧版话轮判定协议已停用。")
                    if semantic_vad is not None and kind in (
                        "utterance", "utterance_start", "utterance_end", "utterance_cancel",
                    ):
                        raise ValueError("语义 VAD 通话由服务器划分话轮。")
                    if kind in ("interrupt", "hangup"):
                        if current_event:
                            current_event.set()
                        pending_id = None
                        pending_pcm = None
                        if kind == "hangup":
                            break
                    elif kind in ("utterance", "utterance_start"):
                        if type(turn) is not int or not last_id < turn < 1_000_000_000:
                            raise ValueError("通话序号无效，请重新连接。")
                        if pending_id is not None:
                            raise ValueError("上一段录音尚未结束。")
                        pending_id = turn
                        last_id = turn
                        pending_pcm = bytearray() if kind == "utterance_start" else None
                    elif kind in ("utterance_end", "utterance_cancel"):
                        if pending_pcm is None or type(turn) is not int or turn != pending_id:
                            raise ValueError("录音状态不同步，请重新连接。")
                        if kind == "utterance_end":
                            await submit_recording(turn, pending_pcm, pcm=True)
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
                            try:
                                model_pcm = await asyncio.to_thread(prepare_call_pcm, event.pcm)
                            except Exception:
                                LOG.exception("Call %s turn %s speech filter failed", session, last_id)
                                await send({"type": "speech_discarded", "id": last_id,
                                            "message": "暂时无法处理这段录音，请再说一次。"})
                                continue
                            if not model_pcm:
                                LOG.info("Call %s turn %s semantic_vad discarded by speech filter, endpoint=%s",
                                         session, last_id, event.reason)
                                await send({"type": "speech_discarded", "id": last_id})
                                continue
                            LOG.info("Call %s turn %s semantic_vad commit_reason=%s probability=%s audio_ms=%.0f",
                                     session, last_id, event.reason, event.probability,
                                     len(event.pcm) * 1000 / (INPUT_SAMPLE_RATE * INPUT_SAMPLE_BYTES))
                            await send({"type": "speech_stopped", "id": last_id, "reason": event.reason})
                            await submit_recording(last_id, model_pcm, pcm=True, filtered=True)
                        continue
                    if pending_id is None:
                        raise ValueError("语音没有对应的通话消息，请重新连接。")
                    if pending_pcm is None:
                        await submit_recording(pending_id, data)
                        pending_id = None
                    else:
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
                active_calls[backend_index] -= 1
            try:
                await socket.close()
            except RuntimeError:
                pass
            LOG.info("Call %s ended", session)

    return app


def parse_args(argv=None):
    parser = create_parser(__doc__)
    parser.set_defaults(output_dir=str(ROOT / "answer_data/calls"), max_new_tokens=None)
    parser.add_argument("--config", default=str(ROOT / "call_config.json"))
    parser.add_argument("--smart-turn-model", type=Path, metavar="ONNX",
                        help="Smart Turn v3.2 CPU ONNX 本地路径，覆盖配置中的 smart_turn_model")
    parser.add_argument("--replica-gpus", default="", metavar="0,1",
                        help="每张指定的 CUDA 卡各加载一套模型，通话分配给当前接入人数最少的副本")
    parser.add_argument("--max-calls", type=int, default=5, help="每个服务实例同时接入的通话上限（默认 5）")
    parser.add_argument("--download-models", action="store_true", help="Download missing decoder weights and enable online model loading in this process")
    parser.add_argument("--check", action="store_true", help="Check decoder files and configuration without loading GPU models")
    args = parser.parse_args(argv)
    if args.smart_turn_model is not None:
        args.smart_turn_model = args.smart_turn_model.expanduser().resolve()
        if not args.smart_turn_model.is_file():
            parser.error(f"找不到 Smart Turn 权重：{args.smart_turn_model}")
    if args.max_new_tokens is None:
        args.max_new_tokens = 1024 if args.backend == "glm" else 512
    if args.max_calls < 1:
        parser.error("--max-calls 必须大于零")
    args = validate_args(args, parser)
    devices = [value.strip() for value in args.replica_gpus.split(",")] if args.replica_gpus else []
    if devices and args.backend in API_BACKENDS:
        parser.error("--replica-gpus 仅适用于在通话进程内加载模型的后端。")
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
        resolve_smart_turn_model(args, config)
        prepare_decoder(args)
        if args.backend == "glm":
            print("Decoder 文件和客服配置检查通过。GPU 模型尚未加载。")
        elif args.backend == "minicpmo-local":
            create_backend(args).load()
            print(f"{BACKENDS[args.backend][0]} 模型列表和客服配置检查通过。")
        elif args.backend == "minicpmo-gguf":
            create_backend(args).load()
            print(f"{BACKENDS[args.backend][0]} 推理服务和客服配置检查通过。")
        elif args.backend in API_BACKENDS:
            create_backend(args).load()
            print(f"{BACKENDS[args.backend][0]} 参数和客服配置检查通过。尚未发送 API 请求。")
        else:
            print(f"{BACKENDS[args.backend][0]} 参数和客服配置检查通过。模型组件将在启动时从 {args.model_path} 加载。")
        return
    import uvicorn
    app = create_app(args, config)
    uvicorn.run(app, host=args.host, port=args.port, ws_max_size=(args.max_audio_seconds + 1) * 32000 + 8192)


if __name__ == "__main__":
    main()
