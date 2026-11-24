"""Local and hosted APIs for audio understanding and speech synthesis."""
from __future__ import annotations

import asyncio
import base64
from contextlib import suppress
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import threading
from urllib.parse import quote, urlencode

import httpx

from .audio_models import AudioModelBackend
from .backend import LOG, VoiceReply, check_cancelled, pcm16_bytes, waveform_wav


async def await_request(request, cancel_event):
    task = asyncio.create_task(request)
    try:
        while not task.done():
            check_cancelled(cancel_event)
            await asyncio.wait((task,), timeout=0.05)
        check_cancelled(cancel_event)
        return task.result()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def single_audio_messages(messages):
    """Keep text history for runtimes accepting only the current audio file."""
    history = []
    for message in messages[:-1]:
        if message.get("text"):
            history.append(f"{message['role']}:\n{message['text']}")
    current = messages[-1]
    if history:
        current = {**current, "text": "[Conversation history]\n" + "\n\n".join(history)
                                     + "\n[/Conversation history]\n\n" + current.get("text", "")}
    return [current]


class HostedAudioBackend(AudioModelBackend):
    def load(self):
        if self.ready:
            return
        key = os.environ.get(self.key_env, "").strip()
        if not key:
            raise RuntimeError(f"请先设置环境变量 {self.key_env}。")

        import torch

        self.speech_device = torch.device(self.args.speech_device)
        self.headers = {"Authorization": f"Bearer {key}"}
        self.ready = True

    def speech_cache_identity(self):
        return [self.base_url]


class ChatAPIBackend(HostedAudioBackend):
    def chat_messages(self, messages):
        result = []
        for message in messages:
            text = message.get("text", "")
            if message.get("audio"):
                data = base64.b64encode(Path(message["audio"]).read_bytes()).decode("ascii")
                content = [{"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{data}"}}]
                if text:
                    content.append({"type": "text", "text": text})
            else:
                content = text
            result.append({"role": message["role"], "content": content})
        return result

    async def _request(self, path, payload, cancel_event, read_stream):
        async def send(client):
            url = f"{self.base_url}/{path}"
            if read_stream is None:
                return (await client.post(url, json=payload)).raise_for_status()
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                return await read_stream(response)

        async with httpx.AsyncClient(headers=self.headers, timeout=httpx.Timeout(90, connect=10)) as client:
            return await await_request(send(client), cancel_event)

    def request(self, path, payload, cancel_event, *, read_stream=None):
        check_cancelled(cancel_event)
        try:
            return asyncio.run(self._request(path, payload, cancel_event, read_stream))
        except httpx.HTTPStatusError as error:
            raise RuntimeError(f"{self.provider} {path} 请求失败（HTTP {error.response.status_code}）。") from error
        except httpx.RequestError as error:
            raise RuntimeError(f"{self.provider} {path} 连接失败或超时。") from error


class AudioAPIBackend(ChatAPIBackend):
    max_tokens_field = "max_tokens"

    def speech_cache_identity(self):
        return [self.base_url, self.tts_model, self.tts_voice]

    def speech_payload(self, text, response_format):
        return {
            "model": self.tts_model, "voice": self.tts_voice, "input": text,
            "response_format": response_format, "sample_rate": self.output_sample_rate,
        }

    def synthesize(self, text, cancel_event):
        import soundfile as sf
        import torch
        import torchaudio

        result = self.request("audio/speech", self.speech_payload(text, "wav"), cancel_event)
        try:
            samples, rate = sf.read(io.BytesIO(result.content), dtype="float32", always_2d=True)
        except (RuntimeError, ValueError) as error:
            raise RuntimeError(f"{self.provider} 返回的语音不是有效的 WAV。") from error
        if not samples.size or rate <= 0:
            raise RuntimeError(f"{self.provider} 返回了空语音。")
        waveform = torch.from_numpy(samples.mean(axis=1)).unsqueeze(0)
        if rate != self.output_sample_rate:
            waveform = torchaudio.functional.resample(waveform, rate, self.output_sample_rate)
        check_cancelled(cancel_event)
        return waveform

    def synthesize_stream(self, text, cancel_event, on_audio, profile):
        import numpy as np
        import torch

        async def read_stream(response):
            pcm = bytearray()
            remainder = b""
            packet = 0
            async for data in self.speech_chunks(response):
                check_cancelled(cancel_event)
                data = remainder + data
                end = len(data) - len(data) % 2
                chunk, remainder = data[:end], data[end:]
                if chunk:
                    pcm.extend(chunk)
                    packet += 1
                    if profile:
                        profile("chunk_pcm_ready", chunk=packet, packet=packet)
                    on_audio(chunk)
            if remainder:
                raise RuntimeError(f"{self.provider} 返回的 PCM 样本不完整。")
            if not pcm:
                raise RuntimeError(f"{self.provider} 返回了空语音。")
            return pcm

        payload = {**self.speech_payload(text, "pcm"), **self.speech_stream_options}
        pcm = self.request("audio/speech", payload, cancel_event, read_stream=read_stream)
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
        return torch.from_numpy(samples).unsqueeze(0)

    def synthesize_greeting(self, greeting, language):
        return waveform_wav(self.synthesize(greeting, None), self.output_sample_rate)

    def infer_reply(self, messages, cancel_event, on_audio, profile):
        return self.infer(messages, True, self.args.max_new_tokens, cancel_event, profile, on_audio=on_audio)

    def infer(self, messages, speech, max_new_tokens, cancel_event, profile=None, *, on_audio=None):
        check_cancelled(cancel_event)
        payload = {
            "model": self.args.model_path, "messages": self.chat_messages(messages),
            self.max_tokens_field: max_new_tokens, "stream": False,
        }
        if profile:
            profile("generate_start")
        result = self.request("chat/completions", payload, cancel_event)
        try:
            text = result.json()["choices"][0]["message"]["content"].strip()
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as error:
            raise RuntimeError(f"{self.provider} 返回的文本回答格式无效。") from error
        if not text:
            raise RuntimeError(f"{self.provider} 返回了空回答。")
        if profile:
            profile("generate_end")
        waveform = None
        if speech:
            if on_audio is None:
                waveform = self.synthesize(text, cancel_event)
            else:
                waveform = self.synthesize_stream(text, cancel_event, on_audio, profile)
        return VoiceReply(text, waveform)


class MingAPIBackend(AudioAPIBackend):
    provider = "蚂蚁百灵"
    key_env = "ANT_LING_API_KEY"
    base_url = "https://api.ant-ling.com/v1"
    max_tokens_field = "max_completion_tokens"
    tts_model = "Ming-3.0-tts-0.5B"
    tts_voice = "灵小柔"
    speech_stream_options = {"stream_format": "sse"}

    async def speech_chunks(self, response):
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            try:
                event = json.loads(line[5:])
                if not isinstance(event, dict):
                    raise ValueError("expected a JSON object")
                if event.get("type") == "speech.audio.delta":
                    yield base64.b64decode(event["audio"], validate=True)
                elif event.get("type") == "speech.audio.done":
                    return
                elif "error" in event or event.get("type") in ("error", "speech.audio.error"):
                    raise RuntimeError(f"{self.provider} 流式语音合成失败。")
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(f"{self.provider} 返回了无效的 SSE 音频事件。") from error
        raise RuntimeError(f"{self.provider} 的 SSE 音频流提前断开。")

    def chat_messages(self, messages):
        # The hosted API accepts one audio file per request.
        return super().chat_messages(single_audio_messages(messages))


class QwenOmniAPIBackend(AudioAPIBackend):
    provider = "硅基流动"
    key_env = "SILICONFLOW_API_KEY"
    base_url = "https://api.siliconflow.cn/v1"
    tts_model = "FunAudioLLM/CosyVoice2-0.5B"
    tts_voice = "FunAudioLLM/CosyVoice2-0.5B:claire"
    speech_stream_options = {"stream": True}

    async def speech_chunks(self, response):
        content_type = response.headers.get("content-type", "").partition(";")[0].strip().lower()
        if content_type.endswith("json") or content_type.startswith("text/"):
            raise RuntimeError(f"{self.provider} 未返回 PCM 音频流。")
        async for chunk in response.aiter_bytes():
            yield chunk


class MiniCPMOVLLMBackend(ChatAPIBackend):
    provider = "MiniCPM-o 4.5 vLLM-Omni"
    normalize_spoken_links = True

    def __init__(self, args):
        super().__init__(args)
        self.base_url = args.minicpmo_base_url.rstrip("/")

    def load(self):
        if self.ready:
            return
        import torch

        key = os.environ.get("MINICPMO_VLLM_API_KEY", "").strip()
        self.headers = {"Authorization": f"Bearer {key}"} if key else {}
        try:
            response = httpx.get(f"{self.base_url}/models", headers=self.headers, timeout=10).raise_for_status()
        except httpx.HTTPStatusError as error:
            raise RuntimeError(f"{self.provider} 模型列表请求失败（HTTP {error.response.status_code}）。") from error
        except httpx.RequestError as error:
            raise RuntimeError(f"无法连接 {self.provider}，请检查 --minicpmo-base-url 和模型服务。") from error
        try:
            served_models = {model["id"] for model in response.json()["data"]}
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"{self.provider} 返回的模型列表格式无效。") from error
        if self.args.model_path not in served_models:
            raise RuntimeError(f"{self.provider} 模型列表中没有 {self.args.model_path}，请核对 --model-path。")
        self.speech_device = torch.device(self.args.speech_device)
        self.ready = True

    async def read_chat_stream(self, response, speech, cancel_event, on_audio, profile):
        import numpy as np
        import soundfile as sf
        import torch

        text, audio = [], []
        async for line in response.aiter_lines():
            check_cancelled(cancel_event)
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
                if "error" in event:
                    raise RuntimeError(f"{self.provider} 流式生成失败。请检查模型服务日志。")
                for choice in event["choices"]:
                    content = choice["delta"].get("content")
                    if content is None or content == "":
                        continue
                    if not isinstance(content, str):
                        raise TypeError("expected string content")
                    if event.get("modality", "text") == "text":
                        text.append(content)
                    elif speech and event.get("modality") == "audio":
                        # vLLM-Omni sends a separate base64 WAV in each audio delta.
                        wav = base64.b64decode(content, validate=True)
                        samples, rate = sf.read(io.BytesIO(wav), dtype="float32", always_2d=True)
                        if rate != self.output_sample_rate or samples.shape[1] != 1 or not np.isfinite(samples).all():
                            raise ValueError("expected finite mono 24 kHz audio")
                        if not samples.size:
                            continue
                        waveform = torch.from_numpy(samples.T)
                        audio.append(waveform)
                        if on_audio is not None:
                            if profile:
                                profile("chunk_pcm_ready", chunk=len(audio), packet=len(audio))
                            on_audio(pcm16_bytes(waveform))
            except (AttributeError, KeyError, TypeError, ValueError, sf.LibsndfileError) as error:
                raise RuntimeError(f"{self.provider} 返回了无效的文本或 WAV 音频事件。") from error
        else:
            raise RuntimeError(f"{self.provider} 的响应流提前断开。")
        answer = "".join(text).strip()
        if not answer or (speech and not audio):
            raise RuntimeError(f"{self.provider} 未返回完整的文本和所请求的语音。")
        return VoiceReply(answer, torch.cat(audio, dim=1) if audio else None)

    def infer_reply(self, messages, cancel_event, on_audio, profile):
        return self.infer(messages, True, self.args.max_new_tokens, cancel_event, profile, on_audio=on_audio)

    def infer(self, messages, speech, max_new_tokens, cancel_event, profile=None, *, on_audio=None):
        check_cancelled(cancel_event)
        payload = {
            "model": self.args.model_path, "messages": self.chat_messages(messages),
            "max_tokens": max_new_tokens, "stream": True,
            "modalities": ["text", "audio"] if speech else ["text"],
            "chat_template_kwargs": {"use_tts_template": speech, "enable_thinking": False},
        }
        if profile:
            profile("generate_start")

        async def read_stream(response):
            return await self.read_chat_stream(response, speech, cancel_event, on_audio, profile)

        reply = self.request("chat/completions", payload, cancel_event, read_stream=read_stream)
        if profile:
            profile("generate_end")
        return reply


class MiniCPMOAPIBackend(HostedAudioBackend):
    supports_concurrent_inference = True
    normalize_spoken_links = True
    key_env = "MAP_API_KEY"
    base_url = "wss://api.modelbest.cn/v1/realtime"
    input_sample_rate = 16000

    def __init__(self, args):
        super().__init__(args)
        self.retrieval_lock = threading.Lock()

    def retrieve(self, question_path, content, language, profile=None):
        # SONAR and the FAQ index are shared between calls. Each MAP session is independent.
        with self.retrieval_lock:
            if profile:
                profile("sonar_lock_acquired")
            return super().retrieve(question_path, content, language, profile)

    def chat_messages(self, messages, cancel_event):
        import soundfile as sf
        import torch
        import torchaudio

        result = []
        for message in messages:
            check_cancelled(cancel_event)
            text = message.get("text", "")
            content = text
            if path := message.get("audio"):
                samples, rate = sf.read(path, dtype="float32", always_2d=True)
                mono = torch.from_numpy(samples.mean(axis=1))
                if rate != self.input_sample_rate:
                    mono = torchaudio.functional.resample(mono, rate, self.input_sample_rate)
                pcm = mono.numpy().astype("<f4", copy=False).tobytes()
                content = [{"type": "audio", "data": base64.b64encode(pcm).decode("ascii")}]
                if text:
                    content.append({"type": "text", "text": text})
            result.append({"role": message["role"], "content": content})
        return result

    async def receive_event(self, socket):
        try:
            event = json.loads(await asyncio.wait_for(socket.recv(), timeout=90))
            if not isinstance(event, dict) or not isinstance(event.get("type"), str):
                raise ValueError("expected an event object with a type")
        except ValueError as error:
            raise RuntimeError("MiniCPM-o 返回了无效的 WebSocket 事件。") from error
        if event["type"] == "error":
            detail = event.get("error")
            code = detail.get("code", "unknown") if isinstance(detail, dict) else "unknown"
            raise RuntimeError(f"MiniCPM-o API 返回错误（{code}）。")
        if event["type"] == "session.closed":
            raise RuntimeError(f"MiniCPM-o 会话提前结束（{event.get('reason', 'unknown')}）。")
        return event

    async def read_reply(self, socket, speech, cancel_event, on_audio, profile):
        import numpy as np
        import torch

        text_parts, audio_parts = [], []

        def add_audio(data):
            try:
                samples = np.frombuffer(base64.b64decode(data, validate=True), dtype="<f4")
                if not samples.size or not np.isfinite(samples).all():
                    raise ValueError("empty or non-finite PCM")
            except (TypeError, ValueError) as error:
                raise RuntimeError("MiniCPM-o 返回了无效的 PCM 音频。") from error
            waveform = torch.from_numpy(samples.astype(np.float32, copy=True)).unsqueeze(0)
            audio_parts.append(waveform)
            if on_audio is not None:
                if profile:
                    profile("chunk_pcm_ready", chunk=len(audio_parts), packet=len(audio_parts))
                on_audio(pcm16_bytes(waveform))

        while True:
            event = await self.receive_event(socket)
            check_cancelled(cancel_event)
            if event["type"] == "response.done":
                if speech and not audio_parts and event.get("audio"):
                    add_audio(event["audio"])
                text = event.get("text") or "".join(text_parts)
                if not isinstance(text, str) or not text.strip():
                    raise RuntimeError("MiniCPM-o 返回了空回答或无效的文本。")
                waveform = torch.cat(audio_parts, dim=1) if audio_parts else None
                return VoiceReply(text.strip(), waveform)
            if event["type"] == "response.output.delta":
                if event.get("kind") == "text" and isinstance(event.get("text"), str):
                    if profile and not text_parts and event["text"]:
                        profile("minicpmo_first_text")
                    if event["text"]:
                        text_parts.append(event["text"])
                elif speech and event.get("kind") == "audio":
                    add_audio(event.get("audio"))

    async def chat(self, payload, speech, cancel_event, on_audio, profile):
        from websockets.exceptions import ConnectionClosed
        from websockets.legacy.client import connect

        query = urlencode({"mode": "chat", "model": self.args.model_path})
        async with connect(f"{self.base_url}?{query}", extra_headers=self.headers,
                           open_timeout=10, close_timeout=1, max_size=16 * 1024 * 1024) as socket:
            if profile:
                profile("map_connected")
            queued = False
            while True:
                event = await self.receive_event(socket)
                check_cancelled(cancel_event)
                if event["type"] == "session.queue_done":
                    break
                if event["type"] in ("session.queued", "session.queue_update") and not queued:
                    queued = True
                    if profile:
                        profile("map_queued")
            if profile:
                profile("map_queue_done", queued=queued)
            await socket.send(json.dumps({"type": "session.init", "payload": {}}))
            while (await self.receive_event(socket))["type"] != "session.created":
                check_cancelled(cancel_event)
            await socket.send(json.dumps({"type": "input.append", "input": payload}, ensure_ascii=False))
            if profile:
                profile("map_input_sent")
            reply = await self.read_reply(socket, speech, cancel_event, on_audio, profile)
            if profile:
                profile("map_reply_done")
            with suppress(ConnectionClosed):
                await socket.send(json.dumps({"type": "session.close", "reason": "turn_done"}))
            return reply

    def infer_reply(self, messages, cancel_event, on_audio, profile):
        return self.infer(messages, True, self.args.max_new_tokens, cancel_event, profile, on_audio=on_audio)

    def infer(self, messages, speech, max_new_tokens, cancel_event, profile=None, *, on_audio=None):
        from websockets.exceptions import InvalidStatusCode, WebSocketException

        payload = {
            "messages": self.chat_messages(messages, cancel_event),
            "streaming": True,
            "generation": {"max_new_tokens": max_new_tokens},
            "tts": {"enabled": speech},
            "use_tts_template": speech,
        }
        check_cancelled(cancel_event)
        if profile:
            profile("generate_start")
        try:
            reply = asyncio.run(await_request(self.chat(payload, speech, cancel_event, on_audio, profile),
                                              cancel_event))
        except InvalidStatusCode as error:
            raise RuntimeError(f"MiniCPM-o WebSocket 握手失败（HTTP {error.status_code}）。") from error
        except (OSError, TimeoutError, WebSocketException) as error:
            raise RuntimeError("MiniCPM-o WebSocket 连接失败、提前断开或超时。") from error
        if profile:
            profile("generate_end")
        return reply


class MiniCPMOGGUFBackend(MiniCPMOAPIBackend):
    """Connect directly to llama-omni-server's backend WebSocket."""

    supports_concurrent_inference = False
    session_init_timeout = 10

    def __init__(self, args):
        super().__init__(args)
        self.base_url = args.minicpmo_gguf_base_url.rstrip("/")
        self.websocket_url = self.base_url.replace("http", "ws", 1) + "/backend"
        self.reference_audio = None
        self.reference_digest = None
        self.ffmpeg = None

    def prepare_speech(self):
        if self.args.minicpmo_speed != 1:
            self.ffmpeg = shutil.which("ffmpeg")
            if self.ffmpeg is None:
                raise RuntimeError("--minicpmo-speed 变速需要通话主机安装 ffmpeg：apt-get install -y ffmpeg")
        if self.args.minicpmo_ref_audio is not None and self.reference_audio is None:
            import numpy as np
            import soundfile as sf
            import torch
            import torchaudio

            data = self.args.minicpmo_ref_audio.read_bytes()
            try:
                with sf.SoundFile(io.BytesIO(data)) as audio:
                    if audio.format not in ("WAV", "WAVEX") or not 0 < audio.frames / audio.samplerate <= 30:
                        raise ValueError("expected a WAV of 0–30 seconds")
                    samples = audio.read(dtype="float32", always_2d=True)
                if not np.isfinite(samples).all():
                    raise ValueError("non-finite reference audio")
            except (ValueError, sf.LibsndfileError) as error:
                raise ValueError("音色参考必须是 30 秒以内、非空且有效的 WAV。") from error
            mono = torch.from_numpy(samples.mean(axis=1))
            if audio.samplerate != self.input_sample_rate:
                mono = torchaudio.functional.resample(mono, audio.samplerate, self.input_sample_rate)
            self.reference_audio = base64.b64encode(mono.numpy().astype("<f4").tobytes()).decode("ascii")
            self.reference_digest = hashlib.sha256(data).hexdigest()

    def speech_cache_identity(self):
        self.prepare_speech()
        return [self.base_url, self.reference_digest, self.args.minicpmo_speed, self.args.minicpmo_stream]

    def load(self):
        if self.ready:
            return
        self.prepare_speech()
        health_url = f"{self.base_url}/health"
        try:
            httpx.get(health_url, timeout=10, trust_env=False).raise_for_status()
        except httpx.HTTPStatusError as error:
            raise RuntimeError(
                f"llama-omni-server 健康检查失败：{health_url} 返回 HTTP {error.response.status_code}。"
                "请等待模型加载完成，或查看推理服务日志。"
            ) from error
        except httpx.RequestError as error:
            raise RuntimeError(
                f"无法访问 llama-omni-server：{health_url}（{error}）。"
                "通话服务不会启动推理服务；请确认推理服务已启动且端口可达，"
                "必要时设置 --minicpmo-gguf-base-url。"
            ) from error

        import torch

        self.speech_device = torch.device(self.args.speech_device)
        self.ready = True

    def infer(self, messages, speech, max_new_tokens, cancel_event, profile=None, *, on_audio=None):
        self.prepare_speech()
        return super().infer(messages, speech, max_new_tokens, cancel_event, profile, on_audio=on_audio)

    async def read_reply(self, socket, speech, cancel_event, on_audio, profile):
        if not speech or self.args.minicpmo_speed == 1:
            return await super().read_reply(socket, speech, cancel_event, on_audio, profile)
        import numpy as np
        import torch
        from .tempo import StreamingTempo

        packet = 0

        def ready(pcm):
            nonlocal packet
            check_cancelled(cancel_event)
            packet += 1
            if profile:
                profile("chunk_pcm_ready", chunk=packet, packet=packet)
            if on_audio is not None:
                on_audio(pcm)

        def mark(stage, **details):
            if profile and stage != "chunk_pcm_ready":
                profile(stage, **details)

        async with StreamingTempo(self.ffmpeg, self.args.minicpmo_speed, self.output_sample_rate, ready) as tempo:
            reply = await super().read_reply(socket, speech, cancel_event, tempo.feed, mark)
            pcm = await tempo.finish()
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
        return VoiceReply(reply.text, torch.from_numpy(samples).unsqueeze(0))

    def chat_messages(self, messages, cancel_event):
        # The C++ runtime reads audio from the final user message and truncates
        # text per message when there are multiple text turns. Keep the FAQ intact.
        return super().chat_messages(single_audio_messages(messages), cancel_event)

    async def open_session(self, cancel_event):
        from websockets.exceptions import ConnectionClosedOK
        from websockets.legacy.client import connect

        deadline = asyncio.get_running_loop().time() + self.session_init_timeout
        retried = False
        while True:
            check_cancelled(cancel_event)
            socket = await connect(self.websocket_url, open_timeout=10, close_timeout=1,
                                   max_size=16 * 1024 * 1024)
            try:
                payload = {"mode": "turn_based"}
                if self.reference_audio is not None:
                    payload["voice"] = {"ref_audio": self.reference_audio}
                await socket.send(json.dumps({"type": "session.init", "payload": payload}))
                session = await self.receive_event(socket)
                session_id = session.get("session_id")
                if session["type"] != "session.created" or not isinstance(session_id, str) or not session_id:
                    raise RuntimeError("llama-omni-server 未返回有效的会话。")
                return socket, session_id
            except ConnectionClosedOK as error:
                await socket.close()
                check_cancelled(cancel_event)
                # The native server closes new sockets while an old session is
                # still active. No input has been sent, so retrying is safe.
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise RuntimeError(
                        "llama-omni-server 创建会话前关闭了连接，等待重试后仍未恢复。"
                        "请检查推理服务日志中是否有 active session exists，以及上一会话是否已释放。"
                    ) from error
                if not retried:
                    LOG.info("llama-omni-server closed during session initialization; retrying before input")
                    retried = True
                await asyncio.sleep(min(0.2, remaining))
            except BaseException:
                await socket.close()
                raise

    async def chat(self, payload, speech, cancel_event, on_audio, profile):
        payload = {**payload, "streaming": self.args.minicpmo_stream}
        socket, session_id = await self.open_session(cancel_event)
        reason = "client_closed"
        try:
            check_cancelled(cancel_event)
            await socket.send(json.dumps({"type": "input.append", "input": payload}, ensure_ascii=False))
            reply = await self.read_reply(socket, speech, cancel_event, on_audio, profile)
            if speech and reply.waveform is None:
                raise RuntimeError("llama-omni-server 未返回语音，请检查 Token2Wav 日志。")
            reason = "turn_done"
            return reply
        finally:
            # HTTP close stops native generation; transport closure alone may
            # finish before the inference threads have released the session.
            try:
                async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
                    url = f"{self.base_url}/sessions/{quote(session_id, safe='')}/close"
                    response = await client.post(url, json={"reason": reason})
                    if response.status_code != 404:
                        response.raise_for_status()
            except httpx.HTTPError as error:
                LOG.warning("llama-omni-server session %s cleanup failed (%s): %s",
                            session_id, type(error).__name__, error)
            finally:
                await socket.close()
