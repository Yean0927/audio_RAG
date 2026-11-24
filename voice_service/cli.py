"""Shared backend selection and arguments for the web and call entry points."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from urllib.parse import urlsplit

from . import ROOT
from .backend import API_BACKENDS, BACKENDS
from .glm import GLM_TOKENIZER_REPO, VoiceBackend


def create_backend(args):
    name = getattr(args, "backend", "glm")
    if name == "glm":
        return VoiceBackend(args)
    if name in API_BACKENDS:
        from .api import (
            MingAPIBackend,
            MiniCPMOAPIBackend,
            MiniCPMOGGUFBackend,
            MiniCPMOVLLMBackend,
            QwenOmniAPIBackend,
        )

        return {"ming": MingAPIBackend, "omni": QwenOmniAPIBackend, "minicpmo": MiniCPMOAPIBackend,
                "minicpmo-local": MiniCPMOVLLMBackend, "minicpmo-gguf": MiniCPMOGGUFBackend}[name](args)
    if name == "omni-local":
        from .audio_models import QwenOmniBackend
        return QwenOmniBackend(args)
    raise KeyError(name)


def create_parser(description=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--backend", choices=BACKENDS, default="glm", help="语音模型后端")
    parser.add_argument("--model-path", help="模型目录、Hugging Face ID 或 MiniCPM-o vLLM-Omni 服务的模型名")
    parser.add_argument("--tokenizer-path", help="GLM 独立语音分词器目录或 Hugging Face ID")
    parser.add_argument("--flow-path", help="GLM 独立语音 Decoder 目录")
    parser.add_argument("--flow-steps", type=int,
                        help="语音 Flow 的 Euler 采样步数（4–10，默认 10）")
    parser.add_argument("--quantization", choices=["4bit", "bf16"], help="本地模型量化，默认 GLM 4bit，本地 Omni bf16")
    parser.add_argument("--vllm-base-url", help="GLM vLLM API 地址，例如 http://127.0.0.1:18000/v1")
    parser.add_argument("--vllm-model", help="vLLM 的 --served-model-name，默认 glm-voice")
    parser.add_argument("--minicpmo-base-url", help="MiniCPM-o vLLM-Omni API 地址，默认 http://127.0.0.1:8099/v1")
    parser.add_argument("--minicpmo-gguf-base-url", help="llama-omni-server 地址，默认 http://127.0.0.1:8006")
    parser.add_argument("--minicpmo-ref-audio", type=Path, help="GGUF 音色参考 WAV（通话主机上的路径）")
    parser.add_argument("--minicpmo-speed", type=float, help="GGUF 输出语速倍率，0.5–2，默认 1；变速需要通话主机安装 ffmpeg")
    parser.add_argument("--minicpmo-stream", action=argparse.BooleanOptionalAction, default=None,
                        help="GGUF 流式输出，默认开启；--no-minicpmo-stream 等整轮生成后出音")
    parser.add_argument("--omni-speaker", choices=["Chelsie", "Ethan"], default="Chelsie", help="本地 Qwen2.5-Omni 音色")
    parser.add_argument("--attn-implementation", choices=["auto", "eager", "sdpa", "flash_attention_2"],
                        default="auto", help="GLM 注意力实现；auto 保持 Transformers 默认选择")
    parser.add_argument("--llm-device", default="cuda:0")
    parser.add_argument("--speech-device", help="语音组件与 SONAR 设备，API 后端默认使用 CPU 检索")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--history-turns", type=int, default=3,
                        help="回答时携带的已完成历史轮数（0–10，默认 3；0 关闭上下文）")
    parser.add_argument("--max-audio-seconds", type=int, default=30)
    parser.add_argument("--retrieval-metric", choices=("euclidean", "cosine"), default="euclidean",
                        help="SONAR 检索使用平方欧氏距离或余弦相似度（默认 euclidean）")
    parser.add_argument("--compare-retrieval", action="store_true",
                        help="对同一轮录音记录两种检索排序，回答使用 --retrieval-metric 指定的结果")
    parser.add_argument("--output-dir", default=str(ROOT / "answer_data/web"))
    return parser


def validate_endpoint(url, option, example, parser):
    try:
        address = urlsplit(url)
        valid = (address.scheme in ("http", "https") and bool(address.hostname)
                 and not address.query and not address.fragment)
    except ValueError:
        valid = False
    if not valid:
        parser.error(f"{option} 需要 HTTP(S) API 地址，例如 {example}")


def validate_args(args, parser):
    api_backend = args.backend in API_BACKENDS
    if args.vllm_base_url is not None:
        validate_endpoint(args.vllm_base_url, "--vllm-base-url", "http://127.0.0.1:18000/v1", parser)
        if args.backend != "glm":
            parser.error("--vllm-base-url 仅适用于 --backend glm")
        if args.attn_implementation != "auto":
            parser.error("使用 vLLM 时请在 vLLM 服务端设置注意力后端")
        args.vllm_model = args.vllm_model or "glm-voice"
    elif args.vllm_model is not None:
        parser.error("--vllm-model 需要同时指定 --vllm-base-url")
    if args.minicpmo_base_url is not None:
        validate_endpoint(args.minicpmo_base_url, "--minicpmo-base-url", "http://127.0.0.1:8099/v1", parser)
        if args.backend != "minicpmo-local":
            parser.error("--minicpmo-base-url 仅适用于 --backend minicpmo-local")
    elif args.backend == "minicpmo-local":
        args.minicpmo_base_url = "http://127.0.0.1:8099/v1"
    if args.minicpmo_gguf_base_url is not None:
        validate_endpoint(args.minicpmo_gguf_base_url, "--minicpmo-gguf-base-url", "http://127.0.0.1:8006", parser)
        if args.backend != "minicpmo-gguf":
            parser.error("--minicpmo-gguf-base-url 仅适用于 --backend minicpmo-gguf")
    elif args.backend == "minicpmo-gguf":
        args.minicpmo_gguf_base_url = "http://127.0.0.1:8006"
    if any(value is not None for value in (args.minicpmo_ref_audio, args.minicpmo_speed, args.minicpmo_stream)):
        if args.backend != "minicpmo-gguf":
            parser.error("--minicpmo-ref-audio / --minicpmo-speed / --[no-]minicpmo-stream 仅适用于 minicpmo-gguf")
    if args.minicpmo_ref_audio is not None:
        args.minicpmo_ref_audio = args.minicpmo_ref_audio.expanduser().resolve()
        if not args.minicpmo_ref_audio.is_file():
            parser.error(f"找不到音色参考 WAV：{args.minicpmo_ref_audio}")
    args.minicpmo_speed = 1.0 if args.minicpmo_speed is None else args.minicpmo_speed
    if not math.isfinite(args.minicpmo_speed) or not 0.5 <= args.minicpmo_speed <= 2:
        parser.error("--minicpmo-speed 必须在 0.5–2 之间")
    args.minicpmo_stream = True if args.minicpmo_stream is None else args.minicpmo_stream
    if args.backend != "glm":
        if args.tokenizer_path or args.flow_path or args.flow_steps is not None:
            parser.error("--tokenizer-path / --flow-path / --flow-steps 仅适用于 GLM")
    if api_backend:
        if args.backend != "minicpmo-local" and args.model_path and args.model_path != BACKENDS[args.backend][1]:
            parser.error("API 后端使用固定模型 ID。本地 Qwen2.5-Omni 请选 --backend omni-local。")
        if args.quantization or args.attn_implementation != "auto":
            parser.error("API 后端的量化和注意力实现由模型服务端设置。")
    elif args.backend == "omni-local" and args.quantization == "4bit":
        parser.error("本地 Omni 当前仅支持 --quantization bf16")
    args.model_path = args.model_path or BACKENDS[args.backend][1]
    args.tokenizer_path = args.tokenizer_path or GLM_TOKENIZER_REPO
    args.flow_path = args.flow_path or str(ROOT / "glm-4-voice-decoder")
    args.flow_steps = 10 if args.flow_steps is None else args.flow_steps
    if not api_backend:
        args.quantization = args.quantization or ("4bit" if args.backend == "glm" else "bf16")
    if not args.speech_device:
        args.speech_device = "cpu" if api_backend else {"glm": "cuda:1"}.get(args.backend, args.llm_device)
    if not 4 <= args.flow_steps <= 10:
        parser.error("--flow-steps must be 4–10")
    if not 64 <= args.max_new_tokens <= 2048 or not 0 <= args.history_turns <= 10:
        parser.error("--max-new-tokens must be 64–2048; --history-turns must be 0–10")
    if not 1 <= args.max_audio_seconds <= 60:
        parser.error("--max-audio-seconds must be 1–60")
    return args
