"""Stress-test streaming GLM NF4 turns with one shared backend on one CUDA GPU."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
import csv
from functools import partial
import json
import logging
import math
import os
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import threading
import time


from voice_service import ROOT
MIB = 1024**2
AUDIO_SUFFIXES = {".wav", ".wave", ".flac", ".mp3", ".ogg", ".opus", ".aif", ".aiff", ".caf"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0, help="当前进程可见的 CUDA 序号")
    parser.add_argument("--concurrency", type=int, default=5, help="每轮同时发起的通话数")
    parser.add_argument("--turns", type=int, default=3, help="每路问答轮数，保留各自的历史")
    audio = parser.add_mutually_exclusive_group()
    audio.add_argument("--wav", action="append", type=Path, help="客户录音路径，可重复指定并轮换使用")
    audio.add_argument("--wav-dir", type=Path, help="随机打乱目录中的录音，全部使用一遍后循环")
    parser.add_argument("--audio-seed", type=int, help="复现 --wav-dir 的选样顺序")
    parser.add_argument("--config", type=Path, default=ROOT / "call_config.json")
    parser.add_argument("--model-path")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--flow-path")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--history-turns", type=int)
    parser.add_argument("--flow-steps", type=int)
    parser.add_argument("--attn-implementation", choices=("auto", "eager", "sdpa", "flash_attention_2"))
    parser.add_argument("--sample-ms", type=int, default=100, help="整卡用量采样间隔")
    parser.add_argument("--memory-csv", type=Path, help="保存整卡用量、PyTorch 活跃张量和预留显存的时间线")
    parser.add_argument("--trace-kv", action="store_true", help="记录每轮前两次 GLM 前向和 KV 张量峰值")
    parser.add_argument("--download-models", action="store_true", help="允许下载缺失的模型权重")
    parser.add_argument("--json", type=Path, default=ROOT / "answer_data/calls/nf4-concurrency.json")
    args = parser.parse_args(argv)
    if args.gpu < 0 or min(args.concurrency, args.turns, args.sample_ms) < 1:
        parser.error("--gpu 不能为负数，并发数、问答轮数和采样间隔必须大于零。")
    if args.memory_csv is not None and args.memory_csv.resolve() == args.json.resolve():
        parser.error("--memory-csv 和 --json 需要使用不同的输出文件。")
    if args.audio_seed is not None and args.wav_dir is None:
        parser.error("--audio-seed 需要与 --wav-dir 一起使用。")
    return args


def test_recordings(args, report):
    if args.wav_dir is None:
        return args.wav or [ROOT / "speech_data/simple/question.wav"]

    directory = args.wav_dir.resolve()
    if not directory.is_dir():
        raise ValueError(f"录音目录不存在：{directory}")
    recordings = sorted(path for path in directory.rglob("*")
                        if path.suffix.lower() in AUDIO_SUFFIXES and path.is_file()
                        and not any(part.startswith(".") for part in path.relative_to(directory).parts))
    if not recordings:
        raise ValueError(f"录音目录中没有支持的音频文件：{directory}")
    seed = args.audio_seed if args.audio_seed is not None else random.SystemRandom().getrandbits(32)
    random.Random(seed).shuffle(recordings)
    report.update(audio_dir=str(directory), audio_seed=seed)
    print(f"随机测试录音 {len(recordings)} 条，种子 {seed}", flush=True)
    return recordings


@contextmanager
def monitor_memory(cuda, device, name, report, sample_ms, timeline=None):
    """Capture CUDA allocator high-water marks and sample device-wide usage."""
    cuda.reset_peak_memory_stats(device)
    free, total = cuda.mem_get_info(device)
    peak_used = total - free
    samples = 1
    started = time.perf_counter()

    def allocator_usage():
        stats = cuda.memory_stats(device)
        return stats["allocated_bytes.all.current"] / MIB, stats["reserved_bytes.all.current"] / MIB

    def write_sample(point, used, allocated, reserved):
        if timeline is not None:
            timeline.writerow((name, point, round((time.perf_counter() - started) * 1000, 3),
                               used / MIB, allocated, reserved))

    allocated_start, reserved_start = allocator_usage()
    row = {"stage": name, "gpu_total_mib": total / MIB, "gpu_used_start_mib": peak_used / MIB,
           "torch_allocated_start_mib": allocated_start, "torch_reserved_start_mib": reserved_start}
    report["memory"].append(row)
    stop = threading.Event()
    write_sample("start", peak_used, allocated_start, reserved_start)

    def sample():
        nonlocal peak_used, samples
        free, total = cuda.mem_get_info(device)
        used = total - free
        peak_used = max(peak_used, used)
        samples += 1
        return used

    def watch():
        next_log = started + 5
        try:
            while not stop.wait(sample_ms / 1000):
                used = sample()
                if timeline is not None:
                    write_sample("sample", used, *allocator_usage())
                now = time.perf_counter()
                if now >= next_log:
                    print(f"[{name}] 整卡 {used / MIB:.0f}/{total / MIB:.0f} MiB，"
                          f"采样峰值 {peak_used / MIB:.0f} MiB", flush=True)
                    next_log = now + 5
        except Exception as error:
            row["sampling_error"] = str(error)
            logging.exception("整卡显存采样中断")

    watcher = threading.Thread(target=watch, name="gpu-memory-sampler", daemon=True)
    print(f"[{name}] 开始，整卡已用 {peak_used / MIB:.0f}/{total / MIB:.0f} MiB", flush=True)
    watcher.start()
    try:
        yield
        cuda.synchronize(device)
    finally:
        stop.set()
        watcher.join()
        row.update(elapsed_s=round(time.perf_counter() - started, 3))
        # Keep the samples and allocator counters available after a failed stage, including OOM.
        try:
            used_end = sample()
            row["gpu_used_end_mib"] = used_end / MIB
        except Exception as error:
            used_end = None
            row["sampling_error"] = str(error)
        row.update(gpu_used_sampled_peak_mib=peak_used / MIB, gpu_samples=samples)
        try:
            row["torch_allocated_end_mib"], row["torch_reserved_end_mib"] = allocator_usage()
            row["torch_peak_allocated_mib"] = cuda.max_memory_allocated(device) / MIB
            row["torch_peak_reserved_mib"] = cuda.max_memory_reserved(device) / MIB
            row["torch_peak_allocated_delta_mib"] = row["torch_peak_allocated_mib"] - allocated_start
        except Exception as error:
            row["allocator_error"] = str(error)
        if used_end is not None and "torch_allocated_end_mib" in row:
            try:
                write_sample("end", used_end, row["torch_allocated_end_mib"], row["torch_reserved_end_mib"])
            except Exception as error:
                row["sampling_error"] = str(error)
        print(f"[{name}] 整卡采样峰值 {row['gpu_used_sampled_peak_mib']:.0f} MiB，"
              f"PyTorch 分配/预留峰值 {row.get('torch_peak_allocated_mib', '-')} / "
              f"{row.get('torch_peak_reserved_mib', '-')} MiB", flush=True)


def failure(error):
    message = str(error)
    return {"error_type": type(error).__name__, "error": message or type(error).__name__,
            "oom": type(error).__name__ == "OutOfMemoryError" or "out of memory" in message.lower()}


@contextmanager
def trace_glm_kv(model):
    # Each streaming generate() starts and ends in its own worker thread.
    active = threading.local()

    def capture(module, args, kwargs):
        row = getattr(active, "row", None)
        if row is None or len(row["glm_kv_forwards"]) >= 2:
            return
        input_ids = args[0] if args else kwargs.get("input_ids")
        past = kwargs.get("past_key_values")
        past_tokens = None
        if past is None:
            past_tokens = 0
        elif isinstance(past, (tuple, list)):
            if not past:
                past_tokens = 0
            elif isinstance(past[0], (tuple, list)) and past[0] and past[0][0] is not None:
                # GLM's layer key is shaped [batch, KV heads, sequence, head dim].
                past_tokens = past[0][0].shape[-2]
        use_cache = kwargs.get("use_cache")
        if use_cache is None:
            use_cache = getattr(module.config, "use_cache", None)
        row["glm_kv_forwards"].append({
            "input_tokens": input_ids.shape[-1] if input_ids is not None else None,
            "past_kv_tokens": past_tokens, "past_kv_type": type(past).__name__, "use_cache": use_cache,
        })

    def capture_cache_size(_module, _args, output):
        row = getattr(active, "row", None)
        if row is None:
            return
        cache = getattr(output, "past_key_values", None)
        if not isinstance(cache, (tuple, list)) or not cache:
            return
        size = 0
        for layer in cache:
            if not isinstance(layer, (tuple, list)) or len(layer) != 2:
                return
            size += sum(tensor.numel() * tensor.element_size() for tensor in layer if tensor is not None)
        row["glm_peak_kv_tensor_mib"] = max(row.get("glm_peak_kv_tensor_mib", 0), size / MIB)

    with ExitStack() as hooks:
        hooks.callback(model.register_forward_pre_hook(capture, with_kwargs=True).remove)
        hooks.callback(model.register_forward_hook(capture_cache_size).remove)
        yield active


def measure_turn(session, wav, history, *, backend, cuda, device, knowledge, language,
                 profile_type, block_sizes, run_start, gate=None, kv_trace=None):
    first_audio = None
    chunks = 0

    def on_audio(pcm):
        nonlocal first_audio, chunks
        if pcm:
            if first_audio is None:
                first_audio = time.perf_counter()
            chunks += 1

    row = {"session": session, "wav": str(wav), "history_turns": len(history)}
    if kv_trace is not None:
        row["glm_kv_forwards"] = []
    updated = history
    profile = None

    def mark_with_kv(stage, **details):
        profile.mark(stage, **details)
        if stage == "generate_start":
            kv_trace.row = row
        elif stage == "generate_end":
            kv_trace.row = None

    try:
        with cuda.device(device):
            if gate is not None:
                gate.wait(timeout=30)
            profile = profile_type(str(session), block_sizes)
            _, audio, _, _, updated = backend.answer(
                str(wav), True, knowledge, None, language, history, on_audio=on_audio,
                profile=mark_with_kv if kv_trace is not None else profile.mark,
            )
            if not audio:
                raise RuntimeError("未生成可播放的语音回复。")
            if not updated[-1].get("retrieval"):
                raise RuntimeError("本轮未执行 SONAR 检索。")
    except Exception as error:
        row.update(failure(error))
    ended = time.perf_counter()
    row.update(ok="error" not in row, audio_chunks=chunks)
    if profile is not None:
        start = profile.started - run_start
        row.update(start_s=start, end_s=ended - run_start, reply_ms=round((ended - profile.started) * 1000, 1))
        if first_audio is not None or row["ok"]:
            audible = first_audio if first_audio is not None else ended
            row["first_audio_ms"] = round((audible - profile.started) * 1000, 1)
            row["first_audio_source"] = "stream" if first_audio is not None else "completed_wav"
        for event in profile.snapshot()["events"]:
            if event["stage"] in ("generate_start", "generate_end"):
                row[event["stage"] + "_s"] = start + event["ms"] / 1000
            elif kv_trace is not None and event["stage"] == "prompt_ready":
                row["glm_prompt_tokens"] = event["tokens"]
    return row, updated


def max_overlap(rows, start_key, end_key):
    points = []
    for row in rows:
        start, end = row.get(start_key), row.get(end_key)
        if start is not None and end is not None and start < end:
            points.extend(((start, 1), (end, -1)))
    active = peak = 0
    for _, delta in sorted(points):
        active += delta
        peak = max(peak, active)
    return peak


def check_model_placement(backend, device):
    config = backend.model.config.quantization_config
    if not backend.model.is_loaded_in_4bit or config.bnb_4bit_quant_type != "nf4":
        raise RuntimeError("GLM 未按 NF4 量化加载。")
    modules = {
        "GLM": backend.model, "Whisper": backend.whisper, "Flow": backend.decoder.flow, "HiFT": backend.decoder.hift,
        "SONAR speech": backend.embedding.client_speech.model, "SONAR text": backend.embedding.client_text.model,
    }
    placement = {}
    for name, module in modules.items():
        devices = sorted({str(param.device) for param in module.parameters()})
        if devices != [str(device)]:
            raise RuntimeError(f"{name} 参数位于 {devices}，预期全部在 {device}。")
        placement[name] = devices[0]
    return placement


def run(args, report):
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("请在已安装服务依赖的 NVIDIA CUDA 服务器上使用 .venv/bin/python 运行。") from error
    cuda = torch.cuda
    if not cuda.is_available() or args.gpu >= cuda.device_count():
        raise RuntimeError(f"当前 PyTorch 进程无法使用 cuda:{args.gpu}。")

    import soundfile as sf
    from voice_call_server import CallLatencyProfile, load_config, parse_args as parse_call_args, prepare_decoder
    from voice_service.backend import STREAM_BLOCK_SIZES
    from voice_service.cli import create_backend

    device = torch.device(f"cuda:{args.gpu}")
    cuda.set_device(device)
    report.update(device=str(device), gpu_name=cuda.get_device_name(device), torch_version=torch.__version__,
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    flags = ["--quantization", "4bit", "--llm-device", str(device), "--speech-device", str(device)]
    for name in ("model_path", "tokenizer_path", "flow_path", "max_new_tokens", "history_turns", "flow_steps",
                 "attn_implementation"):
        value = getattr(args, name)
        if value is not None:
            flags.extend(("--" + name.replace("_", "-"), str(value)))
    if args.download_models:
        flags.append("--download-models")
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            os.environ[key] = "0"
    options = parse_call_args(flags)
    report.update(model=options.model_path, max_new_tokens=options.max_new_tokens,
                  prompt_history_turns=options.history_turns, flow_steps=options.flow_steps)
    if not args.config.is_file():
        raise ValueError(f"找不到客服配置：{args.config}")
    config = load_config(args.config)
    knowledge = config["knowledge_text"]
    if not knowledge:
        raise ValueError("客服配置须包含有内容的 knowledge_file，才能测量 SONAR 全流程。")
    options.system_prompt = config.get("system_prompt", "")
    language = config.get("language", "zh")
    report["language"] = language
    recordings = test_recordings(args, report)
    for wav in recordings:
        info = sf.info(wav)
        if not 0 < info.duration <= options.max_audio_seconds:
            raise ValueError(f"录音 {wav} 应为 0 到 {options.max_audio_seconds} 秒的有效语音。")
        report["inputs"].append({"wav": str(wav), "audio_s": info.duration})

    with TemporaryDirectory(prefix="glm-nf4-benchmark-") as output_dir, ExitStack() as cleanup:
        timeline = None
        if args.memory_csv is not None:
            args.memory_csv.parent.mkdir(parents=True, exist_ok=True)
            stream = cleanup.enter_context(args.memory_csv.open("w", encoding="utf-8", newline=""))
            timeline = csv.writer(stream)
            timeline.writerow(("stage", "point", "stage_elapsed_ms", "gpu_used_mib",
                               "torch_allocated_mib", "torch_reserved_mib"))
            report["memory_csv"] = str(args.memory_csv.resolve())
        monitor = partial(monitor_memory, cuda, device, report=report, sample_ms=args.sample_ms, timeline=timeline)
        options.output_dir = output_dir
        with monitor("load_models"):
            prepare_decoder(options)
            backend = create_backend(options)
            backend.load()
        with monitor("load_sonar_and_index"):
            backend.prepare_knowledge(knowledge, language)
            report["model_devices"] = check_model_placement(backend, device)
        kv_trace = cleanup.enter_context(trace_glm_kv(backend.model)) if args.trace_kv else None
        measure = partial(measure_turn, backend=backend, cuda=cuda, device=device, knowledge=knowledge,
                          language=language, profile_type=CallLatencyProfile, block_sizes=STREAM_BLOCK_SIZES,
                          run_start=time.perf_counter(), kv_trace=kv_trace)
        with monitor("warmup"):
            report["warmup"], _ = measure(0, recordings[0], [])
        if not report["warmup"]["ok"]:
            report.update({key: report["warmup"][key] for key in ("error_type", "error", "oom")})
            logging.error("单路预热失败：%s", report["error"])
            return

        histories = [[] for _ in range(args.concurrency)]
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            for turn in range(args.turns):
                gate = threading.Barrier(args.concurrency)
                audio_offset = turn * args.concurrency if args.wav_dir is not None else turn
                with monitor(f"concurrent_turn_{turn + 1}"):
                    futures = [executor.submit(measure, index + 1, recordings[(audio_offset + index) % len(recordings)],
                                               history, gate=gate)
                               for index, history in enumerate(histories)]
                    results, histories = map(list, zip(*(future.result() for future in futures)))
                end_to_end = max_overlap(results, "start_s", "end_s")
                generating = max_overlap(results, "generate_start_s", "generate_end_s")
                report["waves"].append({"turn": turn + 1, "max_concurrent_turns": end_to_end,
                                        "max_concurrent_glm_generations": generating, "results": results})
                failures = [row for row in results if not row["ok"]]
                print(f"第 {turn + 1} 轮完成 {len(results) - len(failures)}/{len(results)} 路，"
                      f"问答重叠 {end_to_end} 路，GLM 生成重叠 {generating} 路", flush=True)
                if failures:
                    for row in failures:
                        print(f"  第 {row['session']} 路失败：{row['error']}", flush=True)
                    break


def summarize(report):
    waves = report["waves"]
    rows = [row for wave in waves for row in wave["results"]]
    successful = [row for row in rows if row["ok"]]
    report["oom"] = report.get("oom", False) or any(row.get("oom", False) for row in rows)
    report["passed"] = (len(waves) == report["turns"] and len(successful) == report["expected_turns"]
                        and all(wave["max_concurrent_turns"] == report["concurrency"] for wave in waves)
                        and not report.get("error")
                        and all(not (stage.get("sampling_error") or stage.get("allocator_error"))
                                for stage in report["memory"]))
    if successful:
        for key in ("first_audio_ms", "reply_ms"):
            values = sorted(row[key] for row in successful)
            report[f"p95_{key}"] = values[math.ceil(0.95 * len(values)) - 1]
        print(f"P95 首段语音 {report['p95_first_audio_ms']:.0f} ms，"
              f"完整回复 {report['p95_reply_ms']:.0f} ms", flush=True)
    for key in ("gpu_used_sampled_peak_mib", "torch_peak_allocated_mib", "torch_peak_reserved_mib"):
        values = [stage[key] for stage in report["memory"] if key in stage]
        if values:
            report[key] = max(values)
    if report["memory"]:
        total = report["memory"][0]["gpu_total_mib"]
        peak = report["gpu_used_sampled_peak_mib"]
        report["gpu_sampled_min_free_mib"] = total - peak
        print(f"全程整卡采样峰值 {peak:.0f}/{total:.0f} MiB，最少剩余 {total - peak:.0f} MiB", flush=True)
    print(f"完整语音回复 {len(successful)}/{report['expected_turns']}，"
          f"OOM {'有' if report['oom'] else '无'}，容量测试{'通过' if report['passed'] else '未通过'}", flush=True)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = {"quantization": "nf4", "concurrency": args.concurrency, "turns": args.turns,
              "expected_turns": args.concurrency * args.turns, "sample_ms": args.sample_ms,
              "memory": [], "waves": [], "inputs": []}
    try:
        run(args, report)
    except Exception as error:
        report.update(failure(error))
        logging.error("压测中断：%s", report["error"])
    except KeyboardInterrupt:
        report["error"] = "压测被中断。"
    summarize(report)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"报告 {args.json.resolve()}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
