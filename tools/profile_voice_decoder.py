"""Replay a saved reply through the streaming decoder and inspect CUDA activity."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, nullcontext
import json
from pathlib import Path
import re
from statistics import median
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace

from voice_service import ROOT
from voice_service.backend import STREAM_BLOCK_SIZES
from voice_service.glm import VoiceBackend


AUDIO_TOKEN = re.compile(r"<\|audio_(\d+)\|>")
CHUNK_MARKER = "voice_decode_chunk_"


def read_reply(path):
    files = [path] if path else sorted(
        (ROOT / "answer_data").glob("*/*/turn.json"),
        key=lambda item: item.stat().st_mtime, reverse=True,
    )
    for file in files:
        reply = json.loads(file.read_text(encoding="utf-8"))
        tokens = [int(value) for value in AUDIO_TOKEN.findall(reply.get("assistant_tokens", ""))]
        if len(tokens) >= STREAM_BLOCK_SIZES[0]:
            return file, tokens
    raise ValueError("找不到至少含 25 个回复语音 token 的 turn.json。请先完成一次通话，或用 --turn 指定已保存的回复。")


def replay(backend, tokens, traced=False):
    import torch

    chunks = {}
    with ExitStack() as markers:
        def mark(stage, *, at=None, **fields):
            if stage == "chunk_tokens_ready":
                chunks[fields["chunk"]] = {"tokens": fields["tokens"], "final": fields["final"]}
            elif stage == "chunk_decode_start":
                number = fields["chunk"]
                if traced:
                    markers.enter_context(torch.profiler.record_function(f"{CHUNK_MARKER}{number}"))
                chunks[number]["started"] = time.perf_counter()
            elif stage == "chunk_decoded":
                chunk = chunks[fields["chunk"]]
                chunk["ms"] = (time.perf_counter() - chunk["started"]) * 1000
                if traced:
                    markers.pop_all().close()

        def generate(streamer, stop_stream):
            streamer.put(tokens[0])  # The streamer skips its first put (the prompt).
            for token in tokens:
                if stop_stream.is_set():
                    break
                streamer.put(token)

        with torch.inference_mode():
            VoiceBackend.generate_audio_stream(backend, generate, 0, None, lambda pcm: None, mark)
    return chunks


def event_time(events, start, end):
    """Milliseconds covered by the union of the events, clipped to the chunk."""
    intervals = sorted((max(start, event["ts"]), min(end, event["ts"] + event["dur"]))
                       for event in events)
    cursor, total = start, 0
    for left, right in intervals:
        left = max(left, cursor)
        if right > left:
            total += right - left
            cursor = right
    return total / 1000


def report_trace(trace, baseline):
    events = [event for event in trace["traceEvents"] if event.get("ph") == "X" and "dur" in event]
    kernels = [event for event in events if event.get("cat") == "kernel"]
    if not kernels:
        raise RuntimeError("CUDA trace 里没有 GPU kernel 记录，无法判断 GPU 忙闲。请检查 PyTorch Profiler/CUPTI 是否可用。")
    copies = [event for event in events if event.get("cat") == "gpu_memcpy"]
    memsets = [event for event in events if event.get("cat") == "gpu_memset"]
    runtime = [event for event in events if event.get("cat") in ("cuda_runtime", "cuda_driver")]

    print("块  token    基线  跟踪墙钟  GPU kernel   GPU拷贝  无GPU活动  CPU同步/拷贝API  CPU提交kernel（ms）")
    for marker in events:
        if not marker.get("name", "").startswith(CHUNK_MARKER):
            continue
        number = int(marker["name"].removeprefix(CHUNK_MARKER))
        start = marker["ts"]
        end = start + marker["dur"]
        calls = [event for event in runtime if event["pid"] == marker["pid"] and event["tid"] == marker["tid"]]
        sync = [event for event in calls if "Synchronize" in event["name"]
                or ("Memcpy" in event["name"] and "Async" not in event["name"])]
        launches = [event for event in calls if event["name"].startswith(("cudaLaunch", "cuLaunch"))]
        ms = marker["dur"] / 1000
        busy = event_time(kernels + copies + memsets, start, end)
        print(f"{number:2} {baseline[number]['tokens']:6} {baseline[number]['ms']:8.1f} {ms:9.1f} "
              f"{event_time(kernels, start, end):11.1f} {event_time(copies, start, end):9.1f} "
              f"{ms - busy:10.1f} {event_time(sync, start, end):16.1f} {event_time(launches, start, end):14.1f}")
    print("CPU 等待与 GPU 运行可以重叠，表中各列不能直接相加。无 GPU 活动的时间也包含调度空隙，需结合 trace 定位。")
    print("跟踪会带来额外开销。用基线比较真实延迟，用跟踪列判断时间花在哪里。")


def compare_cfg(backend, tokens, device, warmup, runs):
    import torch

    flow = backend.decoder.flow.decoder
    if flow.inference_cfg_rate <= 0:
        raise ValueError("当前解码器没有启用 CFG，无法对照。")
    results = {False: [], True: []}
    for batch in results:
        flow.batch_cfg = batch
        for _ in range(warmup):
            replay(backend, tokens)
    for trial in range(runs):
        # Alternate the order so both modes run first in some trials.
        for batch in (False, True) if trial % 2 == 0 else (True, False):
            flow.batch_cfg = batch
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            allocated = torch.cuda.memory_allocated(device)
            chunks = replay(backend, tokens)
            peak_extra = (torch.cuda.max_memory_allocated(device) - allocated) / 1024**2
            results[batch].append((chunks, peak_extra))

    print("CFG 模式  首块中位数(ms)  解码总耗时中位数(ms)  峰值新增显存(MiB)")
    for batch, measurements in results.items():
        first = median(chunks[1]["ms"] for chunks, _ in measurements)
        total = median(sum(chunk["ms"] for chunk in chunks.values()) for chunks, _ in measurements)
        peak = max(memory for _, memory in measurements)
        print(f"{'合批' if batch else '串行':<7} {first:14.1f} {total:20.1f} {peak:18.1f}")
    print("两种模式回放相同 token。解码耗时含 PCM 回传，生成、网络和播放未参与计时。")


def compare_steps(decoder, tokens, device, warmup, runs):
    import torch

    flow = decoder.flow
    if not hasattr(flow, "inference_steps"):
        raise RuntimeError("当前 Flow 版本不支持设置采样步数，请同步 src/cosyvoice/flow/flow.py。")
    first = torch.stack(tokens[:STREAM_BLOCK_SIZES[0]]).unsqueeze(0).to(device)
    lengths = torch.tensor([first.size(1), 0], dtype=torch.int32, device=device)
    prompt_mel = torch.empty(1, 0, flow.output_size, device=device)
    speaker = torch.zeros(1, flow.spk_embed_affine_layer.in_features, device=device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def measure(steps):
        flow.inference_steps = steps
        start.record()
        flow.inference(first, lengths[:1], first[:, :0], lengths[1:], prompt_mel, lengths[1:], speaker)
        end.record()
        end.synchronize()
        return start.elapsed_time(end)

    results = {steps: [] for steps in (10, 8, 6)}
    for steps in results:
        for _ in range(warmup):
            measure(steps)
    order = tuple(results)
    for trial in range(runs):
        offset = trial % len(order)
        for steps in order[offset:] + order[:offset]:
            results[steps].append(measure(steps))

    baseline = median(results[10])
    print(f"固定首块 {first.size(1)} 个语音 token，仅计 Flow（token → Mel）")
    print("采样步数  Flow 中位数(ms)   较 10 步节省(ms)")
    for steps, samples in results.items():
        elapsed = median(samples)
        print(f"{steps:8} {elapsed:16.1f} {baseline - elapsed:18.1f}")


def run(args):
    import torch
    from spoken_chatbot.model import AudioDecoder

    reply, ids = read_reply(args.turn)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("请在有 NVIDIA GPU 的服务器上运行，并指定 CUDA 设备。")
    config, flow, hift = (args.flow_path / name for name in ("config.yaml", "flow.pt", "hift.pt"))
    print(f"回复：{reply}\n语音 token：{len(ids)}，分块：{STREAM_BLOCK_SIZES}，设备：{device}", flush=True)
    with torch.cuda.device(device):
        decoder = AudioDecoder(str(config), str(flow), str(hift), device=device)
        decoder.flow.eval()
        decoder.hift.eval()
        backend = SimpleNamespace(decoder=decoder, speech_device=device)
        tokens = torch.tensor(ids, dtype=torch.long, device="cpu").unbind()
        if args.compare_steps:
            compare_steps(decoder, tokens, device, args.warmup, args.runs)
            return
        if args.compare_cfg:
            compare_cfg(backend, tokens, device, args.warmup, args.runs)
            return
        for _ in range(args.warmup):
            replay(backend, tokens)
        runs = [replay(backend, tokens) for _ in range(args.runs)]
        baseline = {number: {**chunk, "ms": median(run[number]["ms"] for run in runs)}
                    for number, chunk in runs[0].items()}
        torch.cuda.synchronize(device)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as profiler:
            replay(backend, tokens, traced=True)

    with (nullcontext() if args.trace else TemporaryDirectory()) as temporary:
        path = args.trace or (Path(temporary) / "decoder_trace.json")
        profiler.export_chrome_trace(str(path))
        if args.trace:
            print(f"CUDA trace：{path.resolve()}")
        report_trace(json.loads(path.read_text(encoding="utf-8")), baseline)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turn", type=Path, help="已保存的 turn.json，默认取最近一条至少有 25 个语音 token 的回复")
    parser.add_argument("--device", default="cuda:1", help="与通话服务的语音解码设备一致")
    parser.add_argument("--flow-path", type=Path, default=ROOT / "glm-4-voice-decoder")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=3)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--trace", type=Path, help="保存可用 Perfetto 查看详情的 CUDA trace JSON")
    mode.add_argument("--compare-cfg", action="store_true", help="用同一批回复 token 对照串行与合批 CFG")
    mode.add_argument("--compare-steps", action="store_true", help="用同一首块 token 对照 Flow 的 10、8、6 步耗时")
    args = parser.parse_args()
    if args.runs < 1 or args.warmup < 0:
        parser.error("--runs 必须大于零，--warmup 不能为负数。")
    try:
        run(args)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"诊断失败：{error}\n")


if __name__ == "__main__":
    main()
