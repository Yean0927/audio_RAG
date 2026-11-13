"""Measure GPU-resident GLM-4-Voice inference from PCM to waveform."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import logging
from pathlib import Path
from statistics import median
import time

from voice_web_demo import ROOT, VoiceBackend, make_prompt
from voice_gpu import audio_prompt_slice, audio_token_range, make_gpu_encoder


CALL_SAMPLE_NAMES = (
    "灯具连接APP--检索正常回答重复且不会读网址",
    "国外项目采购",
    "加盟",
    "品牌合作",
    "如何采购",
    "以旧换新",
)


class InsufficientAudioTokens(ValueError):
    pass


def summarize(rows):
    decode_rates = [row["glm_decode_tokens_per_second"] for row in rows
                    if row["glm_decode_tokens_per_second"] is not None]
    return {
        "wall_ms": median(row["wall_ms"] for row in rows),
        "glm_ms": median(row["stages_ms"]["GLM 生成"] for row in rows),
        "glm_ttft_ms": median(row["glm_ttft_ms"] for row in rows),
        "glm_decode_tokens_per_second": median(decode_rates) if decode_rates else None,
    }


def trace_device_activity(path, measure, first):
    import torch

    marker = "pcm_to_waveform"
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as profiler:
        with torch.profiler.record_function(marker):
            measure(first, trace=True)
    profiler.export_chrome_trace(str(path))
    events = json.loads(path.read_text(encoding="utf-8"))["traceEvents"]
    span = next(event for event in events if event.get("name") == marker and event.get("ph") == "X")
    start = span["ts"]
    end = start + span["dur"]
    activities = [event for event in events if event.get("ph") == "X"
                  and event.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}]
    if not any(event.get("cat") == "kernel" for event in activities):
        print(f"  CUDA trace 无 kernel 记录，无法估算 GPU 活动时间：{path}")
        return
    # Union across both GPUs so overlapping work is counted once.
    intervals = sorted((max(start, event["ts"]), min(end, event["ts"] + event["dur"]))
                       for event in activities)
    cursor, active = start, 0
    for left, right in intervals:
        left = max(cursor, left)
        if right > left:
            active += right - left
            cursor = right
    print(f"  CUDA 跟踪墙钟 {span['dur'] / 1000:.1f} ms，GPU 活动并集 {active / 1000:.1f} ms，"
          f"无已记录 GPU 活动 {(span['dur'] - active) / 1000:.1f} ms")
    print(f"  跟踪含采样开销；无活动时间包含调度、同步及其他等待：{path}")


def make_gpu_prompt(backend, reference_ids, language, persona):
    import torch

    tokenizer = backend.tokenizer
    offset, vocab = audio_token_range(tokenizer, backend.decoder)
    question = "<|begin_of_audio|>" + "".join(f"<|audio_{i}|>" for i in reference_ids) + "<|end_of_audio|>"
    inputs = tokenizer([make_prompt([], question, "", language, persona)], return_tensors="pt")
    audio_slice = audio_prompt_slice(tokenizer, inputs.input_ids, len(reference_ids))
    expected = torch.as_tensor(reference_ids) + offset
    if not torch.equal(inputs.input_ids[0, audio_slice], expected):
        raise ValueError("GLM tokenizer 对这段语音的 ID 映射与 GPU 路径不一致。")
    inputs = inputs.to(backend.llm_device)

    def pack(ids):
        packed = inputs.input_ids.clone()
        packed[:, audio_slice].copy_(ids.to(backend.llm_device, non_blocking=True) + offset)
        return {**inputs, "input_ids": packed}

    return pack, offset, vocab, inputs.input_ids.shape[1]


def benchmark(backend, args, pcm, encode, pack, offset, vocab, *, trace_dir, allow_short_audio):
    import torch
    from transformers import StoppingCriteriaList

    decoder = backend.decoder
    speech_device = backend.speech_device
    llm_device = backend.llm_device
    empty_tokens = torch.empty((1, 0), dtype=torch.long, device=speech_device)
    empty_mel = torch.empty((1, 0, 80), device=speech_device)
    empty_source = torch.empty((1, 1, 0), device=speech_device)
    zero_length = torch.zeros(1, dtype=torch.int32, device=speech_device)
    speaker = torch.zeros((1, decoder.flow.spk_embed_affine_layer.in_features), device=speech_device)
    decoder.hift.stft_window = decoder.hift.stft_window.to(speech_device)

    def is_audio(ids):
        return (ids >= offset) & (ids < offset + vocab)

    def measure(first, trace=False):
        torch.manual_seed(args.seed)
        audio_count = torch.zeros(1, dtype=torch.int32, device=llm_device) if first else None
        events = {}

        def mark(name, device):
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(device))
            events[name] = event

        def observe_token(input_ids, scores, **kwargs):
            # Transformers calls this after adding the newly sampled token to input_ids.
            if "glm_first_token" not in events:
                mark("glm_first_token", llm_device)
            if audio_count is None:
                return False
            audio_count.add_(is_audio(input_ids[:, -1]))
            return audio_count >= args.first_audio_tokens

        criteria = StoppingCriteriaList([observe_token])
        started = time.perf_counter()
        mark("start", speech_device)
        ids = encode(pcm)
        mark("encoded", speech_device)
        mark("pack_start", llm_device)
        inputs = pack(ids)
        mark("glm_start", llm_device)
        with torch.profiler.record_function("glm_generate") if trace else nullcontext():
            result = backend.model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=True, temperature=0.2, top_p=0.8,
                stopping_criteria=criteria, return_dict_in_generate=False,
            )
        mark("glm_end", llm_device)
        generated = result[0, inputs["input_ids"].shape[-1]:]
        audio = (generated[is_audio(generated)] - offset).unsqueeze(0)
        count = audio.shape[-1]
        if count == 0 or (first and count != args.first_audio_tokens):
            raise InsufficientAudioTokens(f"GLM 只输出了 {count} 个语音 token，无法测量所选模式。")
        mark("audio_selected", llm_device)
        mark("flow_input", speech_device)
        audio = audio.to(speech_device, non_blocking=True)
        length = torch.full((1,), count, dtype=torch.int32, device=speech_device)
        mark("flow_start", speech_device)
        mel = decoder.flow.inference(audio, length, empty_tokens, zero_length, empty_mel, zero_length, speaker)
        mark("flow_end", speech_device)
        if first:
            mel = mel[:, :, :-decoder.mel_overlap_len]
        waveform, _ = decoder.hift.inference(mel=mel, cache_source=empty_source)
        if first:
            waveform = waveform[:, :-decoder.source_cache_len]
        mark("waveform_ready", speech_device)
        events["waveform_ready"].synchronize()
        wall = (time.perf_counter() - started) * 1000
        intervals = {
            "Mel + 语音编码": ("start", "encoded"), "输入 token 组装与交接": ("pack_start", "glm_start"),
            "GLM 生成": ("glm_start", "glm_end"), "筛选输出语音 token": ("glm_end", "audio_selected"),
            "解码前 token 交接": ("flow_input", "flow_start"), "Flow": ("flow_start", "flow_end"),
            "HiFT": ("flow_end", "waveform_ready"),
        }
        tokens = generated.numel()
        ttft_ms = events["glm_start"].elapsed_time(events["glm_first_token"])
        decode_ms = events["glm_first_token"].elapsed_time(events["glm_end"])
        decode_tps = (tokens - 1) * 1000 / decode_ms if tokens > 1 and decode_ms > 0 else None
        return {"wall_ms": wall, "tokens": tokens, "audio_tokens": count,
                "glm_ttft_ms": ttft_ms, "glm_decode_tokens_per_second": decode_tps,
                "audio_seconds": waveform.shape[-1] / decoder.hift.sampling_rate,
                "stages_ms": {name: events[a].elapsed_time(events[b]) for name, (a, b) in intervals.items()}}

    modes = ("first", "full") if args.mode == "both" else (args.mode,)
    measurements = {}
    errors = {}
    for mode in modes:
        print(f"测量{'首块' if mode == 'first' else '整段'}，预热 {args.warmup} 次，正式 {args.runs} 次", flush=True)
        try:
            for _ in range(args.warmup):
                measure(mode == "first")
            rows = [measure(mode == "first") for _ in range(args.runs)]
        except InsufficientAudioTokens as error:
            if not allow_short_audio:
                raise
            errors[mode] = str(error)
            print(f"  {error}", flush=True)
            continue
        best = min(rows, key=lambda row: row["wall_ms"])
        summary = summarize(rows)
        print(f"  PCM 在 GPU → 波形在 GPU：最快 {best['wall_ms']:.1f} ms，"
              f"中位数 {summary['wall_ms']:.1f} ms")
        print(f"  GLM 中位数：生成 {summary['glm_ms']:.1f} ms，首 token {summary['glm_ttft_ms']:.1f} ms")
        if summary["glm_decode_tokens_per_second"] is not None:
            print(f"  GLM 首 token 后的生成速度中位数：{summary['glm_decode_tokens_per_second']:.2f} token/s")
        print(f"  最快一轮：GLM {best['tokens']} token，语音 {best['audio_tokens']} token，"
              f"生成音频 {best['audio_seconds']:.2f} 秒")
        for name, ms in best["stages_ms"].items():
            print(f"  {name:<19} {ms:8.1f} ms")
        measurements[mode] = rows
        if trace_dir:
            trace_dir.mkdir(parents=True, exist_ok=True)
            trace_device_activity(trace_dir / f"voice_gpu_{mode}.json", measure, mode == "first")
    return measurements, errors


def run_recording(backend, args, wav, config, *, trace_dir, allow_short_audio):
    import torch
    import torchaudio
    from speech_tokenizer.utils import extract_speech_token

    audio, rate = torchaudio.load(str(wav))
    if not 0 < audio.shape[-1] <= 30 * rate:
        raise ValueError(f"请使用不超过 30 秒的非空录音：{wav}")
    audio = audio.mean(dim=0, keepdim=True)
    with torch.inference_mode():
        pcm = audio.to(backend.speech_device)
        encode = make_gpu_encoder(backend, rate)
        reference = extract_speech_token(backend.whisper, backend.feature_extractor, [(audio, rate)],
                                         device=str(backend.speech_device))[0]
        candidate = encode(pcm).reshape(-1).tolist()
        if candidate != reference:
            raise ValueError("GPU 直连编码和现有服务的语音 token 不一致，未开始计时。")
        pack, offset, vocab, prompt_length = make_gpu_prompt(
            backend, reference, config.get("language", "zh"), config.get("system_prompt", ""),
        )
        if prompt_length + args.max_new_tokens > min(getattr(backend.model.config, "seq_length", 8192), 8192):
            raise ValueError("录音和回复上限超过 GLM 的上下文长度。")
        print(f"录音：{wav.resolve()}，输入语音 token：{len(reference)}，编码与现有路径一致。", flush=True)
        results, errors = benchmark(backend, args, pcm, encode, pack, offset, vocab,
                                    trace_dir=trace_dir, allow_short_audio=allow_short_audio)
    return {"wav": str(wav.resolve()), "input_audio_tokens": len(reference), "modes": results,
            "errors": errors, "summary": {mode: summarize(rows) for mode, rows in results.items()}}


def run(args):
    if args.call_samples:
        wavs = [ROOT / "answer_data" / "calls" / name / "question.wav" for name in CALL_SAMPLE_NAMES]
    else:
        wavs = args.wav or [ROOT / "speech_data/simple/question.wav"]
    missing = [str(wav) for wav in wavs if not wav.is_file()]
    if missing:
        raise ValueError("找不到录音：\n  " + "\n  ".join(missing))

    import torch

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config.is_file() else {}
    backend = VoiceBackend(args)
    backend.load()
    attention = backend.model.config._attn_implementation
    print(f"GLM：{args.quantization} / {backend.llm_device}，attention {attention}，"
          f"编解码：{backend.speech_device}，首块：{args.first_audio_tokens} token，"
          f"matmul TF32：{torch.backends.cuda.matmul.allow_tf32}", flush=True)

    multiple = len(wavs) > 1
    samples = []
    for index, wav in enumerate(wavs, 1):
        trace_dir = args.trace_dir
        if multiple:
            print(f"\n样本 {index}/{len(wavs)} · {wav.parent.name}/{wav.name}", flush=True)
            if trace_dir:
                trace_dir = trace_dir / f"{index:02d}"
        samples.append(run_recording(backend, args, wav, config, trace_dir=trace_dir, allow_short_audio=multiple))

    results = samples[0]["modes"]
    if multiple:
        modes = ("first", "full") if args.mode == "both" else (args.mode,)
        suite_summary = {}
        print("\n先取各录音的重复测量中位数，再取录音间中位数。", flush=True)
        for mode in modes:
            summaries = [sample["summary"][mode] for sample in samples if mode in sample["summary"]]
            summary = {"valid_samples": len(summaries), "total_samples": len(samples)}
            label = "首块" if mode == "first" else "整段"
            if summaries:
                for key in summaries[0]:
                    values = [sample[key] for sample in summaries if sample[key] is not None]
                    summary[key] = median(values) if values else None
                print(f"  {label} {len(summaries)}/{len(samples)}：GLM {summary['glm_ms']:.1f} ms，"
                      f"首 token {summary['glm_ttft_ms']:.1f} ms，总耗时 {summary['wall_ms']:.1f} ms")
                if summary["glm_decode_tokens_per_second"] is not None:
                    print(f"    GLM 首 token 后 {summary['glm_decode_tokens_per_second']:.2f} token/s")
            else:
                print(f"  {label} 0/{len(samples)}：没有有效结果")
            suite_summary[mode] = summary
        results = {"quantization": args.quantization, "attention": attention, "seed": args.seed,
                   "first_audio_tokens": args.first_audio_tokens, "max_new_tokens": args.max_new_tokens,
                   "samples": samples, "summary": suite_summary}
    if args.json:
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not any(sample["modes"] for sample in samples):
        raise ValueError("所有录音均未得到有效的语音测速结果。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--wav", type=Path, nargs="+", action="extend", help="录音路径，可传入多个")
    input_group.add_argument("--call-samples", action="store_true", help="使用固定的六个中文客服录音")
    parser.add_argument("--config", type=Path, default=ROOT / "call_config.json")
    parser.add_argument("--model-path", default="zai-org/glm-4-voice-9b")
    parser.add_argument("--tokenizer-path", default="zai-org/glm-4-voice-tokenizer")
    parser.add_argument("--flow-path", type=Path, default=ROOT / "glm-4-voice-decoder")
    parser.add_argument("--quantization", choices=("4bit", "bf16"), default="4bit")
    parser.add_argument("--attn-implementation", choices=("auto", "eager", "sdpa", "flash_attention_2"),
                        default="auto", help="GLM 注意力实现；auto 保持 Transformers 默认选择")
    parser.add_argument("--llm-device", default="cuda:0")
    parser.add_argument("--speech-device", default="cuda:1")
    parser.add_argument("--mode", choices=("first", "full", "both"), default="both")
    parser.add_argument("--first-audio-tokens", type=int, default=25)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--tf32", action="store_true", help="允许 FP32 矩阵计算和卷积使用 TF32")
    parser.add_argument("--trace-dir", type=Path, help="另跑 CUDA 跟踪，保存时间线并统计 GPU 活动")
    parser.add_argument("--json", type=Path, help="保存各轮分阶段耗时")
    args = parser.parse_args()
    if not 10 <= args.first_audio_tokens <= 100 or args.max_new_tokens < args.first_audio_tokens:
        parser.error("首块需要 10–100 个 token，且不能超过 --max-new-tokens。")
    if args.warmup < 0 or args.runs < 1:
        parser.error("--runs 必须大于零，--warmup 不能为负数。")
    logging.basicConfig(level=logging.INFO)
    try:
        run(args)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"测速失败：{error}\n")


if __name__ == "__main__":
    main()
