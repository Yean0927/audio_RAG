"""Measure a streamed call using a fixed recording and the real WebSocket endpoint."""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
from pathlib import Path
from statistics import median
import time
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
import wave

import numpy as np
import soundfile as sf
from websockets import connect
from websockets.exceptions import WebSocketException


ROOT = Path(__file__).resolve().parent
INPUT_RATE = 16000
OUTPUT_RATE = 22050
# Match the current scheduler in call_web/call.js.
PLAYBACK_LEAD_MS = 20


def read_recording(path):
    samples, rate = sf.read(path, dtype="float32", always_2d=True)
    if not len(samples) or not np.isfinite(samples).all():
        raise ValueError("录音为空或含无效采样。")
    mono = samples.mean(axis=1)
    if rate != INPUT_RATE:
        positions = np.arange(len(mono) * INPUT_RATE // rate) * (rate / INPUT_RATE)
        mono = np.interp(positions, np.arange(len(mono)), mono)
    pcm = np.rint(np.clip(mono, -1, 1) * 32767).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setparams((1, 2, INPUT_RATE, 0, "NONE", "not compressed"))
        stream.writeframes(pcm.tobytes())
    return output.getvalue(), len(pcm) / INPUT_RATE


def probe_url(url, block_sizes=None):
    parsed = urlsplit(url)
    if parsed.scheme not in ("ws", "wss", "http", "https") or not parsed.netloc:
        raise ValueError("--url 需要形如 http://127.0.0.1:7860 或 ws://127.0.0.1:7860/api/call 的地址。")
    query = dict(parse_qsl(parsed.query))
    query.update(stream="1", ack="1", profile="1")
    if block_sizes:
        query["block_sizes"] = ",".join(map(str, block_sizes))
    scheme = {"http": "ws", "https": "wss"}.get(parsed.scheme, parsed.scheme)
    return urlunsplit((scheme, parsed.netloc, parsed.path.rstrip("/") or "/api/call", urlencode(query), ""))


def signal_bounds(pcm):
    """Estimate the start/end of audible signal with 10 ms RMS windows at -45 dBFS."""
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
    width = OUTPUT_RATE // 100
    frames = samples[:len(samples) // width * width].reshape(-1, width)
    audible = np.flatnonzero(np.mean(frames ** 2, axis=1) >= 10 ** (-45 / 10))
    if len(audible) == 0:
        return None, None
    return (audible[0] * width * 1000 / OUTPUT_RATE,
            (len(samples) - (audible[-1] + 1) * width) * 1000 / OUTPUT_RATE)


async def measure(url, wav, duration):
    async with connect(url, max_size=16 * 1024 * 1024) as socket:
        hello = json.loads(await socket.recv())
        if hello.get("type") != "ready":
            raise RuntimeError(hello.get("message", "语音服务尚未就绪。"))
        if duration > hello["max_audio_seconds"]:
            raise ValueError(f"录音超过服务器的 {hello['max_audio_seconds']} 秒限制。")
        query_pcm = hello.get("query_pcm")
        query_audio_ms = len(base64.b64decode(query_pcm, validate=True)) * 1000 / (2 * OUTPUT_RATE) if query_pcm else 0
        await socket.send(json.dumps({"type": "utterance", "id": 1}))
        started = time.perf_counter()
        await socket.send(wav)
        upload_ms = (time.perf_counter() - started) * 1000
        chunks, audio = [], []
        acknowledgment = None
        thinking_ms = None
        play_until = 0
        while True:
            raw = await socket.recv()
            arrival = (time.perf_counter() - started) * 1000
            message = json.loads(raw)
            kind = message.get("type")
            if kind == "utterance_received" and query_audio_ms:
                play_at = arrival + PLAYBACK_LEAD_MS
                acknowledgment = {"arrival_ms": arrival, "play_at_ms": play_at, "audio_ms": query_audio_ms}
                play_until = play_at + query_audio_ms
            elif kind == "thinking":
                thinking_ms = arrival
            elif kind == "audio_chunk":
                pcm = base64.b64decode(message["pcm"], validate=True)
                if not pcm or len(pcm) % 2:
                    raise ValueError("服务端返回了无效的 PCM 音频块。")
                decoded = (time.perf_counter() - started) * 1000
                play_at = play_until if play_until > decoded else decoded + PLAYBACK_LEAD_MS
                audio_ms = len(pcm) / 2 / OUTPUT_RATE * 1000
                chunks.append({"arrival_ms": arrival, "client_unpack_ms": decoded - arrival,
                               "audio_ms": audio_ms, "play_at_ms": play_at,
                               "gap_before_ms": play_at - play_until if chunks else 0})
                audio.append(pcm)
                play_until = play_at + audio_ms
            elif kind == "reply":
                await socket.send(json.dumps({"type": "hangup"}))
                await socket.wait_closed()
                for chunk, pcm in zip(chunks, audio):
                    chunk["leading_quiet_ms"], chunk["trailing_quiet_ms"] = signal_bounds(pcm)
                return {"input_seconds": duration, "upload_enqueue_ms": upload_ms,
                        "thinking_ms": thinking_ms, "reply_ms": arrival, "chunks": chunks,
                        "acknowledgment": acknowledgment, "server": message.get("latency_profile")}
            elif kind == "error":
                raise RuntimeError(message.get("message", "服务端推理失败。"))


def report(result):
    chunks = result["chunks"]
    print(f"录音 {result['input_seconds']:.2f} s，提交写入 WebSocket {result['upload_enqueue_ms']:.0f} ms，"
          f"完整回复到达 {result['reply_ms']:.0f} ms")
    if result["thinking_ms"] is not None:
        print(f"提交 → 收到 thinking {result['thinking_ms']:.0f} ms")
    if chunks:
        first = chunks[0]
        print(f"提交 → 正式首块到达 {first['arrival_ms']:.0f} ms，预计开播 {first['play_at_ms']:.0f} ms")
        if acknowledgment := result.get("acknowledgment"):
            ack_end = acknowledgment["play_at_ms"] + acknowledgment["audio_ms"]
            print(f"「正在为您查询」预计在 {ack_end:.0f} ms 播完，"
                  f"与正式首块之间的播放空档 {first['play_at_ms'] - ack_end:.0f} ms")
        if first["leading_quiet_ms"] is not None:
            onset = first["play_at_ms"] + first["leading_quiet_ms"]
            print(f"预计听到首个声音 {onset:.0f} ms，其中首块前导低电平约 {first['leading_quiet_ms']:.0f} ms")
        if len(chunks) >= 2:
            second = chunks[1]
            print(f"首两块到达间隔 {second['arrival_ms'] - first['arrival_ms']:.0f} ms，"
                  f"首块可播 {first['audio_ms']:.0f} ms，预计中间空档 {second['gap_before_ms']:.0f} ms")
            if first["trailing_quiet_ms"] is not None and second["leading_quiet_ms"] is not None:
                edge_quiet = first["trailing_quiet_ms"] + second["leading_quiet_ms"]
                print(f"首块末尾与第二块开头的低电平合计约 {edge_quiet:.0f} ms（PCM 估算）")
    else:
        print("这轮没有收到流式音频块。")

    server = result["server"]
    if not server:
        print("服务端未返回分阶段数据。请更新计时代码后重跑以拆分 token 生成和解码。")
        return
    events = server["events"]
    sizes = "/".join(map(str, server.get("block_sizes", [])))
    blocks = f"，分块 {sizes}" if sizes else ""
    print(f"服务端计时，基准为收到录音，通话 ID {server['session']}{blocks}")
    stages = [
        ("wav_saved", "接收录音"), ("inference_slot_acquired", "等待推理队列"),
        ("worker_started", "启动推理线程"), ("backend_lock_acquired", "等待 GPU 模型锁"),
        ("answer_start", "进入推理"), ("audio_prepared", "音频读取与重采样"),
        ("models_ready", "模型加载检查"), ("input_tokenizer_start", "输入语音 tokenizer 启动"),
        ("input_tokenizer_ready", "输入语音 tokenizer 完成"), ("sonar_start", "SONAR 检索启动"),
        ("sonar_encoder_ready", "SONAR 语音编码器加载/缓存"),
        ("knowledge_vectors_ready", "知识库文本向量化/缓存"),
        ("sonar_query_start", "SONAR 查询向量开始"), ("sonar_query_ready", "SONAR 查询向量完成"),
        ("knowledge_search_ready", "知识库相似度检索完成"), ("sonar_finished", "SONAR 检索结束"),
        ("rag_finished", "检索分支结束"),
        ("knowledge_search_skipped", "未启用知识库"), ("prompt_built", "Prompt 组装"),
        ("glm_tokenizer_start", "GLM 输入 tokenizer 启动"),
        ("glm_tokenizer_ready", "GLM 输入 tokenizer 完成"),
        ("prompt_on_device", "GLM 输入传到 GPU"), ("prompt_ready", "GLM 输入准备完成"),
        ("generate_start", "GLM 开始生成"),
        ("glm_prefill_ready", "GLM 首次前向完成"), ("first_token", "GLM 首 token 输出"),
        ("first_audio_token", "首语音 token 输出"), ("generate_end", "GLM 完整生成完成"),
    ]
    by_stage = {event["stage"]: event["ms"] for event in events if "chunk" not in event and "packet" not in event}
    rag = next((event for event in events if event["stage"] == "rag_finished"), None)
    previous = 0
    for stage, label in stages:
        if stage in by_stage:
            elapsed = by_stage[stage]
            if stage == "rag_finished" and rag is not None:
                label = "RAG 已启用" if rag["enabled"] else "RAG 未启用"
            print(f"  {label:<17} +{elapsed - previous:7.0f} ms  累计 {elapsed:7.0f} ms")
            previous = elapsed

    by_chunk, by_packet = {}, {}
    for event in events:
        if "chunk" in event:
            by_chunk.setdefault(event["chunk"], {})[event["stage"]] = event
        if "packet" in event:
            by_packet.setdefault(event["packet"], {})[event["stage"]] = event
    first_pcm = by_packet.get(1, {}).get("chunk_pcm_ready")
    if first_pcm and "first_audio_token" in by_stage:
        first_ready = by_chunk[first_pcm["chunk"]]["chunk_tokens_ready"]["ms"]
        print(f"  首语音 token → 首块 token 齐备 {first_ready - by_stage['first_audio_token']:.0f} ms")
    print("  块    音频   token 齐备    解码排队  解码/回传  PCM打包  服务端提交   客户端到达    预计空档（ms）")
    for index, chunk in enumerate(chunks, 1):
        packet = by_packet.get(index, {})
        if "chunk_pcm_ready" not in packet or "chunk_send_start" not in packet:
            continue
        pcm = packet["chunk_pcm_ready"]
        block = by_chunk[pcm["chunk"]]
        ready = block["chunk_tokens_ready"]
        start = block["chunk_decode_start"]["ms"]
        decoded = block["chunk_decoded"]["ms"]
        sent = packet["chunk_send_start"]["ms"]
        print(f" {index:2} {chunk['audio_ms']:7.0f} {ready['ms']:10.0f} {start - ready['ms']:11.0f} "
              f"{decoded - start:10.0f} {pcm['ms'] - decoded:9.0f} {sent:11.0f} "
              f"{chunk['arrival_ms']:12.0f} {chunk['gap_before_ms']:11.0f}  ({ready['tokens']} token)")

    if len(chunks) >= 2 and 1 in by_packet and 2 in by_packet:
        first_sent = by_packet[1]["chunk_send_start"]["ms"]
        second_sent = by_packet[2]["chunk_send_start"]["ms"]
        pcm = by_packet[2].get("chunk_pcm_ready")
        extra = chunks[1]["arrival_ms"] - chunks[0]["arrival_ms"] - (second_sent - first_sent)
        if pcm:
            block = by_chunk[pcm["chunk"]]
            remaining = [first_sent, block["chunk_tokens_ready"]["ms"], block["chunk_decode_start"]["ms"],
                         block["chunk_decoded"]["ms"], pcm["ms"], second_sent]
            remaining = [max(first_sent, value) for value in remaining]
            intervals = [end - start for start, end in zip(remaining, remaining[1:])]
            labels = ["等 token", "等解码开始", "解码/回传", "PCM 打包", "服务端排队及编码"]
            detail = "，".join(f"{label} {ms:.0f} ms" for label, ms in zip(labels, intervals))
            print(f"  首块提交后 → 第二块提交 {second_sent - first_sent:.0f} ms，其中{detail}")
            if chunks[1]["gap_before_ms"] > 0:
                largest = max(zip(labels + ["传输与客户端接收额外间隔"], intervals + [extra]), key=lambda item: item[1])
                print(f"  首两块间隔中耗时最长的是{largest[0]}（{largest[1]:.0f} ms）")
        print(f"  首两块的客户端到达间隔比服务端提交间隔多 {extra:+.0f} ms（传输与客户端接收合计差值）")
    for stage, label in [("reply_tokens_decoded", "回复 token 解码成文本"),
                         ("full_audio_decoder_start", "整段语音解码"),
                         ("full_audio_decoder_ready", "整段语音解码完成"),
                         ("answer_wav_saved", "完整 WAV 保存"),
                         ("answer_ready", "回答记录与文件就绪")]:
        if stage in by_stage:
            prior_events = [event["ms"] for event in events if event["stage"] != stage
                            and event["ms"] <= by_stage[stage]]
            elapsed = by_stage[stage]
            print(f"  {label} +{elapsed - max(prior_events, default=0):.0f} ms  累计 {elapsed:.0f} ms")
            previous = by_stage[stage]
    if "reply_ready" in by_stage:
        print(f"  服务端完整回复就绪 {by_stage['reply_ready']:.0f} ms")


async def run(args):
    wav, duration = read_recording(args.wav)
    variants = [None, args.compare_blocks] if args.compare_blocks else [None]
    results = []
    for trial in range(1, args.runs + 1):
        for sizes in variants if trial % 2 else reversed(variants):
            name = "分块 " + "/".join(map(str, sizes)) if sizes else "默认分块"
            print(f"\n第 {trial}/{args.runs} 轮 · {name}", flush=True)
            result = await asyncio.wait_for(measure(probe_url(args.url, sizes), wav, duration), args.timeout)
            report(result)
            if args.compare_blocks:
                reported_sizes = (result["server"] or {}).get("block_sizes")
                if not reported_sizes or (sizes and tuple(reported_sizes) != sizes):
                    raise RuntimeError("服务端尚未支持分块对照，请上传最新后端代码后重启。")
            results.append(result)
    if args.json:
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.compare_blocks:
        for sizes in dict.fromkeys(tuple(result["server"]["block_sizes"]) for result in results):
            matching = [result["chunks"] for result in results if tuple(result["server"]["block_sizes"]) == sizes]
            first = [chunk[0]["play_at_ms"] for chunk in matching if chunk]
            gap = [chunk[1]["gap_before_ms"] for chunk in matching if len(chunk) >= 2]
            total = [sum(item["gap_before_ms"] for item in chunk) for chunk in matching if chunk]
            name = "/".join(map(str, sizes))
            print(f"\n分块 {name}（中位数）")
        for label, values in [("正式回复预计开播", first), ("正式回复首两块空档", gap), ("正式回复累计空档", total)]:
                if values:
                    print(f"  {label} {median(values):.0f} ms")


def parse_block_sizes(value):
    try:
        sizes = tuple(map(int, value.split(",")))
    except ValueError:
        sizes = ()
    if len(sizes) != 3 or any(not 10 <= size <= 100 for size in sizes):
        raise argparse.ArgumentTypeError("请指定三个 10–100 的整数，例如 25,32,100。")
    return sizes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--wav", type=Path, default=ROOT / "speech_data/simple/question.wav")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--compare-blocks", type=parse_block_sizes, help="与默认分块交替对照，例如 25,32,100")
    parser.add_argument("--timeout", type=float, default=180, help="每轮超时秒数")
    parser.add_argument("--json", type=Path, help="保存逐轮原始计时")
    args = parser.parse_args()
    if args.runs < 1 or args.timeout <= 0:
        parser.error("--runs 和 --timeout 必须大于零。")
    try:
        asyncio.run(run(args))
    except (OSError, ValueError, RuntimeError, WebSocketException) as error:
        parser.exit(1, f"诊断失败：{error or '等待服务响应超时'}\n")


if __name__ == "__main__":
    main()
