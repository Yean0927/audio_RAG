"""每批并行提交两条录音，评测 FAQ 通话服务的 ROUGE-L 和首块语音时延。"""

import argparse
import asyncio
import json
import math
from pathlib import Path
import re
from statistics import mean
from time import perf_counter
from urllib.parse import urlsplit, urlunsplit


from voice_service import ROOT
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".aif", ".aiff", ".au"}
FAQ_PREFIX = "## 问题："
BATCH_SIZE = 2


class CharacterTokenizer:
    def tokenize(self, text):
        return [character for character in text if not character.isspace()]


def call_url(value):
    parsed = urlsplit(value)
    scheme = {"http": "ws", "https": "wss", "ws": "ws", "wss": "wss"}.get(parsed.scheme)
    path = parsed.path.rstrip("/") or "/api/call"
    if not scheme or not parsed.hostname or not path.endswith("/api/call") or parsed.query or parsed.fragment:
        raise ValueError("--url 请输入通话服务地址，例如 http://127.0.0.1:7860 或 ws://127.0.0.1:7860/api/call。")
    return urlunsplit((scheme, parsed.netloc, path, "stream=1", ""))


def split_faq(passage):
    question, separator, answer = passage.partition("\n答案：")
    if passage.startswith(FAQ_PREFIX) and separator:
        title = question.split("\n", 1)[0].removeprefix(FAQ_PREFIX).strip()
        return title, answer
    return None, passage


def load_gold(path, faq_file, recordings):
    labels = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(labels, dict) or not labels or any(
        not isinstance(title, str) or not title.strip() for title in labels.values()
    ):
        raise ValueError("--gold 需要 JSON 对象，格式为录音文件名到 FAQ 标题的映射。")
    recording_names = {recording.name for recording in recordings}
    unknown = sorted(labels.keys() - recording_names)
    if unknown:
        raise ValueError(f"标注中的录音不在待测目录里：{', '.join(unknown)}")
    answers = {}
    faq_text = faq_file.read_text(encoding="utf-8-sig").strip()
    for passage in re.split(rf"\n\n(?={FAQ_PREFIX})", faq_text):
        title, answer = split_faq(passage)
        if title is None or not answer.strip():
            continue
        if title in answers:
            raise ValueError(f"知识库存在重复的 FAQ 标题：{title}")
        answers[title] = answer
    missing = sorted(set(labels.values()) - answers.keys())
    if missing:
        raise ValueError(f"知识库中找不到这些标注标题：{', '.join(missing)}")
    return {filename: (title, answers[title]) for filename, title in labels.items()}


def score_reply(reply, result, scorer, gold=None):
    actual = reply.get("text")
    if not isinstance(actual, str):
        raise ValueError("服务端未返回回复文本。")
    result["actual"] = actual

    retrieval = reply.get("retrieval")
    if not isinstance(retrieval, dict):
        raise ValueError("本轮没有检索结果，请检查服务是否启用了知识库。")
    metric = retrieval.get("selected_metric")
    selected = retrieval.get(metric) if isinstance(metric, str) else None
    if not isinstance(selected, list) or not selected or not isinstance(selected[0], dict):
        raise ValueError("本轮没有实际采用的检索条目。")
    top = selected[0]
    passage = top.get("passage")
    if not isinstance(passage, str) or not passage.strip():
        raise ValueError("实际采用的检索条目为空。")

    retrieved_question, expected = split_faq(passage)
    result.update(selected_metric=metric, rank=top.get("rank"), score=top.get("score"),
                  passage=passage, expected=expected, retrieved_question=retrieved_question)
    if not expected.strip():
        raise ValueError("检索条目中的 FAQ 答案为空。")
    rouge_l = scorer.score(expected, actual)["rougeL"].fmeasure
    if gold is not None:
        gold_question, gold_answer = gold
        result.update(gold_answer=gold_answer, retrieval_hit=retrieved_question == gold_question,
                      gold_rouge_l=scorer.score(gold_answer, actual)["rougeL"].fmeasure)
    result.update(status="ok", rouge_l=rouge_l)


async def request_reply(url, wav, duration, result):
    from websockets import connect

    async with connect(url, max_size=16 * 1024 * 1024, close_timeout=2) as socket:
        ready = json.loads(await socket.recv())
        if ready.get("type") != "ready":
            raise RuntimeError(ready.get("message") or "语音服务尚未就绪。")
        limit = ready.get("max_audio_seconds")
        if limit is not None and duration > limit:
            raise ValueError(f"录音超过服务端的 {limit} 秒限制。")
        await socket.send(json.dumps({"type": "utterance", "id": 1}))
        started = perf_counter()
        await socket.send(wav)
        while True:
            raw = await socket.recv()
            arrived = perf_counter()
            message = json.loads(raw)
            kind = message.get("type")
            if kind == "error":
                raise RuntimeError(message.get("message") or "服务端推理失败。")
            if message.get("id") != 1:
                continue
            if result["first_audio_ms"] is None:
                if kind == "audio_chunk" and message.get("pcm"):
                    result.update(first_audio_ms=(arrived - started) * 1000, first_audio_source="audio_chunk")
                elif kind == "reply" and message.get("audio"):
                    result.update(first_audio_ms=(arrived - started) * 1000, first_audio_source="reply_audio")
            if kind == "reply":
                await socket.send(json.dumps({"type": "hangup"}))
                await socket.wait_closed()
                return message


def print_result(result):
    latency = result["first_audio_ms"]
    audio = "首块语音无数据"
    if latency is not None:
        label = "整段语音" if result["first_audio_source"] == "reply_audio" else "首块语音"
        audio = f"{label} {latency:.0f} ms"
    if result["status"] == "error":
        print(f"无法评测  {audio}  {result['file']}\n  {result['error']}")
    else:
        metrics = f"ROUGE-L {result['rouge_l']:.2%}"
        if "gold_rouge_l" in result:
            hit = "命中" if result["retrieval_hit"] else "未命中"
            metrics += f"  标准答案 {result['gold_rouge_l']:.2%}  检索{hit}"
        print(f"{metrics}  {audio}  {result['file']}")


async def evaluate(url, recordings, timeout, *, compare_url=None, gold=None):
    from rouge_score.rouge_scorer import RougeScorer
    from websockets.exceptions import WebSocketException

    from .profile_voice_call import read_recording

    scorer = RougeScorer(["rougeL"], tokenizer=CharacterTokenizer())
    urls = {"primary": url, "comparison": compare_url} if compare_url else {"primary": url}
    results = {service: [] for service in urls}
    gold = gold or {}

    async def collect_reply(endpoint, path, recording):
        label = gold.get(path.name)
        result = {"file": path.name, "rouge_l": None, "first_audio_ms": None, "first_audio_source": None}
        if label is not None:
            result["gold_question"] = label[0]
        try:
            if isinstance(recording, Exception):
                raise recording
            wav, duration = recording
            reply = await asyncio.wait_for(request_reply(endpoint, wav, duration, result), timeout=timeout)
            return result, reply
        except asyncio.TimeoutError:
            result.update(status="error", error=f"等待服务端完成本轮超时（{timeout:g} 秒）。")
        except (OSError, ValueError, RuntimeError, WebSocketException) as error:
            result.update(status="error", error=str(error))
        return result, None

    for start in range(0, len(recordings), BATCH_SIZE):
        paths = recordings[start:start + BATCH_SIZE]
        print(f"[{start + 1}–{start + len(paths)}/{len(recordings)}] 提交 {len(paths)} 条录音", flush=True)
        batch = []
        for path in paths:
            try:
                recording = read_recording(path)
            except (OSError, ValueError, RuntimeError) as error:
                recording = error
            batch.append((path, recording))
        for service, endpoint in urls.items():
            if compare_url:
                print(f"  {'主服务' if service == 'primary' else '对照服务'}", flush=True)
            replies = await asyncio.gather(*(collect_reply(endpoint, path, recording) for path, recording in batch))
            for row, reply in replies:
                if reply is not None:
                    try:
                        score_reply(reply, row, scorer, gold=gold.get(row["file"]))
                    except ValueError as error:
                        row.update(status="error", error=str(error))
                print_result(row)
                results[service].append(row)
    return results if compare_url else results["primary"]


def summarize(results, labeled=0):
    scores = [row["rouge_l"] for row in results if row["rouge_l"] is not None]
    latencies = [row["first_audio_ms"] for row in results if row["first_audio_ms"] is not None]
    summary = {"total": len(results), "evaluated": len(scores), "error": len(results) - len(scores),
               "mean_rouge_l": mean(scores) if scores else None, "first_audio_count": len(latencies),
               "min_first_audio_ms": min(latencies, default=None),
               "max_first_audio_ms": max(latencies, default=None),
               "mean_first_audio_ms": mean(latencies) if latencies else None}
    if labeled:
        scored = [row for row in results if "gold_rouge_l" in row]
        hits = sum(row["retrieval_hit"] for row in scored)
        summary.update(labeled=labeled, gold_evaluated=len(scored), retrieval_hits=hits,
                       retrieval_hit_rate=hits / len(scored) if scored else None,
                       mean_gold_rouge_l=mean(row["gold_rouge_l"] for row in scored) if scored else None)
    return summary


def summarize_pairs(primary, comparison):
    successful = [(left, right) for left, right in zip(primary, comparison, strict=True)
                  if left["status"] == right["status"] == "ok"]
    same_passage = [(left, right) for left, right in successful if left["passage"] == right["passage"]]
    summary = {"both_evaluated": len(successful), "same_retrieval": len(same_passage),
               "different_retrieval": len(successful) - len(same_passage),
               "same_retrieval_rouge_l": None}
    if same_passage:
        summary["same_retrieval_rouge_l"] = {
            "primary": mean(left["rouge_l"] for left, _ in same_passage),
            "comparison": mean(right["rouge_l"] for _, right in same_passage),
        }
    gold_pairs = [(left, right) for left, right in successful
                  if "gold_rouge_l" in left and "gold_rouge_l" in right]
    if gold_pairs:
        summary["gold_pairs"] = len(gold_pairs)
        summary["gold_rouge_l"] = {
            "primary": mean(left["gold_rouge_l"] for left, _ in gold_pairs),
            "comparison": mean(right["gold_rouge_l"] for _, right in gold_pairs),
        }
    return summary


def print_summary(summary):
    print(f"共 {summary['total']} 条，有效评测 {summary['evaluated']}，无法评测 {summary['error']}。")
    score = summary["mean_rouge_l"]
    print("平均 ROUGE-L 无有效样本。" if score is None else f"平均 ROUGE-L {score:.2%}")
    if summary["first_audio_count"]:
        print(f"首块语音时延（{summary['first_audio_count']} 条）"
              f"  最早 {summary['min_first_audio_ms']:.0f} ms"
              f"  最晚 {summary['max_first_audio_ms']:.0f} ms"
              f"  平均 {summary['mean_first_audio_ms']:.0f} ms")
    else:
        print("首块语音时延无有效样本。")
    if "labeled" in summary:
        print(f"标注 {summary['labeled']} 条，有效标注样本 {summary['gold_evaluated']} 条。")
        if summary["gold_evaluated"]:
            print(f"FAQ 检索命中率 {summary['retrieval_hit_rate']:.2%}，"
                  f"标准答案 ROUGE-L {summary['mean_gold_rouge_l']:.2%}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, default=ROOT / "answer_data/test_audio",
                        help="待测录音目录，默认 answer_data/test_audio")
    parser.add_argument("--url", default="http://127.0.0.1:7860", help="通话服务地址")
    parser.add_argument("--compare-url", help="用同一批录音追加评测这个通话服务，并汇总配对结果")
    parser.add_argument("--gold", type=Path, help="可选 JSON，录音文件名映射到正确的 FAQ 首行标题")
    parser.add_argument("--faq-file", type=Path, default=ROOT / "resources/customer_faq.md",
                        help="读取标准答案的 FAQ 知识库，需要与通话服务使用的文件一致")
    parser.add_argument("--timeout", type=float, default=180, help="单条录音的服务响应超时秒数")
    parser.add_argument("--json", type=Path, help="保存逐条结果和汇总的 JSON 文件路径")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout 必须是有限的正数。")
    try:
        url = call_url(args.url)
        compare_url = call_url(args.compare_url) if args.compare_url else None
        if compare_url == url:
            raise ValueError("主服务和对照服务需要使用不同的地址。")
        recordings = sorted(path for path in args.audio_dir.iterdir()
                            if path.is_file() and not path.name.startswith(".")
                            and path.suffix.lower() in AUDIO_SUFFIXES)
        if not recordings:
            raise ValueError(f"{args.audio_dir} 中没有找到受支持的音频文件。")
        gold = load_gold(args.gold, args.faq_file, recordings) if args.gold else {}
        results = asyncio.run(evaluate(url, recordings, args.timeout, compare_url=compare_url, gold=gold))
        report = {"audio_dir": str(args.audio_dir.resolve()), "batch_size": BATCH_SIZE}
        if compare_url:
            services = {}
            for service, name, address in (("primary", "主服务", args.url),
                                           ("comparison", "对照服务", args.compare_url)):
                rows = results[service]
                summary = summarize(rows, len(gold))
                services[service] = {"url": address, "summary": summary, "results": rows}
                print(f"\n{name} {address}")
                print_summary(summary)
            paired = summarize_pairs(results["primary"], results["comparison"])
            print(f"\n两边均有结果 {paired['both_evaluated']} 条，"
                  f"检索条目相同 {paired['same_retrieval']} 条，不同 {paired['different_retrieval']} 条。")
            if paired["same_retrieval_rouge_l"] is not None:
                scores = paired["same_retrieval_rouge_l"]
                print(f"同条 FAQ 的 ROUGE-L  主服务 {scores['primary']:.2%}  对照服务 {scores['comparison']:.2%}")
            if "gold_rouge_l" in paired:
                scores = paired["gold_rouge_l"]
                print(f"共同标注样本 {paired['gold_pairs']} 条的标准答案 ROUGE-L  "
                      f"主服务 {scores['primary']:.2%}  对照服务 {scores['comparison']:.2%}")
            report.update(services=services, paired=paired)
            errors = sum(service["summary"]["error"] for service in services.values())
        else:
            summary = summarize(results, len(gold))
            print()
            print_summary(summary)
            report.update(summary=summary, results=results)
            errors = summary["error"]
        if args.json:
            args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (ImportError, OSError, ValueError) as error:
        parser.exit(2, f"评测失败：{error}\n")
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
