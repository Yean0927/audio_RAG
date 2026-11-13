"""Shared voice conversation, audio I/O and SONAR retrieval."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import logging
import math
from pathlib import Path
import re
import sys
import threading
import time
from typing import TYPE_CHECKING
import uuid
import wave

if TYPE_CHECKING:
    from torch import Tensor

ROOT = Path(__file__).resolve().parent
for directory in (ROOT / "src", ROOT / "third_party/Matcha-TTS"):
    sys.path.insert(0, str(directory))

LOG = logging.getLogger("voice-chat")
STREAM_BLOCK_SIZES = (40, 40, 48)
FAQ_QUESTION_PREFIX = "## 问题："
DEFAULT_OUTPUT_SAMPLE_RATE = 22050
TURN_CHECK_MAX_TOKENS = 32

BACKENDS = {
    "glm": ("GLM-4-Voice", "zai-org/glm-4-voice-9b"),
    "kimi": ("Kimi-Audio", "moonshotai/Kimi-Audio-7B-Instruct"),
    "omni": ("Qwen2.5-Omni", "Qwen/Qwen2.5-Omni-7B"),
}


class InferenceCancelled(Exception):
    """A caller has hung up or started a newer utterance."""


def check_cancelled(event):
    if event is not None and event.is_set():
        raise InferenceCancelled()


def pcm16_bytes(waveform):
    return waveform.clamp(-1, 1).mul(32767).round().short().cpu().numpy().astype("<i2", copy=False).tobytes()


def waveform_wav(waveform, sample_rate):
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setparams((1, 2, sample_rate, 0, "NONE", "not compressed"))
        wav.writeframes(pcm16_bytes(waveform))
    return output.getvalue()


def answer_instruction(context, language):
    instruction = (
        "你是一位语音助手。请用中文简洁、自然地回答，通常两到三句话即可。"
        if language == "zh" else
        "You are a voice assistant. Answer naturally and briefly, usually in two or three sentences. "
    )
    if context:
        instruction += (
            "请听清本轮语音提问，从参考资料中选取相关事实，用自己的话回答。"
            "只输出直接对来电用户说的内容。办理类问题要保留资料中的条件、流程和所需信息。"
            "资料不足时请说明无法确认。业务开放情况以资料明确记载为准。"
            if language == "zh" else
            "Answer the current spoken question in your own words using relevant facts from the reference material. "
            "Output only what you would say directly to the caller. Preserve applicable conditions, steps, "
            "and required information. Say when the available information is insufficient. "
        )
    return instruction


def turn_instruction(language):
    return (
        "听这段语音，只输出一行文字。先辨认用户实际说出的字词，再判断是否说完。"
        "如果只有咳嗽、喷嚏、环境噪音或回声，或者听不清任何字词，只输出 3。"
        "如果用户已说完（包括简短的确认或拒绝），输出 1:，后面写实际听到的开头几个字词。"
        "如果用户还没说完、音频在话中被截断，或听不出是否说完，输出 2:，后面也写实际听到的开头几个字词。"
        "保留原语言，逐字听写，不猜测，不描述声音，不回答用户的问题。"
        if language == "zh" else
        "Listen to the caller and output one line of text. Identify the words actually spoken before judging the turn. "
        "Output only 3 for coughing, sneezing, background noise, echo, or when no caller words are intelligible. "
        "If the caller finished, including a short confirmation or refusal, output 1: followed by the first few words verbatim. "
        "If the caller paused mid-thought, the audio cuts off during speech, or it is unclear whether they finished, "
        "output 2: followed by the first few words verbatim. "
        "Keep the original language. Do not guess, describe sounds, or answer the caller."
    )


def turn_decision(text):
    match = re.fullmatch(r"([12])\s*[:：]\s*([^\r\n]+)", text.strip())
    if match:
        spoken = re.sub(r"\[[^\]]*\]|\([^)]*\)|【[^】]*】|（[^）]*）|<[^>]*>", "", match[2])
        if any(char.isalnum() for char in spoken):
            return "reply" if match[1] == "1" else "wait"
    return "discard"


def greeting_instruction(language):
    return (
        "请用自然的客服语气逐字朗读下一行，只读原文。" if language == "zh" else
        "Read the next line aloud verbatim in a natural customer service voice."
    )


@dataclass
class VoiceReply:
    text: str
    waveform: Tensor | None
    question_tokens: str = ""
    assistant_tokens: str = ""
    input_tokens: list[int] | None = None


class VoiceBackendBase:
    output_sample_rate = DEFAULT_OUTPUT_SAMPLE_RATE

    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.ready = False
        self.embedding = None
        self.embedding_language = None
        self.index_key = None
        self.chunks = []
        self.vectors = None
        self.vector_chunks = None

    def set_devices(self):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("没有检测到可用的 CUDA 显卡，请检查当前 .venv。")
        self.llm_device = torch.device(self.args.llm_device)
        self.speech_device = torch.device(self.args.speech_device)
        for device in (self.llm_device, self.speech_device):
            if device.type != "cuda" or device.index is None or device.index >= torch.cuda.device_count():
                raise RuntimeError(f"显卡 {device} 不可用；单卡可指定 --speech-device cuda:0。")

    def prepare_audio(self, source, directory):
        import numpy as np
        import soundfile as sf

        samples, sample_rate = sf.read(source, dtype="float32", always_2d=True)
        if not len(samples):
            raise ValueError("录音为空，请重新录音。")
        if len(samples) / sample_rate > self.args.max_audio_seconds:
            raise ValueError(f"每次录音请控制在 {self.args.max_audio_seconds} 秒内。")
        mono = samples.mean(axis=1)
        if not np.isfinite(mono).all() or np.max(np.abs(mono)) < 1e-5:
            raise ValueError("录音没有有效声音，请检查麦克风后重新录音。")
        path = directory / "question.wav"
        sf.write(path, mono, sample_rate, subtype="PCM_16")
        return path

    def prepare_knowledge(self, content, language, profile=None):
        """Load the SONAR encoders and cache the reference vectors for this worker."""
        import gc
        import torch
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from rag_module.sonar import SONAR_Wav_Embeddings

        encoder_cached = self.embedding is not None and self.embedding_language == language
        if not encoder_cached:
            self.embedding = None
            self.embedding_language = None
            self.index_key = None
            self.vectors = None
            self.vector_chunks = None
            gc.collect()
            with torch.cuda.device(self.speech_device):
                torch.cuda.empty_cache()
            LOG.info("Loading SONAR (%s) on %s", language, self.speech_device)
            self.embedding = SONAR_Wav_Embeddings(lan=language, device=self.speech_device)
            self.embedding_language = language
        if profile:
            profile("sonar_encoder_ready", cached=encoder_cached)
        key = hashlib.sha256((language + content).encode("utf-8")).hexdigest()
        if self.index_key != key:
            chunks = re.split(rf"\n\n(?={FAQ_QUESTION_PREFIX})", content.strip())
            faq_questions = [
                chunk.removeprefix(FAQ_QUESTION_PREFIX).partition("\n答案：")[0].strip()
                for chunk in chunks
                if chunk.startswith(FAQ_QUESTION_PREFIX) and "\n答案：" in chunk and len(chunk) <= 600
            ]
            if len(faq_questions) == len(chunks) and all(faq_questions):
                # Index each question phrasing separately, then return its full FAQ entry.
                questions_by_chunk = [
                    (index, re.sub(r"^\d+、\s*", "", line.strip()))
                    for index, question in enumerate(faq_questions)
                    for line in question.splitlines()
                    if line.strip()
                ]
                vector_chunks, index_texts = zip(*questions_by_chunk)
            else:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=600, chunk_overlap=100,
                    separators=["\n\n", "\n", "。", ". ", " ", ""],
                )
                chunks = splitter.split_text(content)
                vector_chunks, index_texts = None, chunks
            if not chunks:
                raise ValueError("知识库为空，请上传文件或粘贴文本。")
            vectors = []
            for start in range(0, len(index_texts), 8):
                vectors.append(self.embedding.embed_documents_tensor(index_texts[start:start + 8]))
            self.vectors = torch.cat(vectors).to(device=self.speech_device, dtype=torch.float32)
            self.vector_chunks = (torch.tensor(vector_chunks, dtype=torch.long, device=self.speech_device)
                                  if vector_chunks is not None else None)
            self.chunks = chunks
            self.index_key = key
            LOG.info("SONAR knowledge ready (%s, %s chunks) on %s", language, len(chunks), self.speech_device)
            if profile:
                profile("knowledge_vectors_ready", chunks=len(chunks))
        elif profile:
            profile("knowledge_vectors_ready", cached=True, chunks=len(self.chunks))

    def _rank_knowledge(self, query, metric):
        cosine = metric == "cosine"
        if cosine:
            from torch.nn.functional import cosine_similarity
            scores = cosine_similarity(self.vectors, query, dim=1)
        else:
            scores = (self.vectors - query).square().sum(dim=1)
        if self.vector_chunks is not None:
            # Keep the best score across all phrasings of each FAQ.
            initial = -float("inf") if cosine else float("inf")
            chunk_scores = scores.new_full((len(self.chunks),), initial)
            scores = chunk_scores.scatter_reduce_(0, self.vector_chunks, scores, reduce="amax" if cosine else "amin")
        return scores.argsort(descending=cosine, stable=True)[:1], scores

    def retrieve(self, question_path, content, language, profile=None):
        """Return the selected passages and scores from one SONAR speech embedding."""
        import torchaudio

        self.prepare_knowledge(content, language, profile)
        if profile:
            profile("sonar_query_start")
        query = self.embedding.embed_query_tensor(torchaudio.load(str(question_path))).to(self.vectors)
        if profile:
            profile("sonar_query_ready")
        metric = getattr(self.args, "retrieval_metric", "euclidean")
        metrics = [metric]
        if getattr(self.args, "compare_retrieval", False):
            metrics.append("euclidean" if metric == "cosine" else "cosine")
        retrieval = {"selected_metric": metric}
        for name in metrics:
            order, scores = self._rank_knowledge(query, name)
            retrieval[name] = [
                {"rank": rank, "score": score if math.isfinite(score) else None, "passage": self.chunks[index]}
                for rank, (index, score) in enumerate(zip(order.tolist(), scores[order].tolist()), 1)
            ]
        if profile:
            profile("knowledge_search_ready", matches=len(retrieval[metric]))
        return retrieval

    def prepare_input(self, question_path, input_ids, profile):
        return input_ids

    def generate_reply(self, question_path, input_ids, history, context, language,
                       cancel_event, on_audio, profile, stream_block_sizes):
        raise NotImplementedError

    def answer(self, source, use_rag, knowledge_text, knowledge_file, language, history, cancel_event=None,
               on_audio=None, profile=None, stream_block_sizes=STREAM_BLOCK_SIZES, *, input_ids=None,
               on_retrieval=None):
        if profile:
            profile("answer_start")
        import torch
        import torchaudio

        check_cancelled(cancel_event)
        if not source:
            raise ValueError("请先录音，或上传一段音频。")
        content = ""
        if use_rag:
            if knowledge_file:
                if Path(knowledge_file).stat().st_size > 1_000_000:
                    raise ValueError("知识库文件请控制在 1 MB 内。")
                content = Path(knowledge_file).read_text(encoding="utf-8-sig")
            else:
                content = knowledge_text or ""
            content = content.strip()
            if not content:
                raise ValueError("已启用知识库，请上传 TXT/Markdown 文件或粘贴文本。")
            if len(content) > 200_000:
                raise ValueError("知识库请控制在 20 万字以内。")
        started = time.monotonic()
        directory = Path(self.args.output_dir).resolve() / uuid.uuid4().hex
        directory.mkdir(parents=True)
        question_path = self.prepare_audio(source, directory)
        if profile:
            profile("audio_prepared")
        self.load()
        if profile:
            profile("models_ready")
        check_cancelled(cancel_event)
        with torch.inference_mode():
            input_ids = self.prepare_input(question_path, input_ids, profile)
            if profile:
                profile("sonar_start", enabled=use_rag)
            retrieval = self.retrieve(question_path, content, language, profile) if use_rag else None
            context = ""
            if retrieval is not None:
                selected = retrieval[retrieval["selected_metric"]]
                context = "\n\n".join(hit["passage"] for hit in selected)
            if profile:
                if use_rag:
                    profile("sonar_finished")
                else:
                    profile("knowledge_search_skipped", enabled=False)
                profile("rag_finished", enabled=use_rag)
            check_cancelled(cancel_event)
            previous = list(history or [])[-self.args.history_turns:] if self.args.history_turns else []
            if on_retrieval is not None and retrieval is not None:
                on_retrieval(retrieval)
            reply = self.generate_reply(
                question_path, input_ids, previous, context, language,
                cancel_event, on_audio, profile, stream_block_sizes,
            )
            check_cancelled(cancel_event)
            text = reply.text
            waveform = reply.waveform
            if not text and (waveform is None or not waveform.numel()):
                raise RuntimeError("模型没有生成有效回答，请重新提问。")
            output_audio = None
            if waveform is not None and waveform.numel():
                output_audio = str(directory / "answer.wav")
                torchaudio.save(output_audio, waveform.detach().float().cpu().reshape(1, -1),
                                self.output_sample_rate)
                if profile:
                    profile("answer_wav_saved")

        check_cancelled(cancel_event)

        (directory / "answer.txt").write_text(text, encoding="utf-8")
        (directory / "context.txt").write_text(context, encoding="utf-8")
        if retrieval is not None and getattr(self.args, "compare_retrieval", False):
            (directory / "retrieval_comparison.json").write_text(
                json.dumps(retrieval, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        record = {
            "question_tokens": reply.question_tokens, "assistant_tokens": reply.assistant_tokens, "text": text,
            "question_path": str(question_path), "answer_path": output_audio,
        }
        if reply.input_tokens is not None:
            record["input_tokens"] = reply.input_tokens
        if retrieval is not None:
            record["retrieval"] = retrieval
        updated = (list(history or []) + [record])[-20:]
        (directory / "turn.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        display = [((item["question_path"],), item["text"] or "（语音回答）") for item in updated]
        status = f"回答完成，用时 {time.monotonic() - started:.1f} 秒。"
        if not output_audio:
            status += "本次模型仅生成了文字。"
        if profile:
            profile("answer_ready")
        return display, output_audio, context, status, updated
