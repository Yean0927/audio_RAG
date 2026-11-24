"""Shared voice conversation, audio I/O and SONAR retrieval."""
from __future__ import annotations

from collections import Counter
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
import unicodedata
import uuid
import wave

from . import ROOT

if TYPE_CHECKING:
    from torch import Tensor

for directory in (ROOT / "src", ROOT / "third_party/Matcha-TTS"):
    sys.path.insert(0, str(directory))

LOG = logging.getLogger("voice-chat")
STREAM_BLOCK_SIZES = (25, 40, 48)
FAQ_QUESTION_PREFIX = "## 问题："
DEFAULT_OUTPUT_SAMPLE_RATE = 22050
TURN_CHECK_MAX_TOKENS = 32
CHINESE_FILLERS = frozenset("嗯恩唔呃额額啊哦噢喔哼欸诶誒唉嗚呜哈")
ENGLISH_FILLER = re.compile(r"h*m+|m+h+m+|u(?:h+m*|m+)|huh+|[aeo]h+|e+r+m*")
CHINESE_THANKS = r"(?:谢谢(?:你|您)?|多谢|(?:非常)?感谢(?:你|您)?)(?:了|啊|啦)?"
CHINESE_CLOSING = r"(?:我?(?:没有|没)(?:其他|其它|别的|更多|什么|问)?问题了?|再见|拜拜)"
COURTESY_PHRASE = re.compile(
    rf"(?:好(?:的)?|那)?(?:{CHINESE_THANKS}(?:{CHINESE_CLOSING})?|{CHINESE_CLOSING}(?:{CHINESE_THANKS})?|不用了{CHINESE_THANKS})"
    r"|(?:ok(?:ay)?)?(?:thanks|thankyou)(?:verymuch|somuch)?(?:goodbye|bye)?"
    r"|(?:ihave)?no(?:other|more|further)questions(?:thanks|thankyou)?|thatsall|imallset|goodbye|bye"
)

BACKENDS = {
    "glm": ("GLM-4-Voice", "zai-org/glm-4-voice-9b"),
    "omni-local": ("Qwen2.5-Omni", "Qwen/Qwen2.5-Omni-7B"),
    "omni": ("Qwen3-Omni API", "Qwen/Qwen3-Omni-30B-A3B-Instruct"),
    "ming": ("Ming-3.0 Audio API", "Ming-3.0-audio-flash"),
    "minicpmo": ("MiniCPM-o 4.5 Realtime API", "MiniCPM-o-4.5-Realtime"),
    "minicpmo-local": ("MiniCPM-o 4.5 (vLLM-Omni)", "openbmb/MiniCPM-o-4_5"),
    "minicpmo-gguf": ("MiniCPM-o 4.5 (llama.cpp-omni)", "openbmb/MiniCPM-o-4_5-gguf"),
}
API_BACKENDS = ("ming", "omni", "minicpmo", "minicpmo-local", "minicpmo-gguf")


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


def answer_instruction(context, language, history=(), *, clarify=False, exact_faq=False, courtesy=False):
    if courtesy:
        return (
            "听本轮录音，在以下两句里选一句原样说完。对方只致谢，回复‘不客气。’"
            "对方表示没有其他问题或道别，回复‘好的，感谢您的来电，祝您生活愉快。’说完立即停止。"
            if language == "zh" else
            "Listen to this turn and say exactly one of these sentences. For thanks, say 'You’re welcome.' "
            "If the caller is done or says goodbye, say 'Thank you for calling, and have a good day.' Then stop. "
        )
    if clarify:
        return (
            "本轮需要确认用户的需求。根据本轮实际听到的内容和已发生的对话，"
            "只问一句简短、自然的中文追问，优先确认最影响后续回答的信息。"
            "在追问里带上已经听清的诉求或对象，方便用户直接补充缺少的那一项。"
            "提问后停下，等待用户补充。只输出直接对用户说的话。"
            if language == "zh" else
            "The caller's request needs clarification. Based on the current audio and prior dialogue, "
            "ask one short, natural question about the detail that matters most for answering. "
            "Include the relevant part you already understood so the caller can supply the missing detail. "
            "Then wait for the caller. Output only the question addressed to the caller. "
        )
    if context and exact_faq:
        return (
            "你是电话客服。先听完整录音，判断用户是在直接询问 FAQ 还是在补充这项业务的办理资料。"
            "直接提问时，从参考 FAQ 的答案正文开始，沿用其措辞和顺序复述相关内容。"
            "把每条条件和对应的操作一起说清楚，保留关键步骤、数字及联系方式。只说 FAQ 支持的业务事实。"
            "用户已经提供本项业务的办理资料时，承接已给的信息，只问下一项缺失资料。"
            "资料已齐全时简短确认收到。只有实际完成办理后才能告知办理结果。只说直接对用户说的话。"
            if language == "zh" else
            "You are a phone support agent. Listen to the entire clip and determine whether the caller is asking the FAQ directly "
            "or supplying details for this request. For a direct question, start with the FAQ answer itself, following its wording and order. "
            "State each condition together with its action, preserving important steps, numbers, and contact details. Use only the business facts in the FAQ. "
            "When the caller supplies details for this request, acknowledge them and ask only for the next missing detail. "
            "If the needed details are complete, briefly confirm receipt. Mention completed actions only when they actually happened. "
            "Speak directly to the caller. "
        )
    instruction = (
        "你是一位语音助手。请用中文简洁、自然地回答，通常两到三句话即可。"
        if language == "zh" else
        "You are a voice assistant. Answer naturally and briefly, usually in two or three sentences. "
    )
    instruction += (
        "结合前几轮对话理解省略和指代，连续追问应延续相关话题；用户明确换话题时以新问题为准。"
        "先理解本轮是在提问、补充信息、确认、纠正还是结束对话，再自然回应。"
        "如果听清了话但仍无法确定用户想问什么或指哪件事，先只问一个具体的澄清问题，等对方回答。"
        "如果你上一轮索要了资料，本轮应先承接用户提供的内容，结合多轮对话核对，已提供的信息不重复索取。"
        "信息不全时只追问尚缺的一项；信息齐全时简短确认收到，例如‘好的，收到您提供的信息了’，"
        "再根据资料说明必要的下一步，不重新介绍需求或重念信息清单。用户纠正信息时以最新说法为准，拒绝提供时不要反复索要。"
        "确认收到不等于已登记、已提交或已安排联系；没有实际办理结果时不能声称已完成这些操作。"
        "历史参考资料是检索结果，不是用户原话；仅使用与当前问题相关的资料，不能把无关检索结果当作用户意图。"
        if language == "zh" else
        "Use conversation history to resolve follow-up questions and references. Follow a new topic when the caller changes it. "
        "First understand whether the caller is asking, supplying information, confirming, correcting, or ending the conversation. "
        "If the request or its subject is still unclear, ask one focused clarifying question and wait for the caller. "
        "If you requested details, acknowledge what the caller now supplies and combine details across turns. "
        "Ask only for one still-missing detail; never request details already supplied. When all requested details are present, "
        "briefly acknowledge receipt and explain any necessary next step supported by the reference, without repeating the checklist. "
        "Use the caller's latest correction and respect a refusal to provide information. "
        "Acknowledging receipt does not mean a record was saved, a request submitted, or a callback arranged. "
        "Do not claim those actions without an actual result. "
        "Historical reference material is retrieved information, not a transcript of the caller. "
        "Use only relevant material; unrelated retrieval results do not establish the caller's intent. "
    )
    if context:
        instruction += (
            "请听清本轮语音提问，从参考资料中选取相关事实，用自己的话回答。"
            "只输出直接对来电用户说的内容。参考资料中的答案和信息清单是业务依据，不是每轮都要照读的话术。"
            "条件、流程和所需信息仅在与本轮相关时说明，已说明的内容不因再次检索命中就重复。"
            "资料不足时请说明无法确认。业务开放情况以资料明确记载为准。"
            if language == "zh" else
            "Answer the current spoken question in your own words using relevant facts from the reference material. "
            "Output only what you would say directly to the caller. Reference answers and checklists are business facts, "
            "not scripts to repeat each turn. Explain conditions, steps, and required details when relevant to this turn; "
            "a repeated retrieval match is not a reason to repeat an explanation. "
            "Say when the available information is insufficient. "
        )
    if history and history[-1].get("text"):
        label = "上一轮客服实际回复（用于理解本轮衔接，不要复述）" if language == "zh" else (
            "Previous assistant reply (use to interpret this turn; do not repeat it)"
        )
        instruction += f"\n[{label}]\n{history[-1]['text']}\n[/{label}]\n"
    return instruction


def history_reference(turn):
    """Recover the selected FAQ from saved turns, including existing sessions."""
    retrieval = turn.get("retrieval") or {}
    selected = retrieval.get(retrieval.get("selected_metric"), [])
    context = selected[0].get("passage", "") if selected else ""
    if not context:
        return ""
    return f"[Historical reference material (not a transcript)]\n{context}\n[/Historical reference material]\n"


def turn_instruction(language, previous_reply=""):
    instruction = (
        "听当前录音，只输出一行判定。只引用本轮用户实际说的话。上一轮客服回复仅用于理解指代，不能当作本轮听写。"
        "只有整段录音只是致谢、告别或表达没有更多问题时，输出 5:，后接实际听到的话。"
        "如果致谢后还有提问或补充，按后面的业务内容判断并引用那部分原话。"
        "用户说完了明确的提问、请求或可回应的话，输出 1:，后接本轮实际听到的几个字词。"
        "陈述式请求也算。承接上一轮的简短确认、拒绝、纠正、补充信息或追问也算。"
        "话说到一半或被截断，输出 2:，后接本轮实际听到的几个字词。"
        "用户说完了，能听清是在向客服说话，但需求、指代对象或答复所需的关键信息不明确，"
        "输出 4:，后接本轮实际听到的几个字词。例如上下文没有明确对象时，‘这个怎么处理’属于这一类。"
        "咳嗽、哼唱、环境噪音、回声、旁人说话、仅有语气词或无法听清的话，只输出 3。"
        "逐字引用原语言，禁止猜补、描述声音或回答问题。"
        if language == "zh" else
        "Listen to the current clip and output one line. Quote only the caller's words in this clip. The previous assistant reply is for resolving references only. "
        "Output 5: plus the words heard only when the whole utterance is thanks, a goodbye, or saying there are no further questions. "
        "If thanks are followed by a question or new details, classify and quote that substantive part. "
        "For a complete, clear question, request, or other contribution that warrants a response, output 1: "
        "followed by a few of those words verbatim. A short confirmation, refusal, correction, supplied detail, "
        "or follow-up counts when it follows from the preceding assistant reply. "
        "For an unfinished or cut-off utterance, output 2: followed by a few of the words heard verbatim. "
        "If the caller has finished speaking to you but their request, its subject, or a detail essential to answering is unclear, "
        "output 4: followed by a few of the words heard verbatim. "
        "For coughing, humming, noise, echo, background conversation, filler alone, or unintelligible speech, output only 3. "
        "Use the original language. "
        "Do not guess, describe sounds, or answer the caller."
    )
    if previous_reply:
        label = "上一轮客服回复（仅用于理解本轮衔接）" if language == "zh" else (
            "Previous assistant reply (only for interpreting the current clip)"
        )
        instruction = f"[{label}]\n{previous_reply}\n[/{label}]\n\n{instruction}"
    return instruction


def normalize_spoken_text(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(char for char in text if char.isalnum())


def normalize_faq_request(text):
    words = normalize_spoken_text(text)
    return re.sub(r"^(?:请问)?(?:我想(?:了解|咨询|问|要)?|我要|我需要)?", "", words, count=1)


def faq_phrasings(passage):
    if not passage.startswith(FAQ_QUESTION_PREFIX) or len(passage) > 600:
        return []
    question, separator, _ = passage.removeprefix(FAQ_QUESTION_PREFIX).partition("\n答案：")
    if not separator:
        return []
    return [re.sub(r"^\d+、\s*", "", line.strip()) for line in question.splitlines() if line.strip()]


@dataclass(frozen=True)
class TurnDecision:
    action: str
    faq_request: str = ""


def turn_decision(text, faq_requests=(), *, previous_reply=""):
    match = re.fullmatch(r"([1245])\s*[:：]\s*([^\r\n]+)", text.strip())
    if match:
        spoken = re.sub(r"\[[^\]]*\]|\([^)]*\)|【[^】]*】|（[^）]*）|<[^>]*>", "", match[2])
        # Keep English words together and check Chinese characters individually.
        words = re.findall(r"[a-z]+|[^\W_a-z]", spoken.casefold())
        if any(word not in CHINESE_FILLERS and not ENGLISH_FILLER.fullmatch(word) for word in words):
            action = {"1": "reply", "2": "wait", "4": "clarify", "5": "courtesy"}[match[1]]
            if action == "wait":
                return TurnDecision(action)
            evidence = normalize_spoken_text(spoken)
            if COURTESY_PHRASE.fullmatch(evidence):
                return TurnDecision("courtesy")
            if previous_reply and evidence == normalize_spoken_text(previous_reply):
                return TurnDecision("echo")
            request = normalize_faq_request(spoken)
            # A complete request for a known service can proceed even when details are missing.
            if request in faq_requests:
                return TurnDecision("reply", request)
            return TurnDecision(action)
    return TurnDecision("discard")


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
        self.faq_requests = frozenset()
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
            self.faq_requests = frozenset()
            gc.collect()
            if self.speech_device.type == "cuda":
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
            phrasings_by_chunk = [faq_phrasings(chunk) for chunk in chunks]
            faq_requests = frozenset()
            if all(phrasings_by_chunk):
                # Index each question phrasing separately, then return its full FAQ entry.
                questions_by_chunk = [
                    (index, question)
                    for index, phrasings in enumerate(phrasings_by_chunk)
                    for question in phrasings
                ]
                vector_chunks, index_texts = zip(*questions_by_chunk)
                request_counts = Counter(
                    request for phrasings in phrasings_by_chunk for request in set(map(normalize_faq_request, phrasings))
                )
                faq_requests = frozenset(request for request, count in request_counts.items() if request and count == 1)
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
            self.faq_requests = faq_requests
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
                       cancel_event, on_audio, profile, stream_block_sizes, *, clarify=False, exact_faq=False,
                       courtesy=False):
        raise NotImplementedError

    def answer(self, source, use_rag, knowledge_text, knowledge_file, language, history, cancel_event=None,
               on_audio=None, profile=None, stream_block_sizes=STREAM_BLOCK_SIZES, *, input_ids=None,
               on_retrieval=None, clarify=False, faq_request="", courtesy=False):
        if profile:
            profile("answer_start")
        import torch
        import torchaudio

        check_cancelled(cancel_event)
        if not source:
            raise ValueError("请先录音，或上传一段音频。")
        use_rag = use_rag and not (clarify or courtesy)
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
                context = selected[0]["passage"] if selected else ""
            exact_faq = bool(faq_request and any(
                normalize_faq_request(question) == faq_request for question in faq_phrasings(context)
            ))
            if profile:
                if use_rag:
                    profile("sonar_finished")
                else:
                    profile("knowledge_search_skipped", enabled=False)
                profile("rag_finished", enabled=use_rag)
            check_cancelled(cancel_event)
            previous = list(history or [])[-self.args.history_turns:] if self.args.history_turns else []
            if courtesy:
                previous = []
            elif clarify:
                previous = [{key: value for key, value in turn.items() if key != "retrieval"} for turn in previous]
            if on_retrieval is not None and retrieval is not None:
                on_retrieval(retrieval)
            reply = self.generate_reply(
                question_path, input_ids, previous, context, language,
                cancel_event, on_audio, profile, stream_block_sizes, clarify=clarify, exact_faq=exact_faq,
                courtesy=courtesy,
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
        if exact_faq:
            record["exact_faq"] = True
        updated = (list(history or []) + [record])[-20:]
        (directory / "turn.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        display = [((item["question_path"],), item["text"] or "（语音回答）") for item in updated]
        status = f"回答完成，用时 {time.monotonic() - started:.1f} 秒。"
        if not output_audio:
            status += "本次模型仅生成了文字。"
        if profile:
            profile("answer_ready")
        return display, output_audio, context, status, updated
