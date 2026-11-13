"""Browser microphone chat for GLM-4-Voice, with optional SONAR retrieval.

Run from the server's project directory: python voice_web_demo.py
Models load on the first request or when the load button is pressed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
for directory in (ROOT / "src", ROOT / "third_party/Matcha-TTS"):
    sys.path.insert(0, str(directory))

LOG = logging.getLogger("voice-chat")
INTERLEAVED = (
    "User will provide you with a speech instruction. Respond in an interleaved "
    "manner, with 13 text tokens followed by 26 audio tokens. "
)


def make_prompt(history, question, context, language):
    instruction = (
        "你是一位语音助手。请用中文简洁、自然地回答，通常两到三句话即可。"
        if language == "zh" else
        "You are a voice assistant. Answer naturally and briefly, usually in two or three sentences. "
    )
    if context:
        instruction += (
            "参考资料仅作为回答问题的依据。资料不足时请明确说明，不要编造。"
            if language == "zh" else
            "Use the reference material as evidence. Say when it is insufficient. "
        )
    prompt = f"<|system|>\n{INTERLEAVED}{instruction}"
    for turn in history:
        prompt += f"<|user|>\n{turn['question_tokens']}<|assistant|>streaming_transcription\n{turn['assistant_tokens']}"
    current = question
    if context:
        current = f"Reference material:\n{context}\n\nQuestion: {question}"
    return f"{prompt}<|user|>\n{current}<|assistant|>streaming_transcription\n"


def clean_assistant(raw):
    for marker in ("<|user|>", "<|endoftext|>"):
        raw = raw.split(marker, 1)[0]
    return raw.strip()


class VoiceBackend:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.ready = False
        self.embedding = None
        self.embedding_language = None
        self.index_key = None
        self.chunks = []
        self.vectors = None

    def load(self):
        """Caller holds the lock, including during lazy model initialization."""
        if self.ready:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, WhisperFeatureExtractor
        from speech_tokenizer.modeling_whisper import WhisperVQEncoder
        from spoken_chatbot.model import AudioDecoder

        decoder_dir = Path(self.args.flow_path).resolve()
        missing = [name for name in ("config.yaml", "flow.pt", "hift.pt") if not (decoder_dir / name).is_file()]
        if missing:
            raise RuntimeError(
                f"语音 Decoder 缺少文件：{', '.join(missing)}。请先下载到 {decoder_dir}，或指定 --flow-path。"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("没有检测到可用的 CUDA 显卡，请检查当前 .venv。")
        self.llm_device = torch.device(self.args.llm_device)
        self.speech_device = torch.device(self.args.speech_device)
        for device in (self.llm_device, self.speech_device):
            if device.type != "cuda" or device.index is None or device.index >= torch.cuda.device_count():
                raise RuntimeError(f"显卡 {device} 不可用；单卡可指定 --speech-device cuda:0。")

        LOG.info("Loading language model (%s) on %s", self.args.quantization, self.llm_device)
        kwargs = dict(
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map={"": self.llm_device.index},
            low_cpu_mem_usage=True,
        )
        if self.args.quantization == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        # Keep partially loaded objects local so failed initialization can be retried.
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(self.args.model_path, **kwargs).eval()
        LOG.info("Loading speech tokenizer and decoder on %s", self.speech_device)
        whisper = WhisperVQEncoder.from_pretrained(self.args.tokenizer_path).eval().to(self.speech_device)
        feature_extractor = WhisperFeatureExtractor.from_pretrained(self.args.tokenizer_path)
        decoder = AudioDecoder(
            config_path=str(decoder_dir / "config.yaml"),
            flow_ckpt_path=str(decoder_dir / "flow.pt"),
            hift_ckpt_path=str(decoder_dir / "hift.pt"),
            device=self.speech_device,
        )
        decoder.flow.eval()
        decoder.hift.eval()
        self.tokenizer, self.model = tokenizer, model
        self.whisper, self.feature_extractor, self.decoder = whisper, feature_extractor, decoder
        self.ready = True
        LOG.info("Voice models ready")

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

    def retrieve(self, question_path, content, language):
        """SONAR speech/text embeddings; squared-L2 ranking, as in Chroma's default."""
        import gc
        import numpy as np
        import torch
        import torchaudio
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from rag_module.sonar import SONAR_Wav_Embeddings

        if self.embedding_language != language:
            self.embedding = None
            self.embedding_language = None
            self.index_key = None
            self.vectors = None
            gc.collect()
            with torch.cuda.device(self.speech_device):
                torch.cuda.empty_cache()
            LOG.info("Loading SONAR (%s) on %s", language, self.speech_device)
            self.embedding = SONAR_Wav_Embeddings(lan=language, device=self.speech_device)
            self.embedding_language = language
        key = hashlib.sha256((language + content).encode("utf-8")).hexdigest()
        if self.index_key != key:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=600, chunk_overlap=100,
                separators=["\n\n", "\n", "。", ". ", " ", ""],
            )
            chunks = splitter.split_text(content)
            if not chunks:
                raise ValueError("知识库为空，请上传文件或粘贴文本。")
            vectors = []
            for start in range(0, len(chunks), 8):
                vectors.extend(self.embedding.embed_documents(chunks[start:start + 8]))
            self.chunks = chunks
            self.vectors = np.asarray(vectors, dtype=np.float32)
            self.index_key = key
        query = self.embedding.embed_query(torchaudio.load(str(question_path)))
        distances = np.sum((self.vectors - np.asarray(query, dtype=np.float32)) ** 2, axis=1)
        indices = np.argsort(distances)[:4]
        return "\n\n".join(self.chunks[int(i)] for i in indices)

    def answer(self, source, use_rag, knowledge_text, knowledge_file, language, history):
        import torch
        import torchaudio
        from speech_tokenizer.utils import extract_speech_token

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
        self.load()
        with torch.inference_mode():
            # The repository helper defaults to cuda:0: explicitly pass the speech GPU.
            tokens = extract_speech_token(
                self.whisper, self.feature_extractor, [str(question_path)], device=str(self.speech_device)
            )[0]
            if not tokens:
                raise ValueError("没有提取到语音内容，请重新录一段清晰的声音。")
            question = "<|begin_of_audio|>" + "".join(f"<|audio_{i}|>" for i in tokens) + "<|end_of_audio|>"
            context = self.retrieve(question_path, content, language) if use_rag else ""
            previous = list(history or [])[-self.args.history_turns:] if self.args.history_turns else []
            limit = min(getattr(self.model.config, "seq_length", 8192), 8192)
            while True:
                prompt = make_prompt(previous, question, context, language)
                inputs = self.tokenizer([prompt], return_tensors="pt").to(self.llm_device)
                if inputs.input_ids.shape[1] + self.args.max_new_tokens <= limit:
                    break
                if not previous:
                    raise ValueError("当前问题和参考资料过长，请缩短录音或知识库片段。")
                previous.pop(0)
            response = self.model.generate(
                **inputs, max_new_tokens=self.args.max_new_tokens,
                do_sample=True, temperature=0.2, top_p=0.8,
            )
            # Slice by input length instead of searching for a prompt marker in history.
            generated = response[0, inputs.input_ids.shape[1]:].tolist()
            audio_offset = self.tokenizer.convert_tokens_to_ids("<|audio_0|>")
            audio_ids = [i - audio_offset for i in generated if i >= audio_offset]
            text_ids = [i for i in generated if i < audio_offset]
            text = self.tokenizer.decode(text_ids, skip_special_tokens=True).strip()
            raw = clean_assistant(self.tokenizer.decode(generated, skip_special_tokens=False))
            if not text and not audio_ids:
                raise RuntimeError("模型没有生成有效回答，请重新提问。")
            output_audio = None
            if audio_ids:
                waveform, _ = self.decoder.token2wav(
                    torch.tensor(audio_ids, dtype=torch.long, device=self.speech_device).unsqueeze(0),
                    uuid=uuid.uuid4().hex, finalize=True,
                )
                output_audio = str(directory / "answer.wav")
                torchaudio.save(output_audio, waveform.detach().float().reshape(1, -1).cpu(), 22050)

        (directory / "answer.txt").write_text(text, encoding="utf-8")
        (directory / "context.txt").write_text(context, encoding="utf-8")
        record = {
            "question_tokens": question, "assistant_tokens": raw, "text": text,
            "question_path": str(question_path), "answer_path": output_audio,
        }
        updated = (list(history or []) + [record])[-20:]
        (directory / "turn.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        display = [((item["question_path"],), item["text"] or "（语音回答）") for item in updated]
        status = f"回答完成，用时 {time.monotonic() - started:.1f} 秒。"
        if not output_audio:
            status += "本次模型仅生成了文字。"
        return display, output_audio, context, status, updated


def build_ui(args, backend=None):
    import gradio as gr
    backend = backend or VoiceBackend(args)

    def load_models():
        try:
            with backend.lock:
                backend.load()
            return "语音模型已就绪，可以录音提问。"
        except Exception as error:
            LOG.exception("Model initialization failed")
            raise gr.Error(str(error)) from error

    def answer(*values):
        try:
            with backend.lock:
                return backend.answer(*values)
        except Exception as error:
            LOG.exception("Voice request failed")
            raise gr.Error(str(error)) from error

    with gr.Blocks(title="GLM 语音对话", analytics_enabled=False) as app:
        gr.Markdown("# GLM 语音对话\n录音结束后点击「发送语音」，查看文字回答并播放回复。支持连续多轮提问。")
        state = gr.State([])
        with gr.Row():
            with gr.Column(scale=1):
                audio = gr.Audio(sources=["microphone", "upload"], type="filepath", format="wav", label="录音或上传音频")
                with gr.Row():
                    send = gr.Button("发送语音", variant="primary")
                    clear = gr.Button("开始新对话")
                language = gr.Radio(choices=[("中文", "zh"), ("English", "en")], value="zh", label="对话语言")
                use_rag = gr.Checkbox(label="使用知识库回答", value=False)
                with gr.Accordion("知识库（可选）", open=False):
                    knowledge_file = gr.File(file_types=[".txt", ".md"], type="filepath", label="上传 UTF-8 文本文件")
                    knowledge = gr.Textbox(lines=6, label="或直接粘贴知识库内容", placeholder="开启知识库后，上传的文件优先于此处文本。")
                load = gr.Button("提前加载模型")
                status = gr.Textbox(value="等待录音。首次加载模型可能需要下载权重，请同时查看服务器终端。", label="状态", interactive=False)
            with gr.Column(scale=2):
                chat = gr.Chatbot(label="对话记录", height=400)
                reply_audio = gr.Audio(label="本轮语音回答", type="filepath", interactive=False)
                with gr.Accordion("本轮检索到的参考内容", open=False):
                    context = gr.Textbox(lines=8, label="参考内容", interactive=False)
        send.click(
            answer, [audio, use_rag, knowledge, knowledge_file, language, state],
            [chat, reply_audio, context, status, state], concurrency_limit=1, concurrency_id="voice-model",
            api_name="chat",
        )
        load.click(load_models, outputs=status, concurrency_limit=1, concurrency_id="voice-model", api_name="load_models")
        clear.click(
            lambda: ([], None, "", "已开始新对话。", []),
            outputs=[chat, reply_audio, context, status, state],
            concurrency_limit=1, concurrency_id="voice-model",
            api_name="new_chat",
        )
    return app


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--model-path", default="zai-org/glm-4-voice-9b")
    parser.add_argument("--tokenizer-path", default="zai-org/glm-4-voice-tokenizer")
    parser.add_argument("--flow-path", default=str(ROOT / "glm-4-voice-decoder"))
    parser.add_argument("--quantization", choices=["4bit", "bf16"], default="4bit")
    parser.add_argument("--llm-device", default="cuda:0")
    parser.add_argument("--speech-device", default="cuda:1")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--history-turns", type=int, default=3)
    parser.add_argument("--max-audio-seconds", type=int, default=30)
    parser.add_argument("--output-dir", default=str(ROOT / "answer_data/web"))
    args = parser.parse_args(argv)
    if not 64 <= args.max_new_tokens <= 2048 or not 0 <= args.history_turns <= 10:
        parser.error("--max-new-tokens must be 64–2048; --history-turns must be 0–10")
    if not 1 <= args.max_audio_seconds <= 60:
        parser.error("--max-audio-seconds must be 1–60")
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.chdir(ROOT)  # Decoder YAML and Matcha imports contain repository-relative paths.
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    app = build_ui(args)
    app.queue(default_concurrency_limit=1, max_size=8).launch(
        server_name=args.host, server_port=args.port, share=False, show_error=True,
        allowed_paths=[str(Path(args.output_dir).resolve())],
    )


if __name__ == "__main__":
    main()
