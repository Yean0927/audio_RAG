"""Browser microphone chat for speech-to-speech models, with optional SONAR retrieval.

Run from the server's project directory: python voice_web_demo.py
Models load on the first request or when the load button is pressed.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from voice_service import ROOT
from voice_service.backend import (
    API_BACKENDS, BACKENDS, LOG, STREAM_BLOCK_SIZES as STREAM_BLOCK_SIZES,
    InferenceCancelled as InferenceCancelled, check_cancelled as check_cancelled,
)
from voice_service.cli import create_backend, create_parser as create_backend_parser
from voice_service.cli import validate_args as validate_args
from voice_service.glm import (
    GLM_TOKENIZER_REPO as GLM_TOKENIZER_REPO, VoiceBackend as VoiceBackend,
    make_prompt as make_prompt, resolve_glm_tokenizer_path as resolve_glm_tokenizer_path,
)

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")


def build_ui(args, backend=None):
    import gradio as gr
    backend = backend or create_backend(args)

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

    model_name = BACKENDS[getattr(args, "backend", "glm")][0]
    with gr.Blocks(title=f"{model_name} 语音对话", analytics_enabled=False) as app:
        gr.Markdown(f"# {model_name} 语音对话\n录音结束后点击「发送语音」，查看文字回答并播放回复。支持连续多轮提问。")
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
                status_hint = ("等待录音。音频和匹配的知识库片段将发送到模型 API。"
                               if args.backend in API_BACKENDS else
                               "等待录音。首次加载模型可能需要下载权重，请同时查看服务器终端。")
                status = gr.Textbox(value=status_hint, label="状态", interactive=False)
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


def create_parser():
    return create_backend_parser(__doc__)


def parse_args(argv=None):
    parser = create_parser()
    return validate_args(parser.parse_args(argv), parser)


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
