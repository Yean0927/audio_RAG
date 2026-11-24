"""Prefetch the voice-call model files with terminal download progress.

Run with the same user and virtual environment as start_voice_call.sh.
Hugging Face snapshots use the existing default cache; the decoder uses the
project folder. SONAR pipelines briefly load on CPU to populate their own cache.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys

from voice_service import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-sonar", action="store_true", help="Only download voice-call models (no knowledge retrieval)")
    parser.add_argument("--languages", nargs="+", choices=["zh", "en"], default=["zh", "en"], help="SONAR speech languages; default: zh en")
    args = parser.parse_args()

    # Set these before importing Hub/Transformers, which read flags at import.
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[key] = "0"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import enable_progress_bars

    enable_progress_bars()
    stages = 3 + (0 if args.skip_sonar else 1 + len(dict.fromkeys(args.languages)))
    current = 0

    def stage(label):
        nonlocal current
        current += 1
        print(f"\n[{current}/{stages}] {label}", flush=True)

    print("下载到当前用户已有缓存，不加载 GLM 到显卡。", flush=True)
    print("每个文件显示下载进度；获取远程文件列表时可能暂时没有进度条。", flush=True)
    print("请先停止正在自动下载的网页服务，避免同时写入模型目录。", flush=True)

    decoder_dir = ROOT / "glm-4-voice-decoder"
    stage("语音 Decoder：config.yaml / flow.pt / hift.pt")
    snapshot_download(
        repo_id="zai-org/glm-4-voice-decoder",
        local_dir=str(decoder_dir),
        allow_patterns=["config.yaml", "flow.pt", "hift.pt"],
        max_workers=1,
    )
    for filename in ("config.yaml", "flow.pt", "hift.pt"):
        path = decoder_dir / filename
        if not path.is_file() or path.stat().st_size < 64:
            raise RuntimeError(f"缺少或无效文件：{path}")
        with path.open("rb") as stream:
            if stream.read(80).startswith(b"version https://git-lfs"):
                raise RuntimeError(f"文件是 Git LFS 指针，不是真实权重：{path}")
    print(f"Decoder 已保存：{decoder_dir}", flush=True)

    for repo, label in (
        ("zai-org/glm-4-voice-9b", "GLM-4-Voice-9B 主模型"),
        ("zai-org/glm-4-voice-tokenizer", "语音 Tokenizer"),
    ):
        stage(label)
        path = snapshot_download(repo_id=repo, max_workers=1)
        print(f"已缓存：{path}", flush=True)

    if not args.skip_sonar:
        import torch
        from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
        from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

        stage("SONAR 文本编码器及分词器（CPU 加载校验，需要系统内存）")
        pipeline = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
            device=torch.device("cpu"),
        )
        del pipeline
        gc.collect()
        for language in dict.fromkeys(args.languages):
            encoder = "sonar_speech_encoder_cmn" if language == "zh" else "sonar_speech_encoder_eng"
            stage(f"SONAR {language} 语音编码器（CPU 加载校验）")
            pipeline = SpeechToEmbeddingModelPipeline(encoder=encoder, device=torch.device("cpu"))
            del pipeline
            gc.collect()

    print("\n所选模型下载完成。可以重新运行 bash start_voice_call.sh。", flush=True)
    print("启动时仍需加载模型到显存；下载完成不等于 GPU 推理已验证。", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n下载已停止。重新运行可复用已完成的文件。", file=sys.stderr)
        raise SystemExit(130)
    except Exception:
        print("\n下载未全部完成，请保留下方错误。修复后可重新运行，已有文件会复用。", file=sys.stderr)
        raise
