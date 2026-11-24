#!/usr/bin/env bash
set -euo pipefail

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
vllm_base_url=http://127.0.0.1:18000/v1
vllm_model=glm-voice
model_path=${MODEL:-}
if [[ $# -gt 0 && $1 != -* ]]; then
  model_path=${1:-$model_path}
  shift
fi
if [[ -z "$model_path" ]]; then
  if [[ ! -x "$project_dir/.venv/bin/python" ]]; then
    echo '请先创建项目的 Python 3.11 .venv 并完成环境安装。' >&2
    exit 1
  fi
  if ! model_path=$(cd "$project_dir" && .venv/bin/python - "$vllm_base_url" "$vllm_model" <<'PY'
from pathlib import Path
import sys

from voice_service.vllm import VllmTokenGenerator

try:
    root = VllmTokenGenerator(*sys.argv[1:]).model_card().get("root")
except RuntimeError as error:
    raise SystemExit(str(error)) from None
if not isinstance(root, str) or not Path(root).is_absolute() or not (Path(root) / "config.json").is_file():
    raise SystemExit("vLLM 返回的模型路径在本机不可用。")
print(root)
PY
  ); then
    echo '请确认 bash start_vllm.sh 已启动并加载模型，或运行 bash start_vllm_call.sh /path/to/model。' >&2
    exit 2
  fi
fi

exec env -u CONDA_PREFIX CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  bash "$project_dir/start_voice_call.sh" \
  --host 127.0.0.1 --port 7860 --speech-device cuda:0 \
  --vllm-base-url "$vllm_base_url" --vllm-model "$vllm_model" \
  --model-path "$model_path" --retrieval-metric cosine "$@"
