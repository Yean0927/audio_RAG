#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo '用法：bash start_vllm.sh [模型快照目录]' >&2
  exit 2
fi

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
vllm_bin=${VLLM_DATA_DIR:-$project_dir}/.venv-vllm-cu128/bin
if [[ ! -x "$vllm_bin/vllm" ]]; then
  printf '找不到 %s/vllm，请在项目根目录运行 python3.12 -m tools.install_vllm_cu128。\n' "$vllm_bin" >&2
  exit 1
fi

model_path=${1:-${MODEL:-}}
if [[ -z "$model_path" ]]; then
  model_path=$("$vllm_bin/python" - <<'PY'
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

try:
    print(snapshot_download("zai-org/glm-4-voice-9b", local_files_only=True))
except LocalEntryNotFoundError:
    raise SystemExit("本地缓存中没有 GLM-4-Voice-9B，请运行 bash start_vllm.sh /path/to/model。") from None
PY
  )
fi
if [[ ! -f "$model_path/config.json" ]]; then
  printf '模型目录缺少 config.json：%s\n' "$model_path" >&2
  exit 2
fi
model_path=$(CDPATH= cd -- "$model_path" && pwd)

exec env -u CONDA_PREFIX \
  VLLM_USE_MODELSCOPE=False HF_HUB_OFFLINE=1 \
  HF_MODULES_CACHE=/var/tmp/glm-vllm-hf-modules \
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  "$vllm_bin/vllm" serve "$model_path" \
  --host 127.0.0.1 --port 18000 --served-model-name glm-voice \
  --trust-remote-code --dtype bfloat16 \
  --quantization bitsandbytes --load-format bitsandbytes \
  --gpu-memory-utilization 0.50 --kv-cache-memory-bytes 2147483648 \
  --max-model-len 8192 --max-num-seqs 1 \
  --enforce-eager --skip-tokenizer-init
