#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 && $1 != -* ]]; then
  model_path=$1
  shift
  set -- --model-path "$model_path" "$@"
elif [[ -n ${MODEL:-} ]]; then
  set -- --model-path "$MODEL" "$@"
fi

exec env -u CONDA_PREFIX CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}" \
  bash "$(dirname -- "$0")/start_voice_call.sh" \
  --backend glm --attn-implementation sdpa \
  --llm-device cuda:0 --speech-device cuda:0 --retrieval-metric cosine "$@"
