#!/usr/bin/env bash
set -euo pipefail
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
if [[ ! -x .venv/bin/python ]]; then
  echo "请先创建项目的 Python 3.11 .venv 并完成环境安装。" >&2
  exit 1
fi
if [[ ${HF_HUB_OFFLINE:-} != 1 && ${TRANSFORMERS_OFFLINE:-} != 1 ]]; then
  set -- --download-models "$@"
fi
exec .venv/bin/python voice_call_server.py "$@"
