#!/usr/bin/env bash
# 启动已有 GGUF 容器和通话服务；环境变量或下方命令行参数均可修改默认值。
set -euo pipefail
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

container="${MINICPMO_CONTAINER:-minicpmo-gguf}"
base_url="${MINICPMO_GGUF_BASE_URL:-http://127.0.0.1:8006}"
port="${CALL_PORT:-7860}"
max_calls="${MAX_CALLS:-5}"
max_tokens="${MAX_NEW_TOKENS:-512}"
reference="${MINICPMO_REF_AUDIO:-}"
speed="${MINICPMO_SPEED:-1.0}"
stream="${MINICPMO_STREAM:-true}"
dry_run=false
extra=()

usage() {
  cat <<'EOF'
用法：bash start_minicpmo_gguf_call.sh [选项] [其他通话服务参数]
  --container NAME        已有容器，默认 minicpmo-gguf
  --base-url URL          GGUF 地址，默认 http://127.0.0.1:8006
  --port PORT             通话端口，默认 7860
  --max-calls N           接入上限，默认 5（单个 GGUF 后端仍串行推理）
  --max-new-tokens N      回复最大 token 数，默认 512
  --ref-audio WAV         音色参考 WAV，通话主机路径，最长 30 秒；省略使用默认音色
  --speed RATE            语速倍率 0.5–2，默认 1；非原速需要主机安装 ffmpeg
  --stream / --no-stream  分块出音 / 整轮出音，默认 --stream
  --dry-run              只打印命令，不启动容器或服务
  -h, --help             显示帮助

环境变量：MINICPMO_CONTAINER、MINICPMO_GGUF_BASE_URL、CALL_PORT、MAX_CALLS、
MAX_NEW_TOKENS、MINICPMO_REF_AUDIO、MINICPMO_SPEED、MINICPMO_STREAM（true/false）。
例如：bash start_minicpmo_gguf_call.sh --speed 1.15 --ref-audio resources/voice.wav
EOF
}

while (($#)); do
  case "$1" in
    --container=*|--base-url=*|--minicpmo-gguf-base-url=*|--port=*|--max-calls=*|--max-new-tokens=*|--ref-audio=*|--minicpmo-ref-audio=*|--speed=*|--minicpmo-speed=*)
      set -- "${1%%=*}" "${1#*=}" "${@:2}"
      continue ;;
    --container|--base-url|--minicpmo-gguf-base-url|--port|--max-calls|--max-new-tokens|--ref-audio|--minicpmo-ref-audio|--speed|--minicpmo-speed)
      if (($# < 2)) || [[ -z "$2" || "$2" == --* ]]; then
        echo "$1 缺少参数值。" >&2
        exit 2
      fi
      case "$1" in
        --container) container="$2" ;;
        --base-url|--minicpmo-gguf-base-url) base_url="$2" ;;
        --port) port="$2" ;;
        --max-calls) max_calls="$2" ;;
        --max-new-tokens) max_tokens="$2" ;;
        --ref-audio|--minicpmo-ref-audio) reference="$2" ;;
        --speed|--minicpmo-speed) speed="$2" ;;
      esac
      shift 2 ;;
    --stream|--minicpmo-stream) stream=true; shift ;;
    --no-stream|--no-minicpmo-stream) stream=false; shift ;;
    --dry-run) dry_run=true; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; extra+=("$@"); break ;;
    *) extra+=("$1"); shift ;;
  esac
done

case "$stream" in
  true|1) stream_option=--minicpmo-stream ;;
  false|0) stream_option=--no-minicpmo-stream ;;
  *) echo 'MINICPMO_STREAM 需要 true 或 false。' >&2; exit 2 ;;
esac
base_url="${base_url%/}"
case "$base_url" in
  http://?*|https://?*) ;;
  *) echo '--base-url 需要 HTTP(S) 地址。' >&2; exit 2 ;;
esac
if [[ -n "$reference" && ! -f "$reference" ]]; then
  echo "找不到音色参考 WAV：$reference" >&2
  exit 2
fi

call_args=(--backend minicpmo-gguf
  --minicpmo-gguf-base-url "$base_url" --max-calls "$max_calls" --port "$port"
  --max-new-tokens "$max_tokens" --minicpmo-speed "$speed" "$stream_option")
if [[ -n "$reference" ]]; then
  call_args+=(--minicpmo-ref-audio "$reference")
fi
call_args+=(${extra[@]+"${extra[@]}"})

if ! "$dry_run"; then
  if [[ ! -x .venv/bin/python ]]; then
    echo '请先创建项目的 Python 3.11 .venv 并完成环境安装。' >&2
    exit 1
  fi
  # 在启动容器之前验证参数与音频，避免模型启动后才发现输入配置错误。
  env -u CONDA_PREFIX .venv/bin/python - "${call_args[@]}" <<'PY'
import sys
from voice_call_server import parse_args
from voice_service.cli import create_backend
try:
    create_backend(parse_args(sys.argv[1:])).prepare_speech()
except (ValueError, RuntimeError, OSError) as error:
    sys.exit(str(error))
PY
fi

command=(env -u CONDA_PREFIX bash start_minicpmo_call.sh "${call_args[@]}")
health=(curl --noproxy '*' -fsS --retry 20 --retry-connrefused --retry-delay 3
  --connect-timeout 3 --max-time 10 "$base_url/health")

if "$dry_run"; then
  printf '%q ' docker start "$container"; printf '\n'
  printf '%q ' "${health[@]}"; printf '\n'
  printf '%q ' "${command[@]}"; printf '\n'
  exit 0
fi

docker start "$container"
echo "等待 GGUF 推理服务：$base_url/health"
if ! "${health[@]}"; then
  echo "GGUF 健康检查失败，请检查容器日志：docker logs --tail 100 $container" >&2
  exit 1
fi
printf '\n启动通话服务：端口=%s，接入上限=%s，语速=%s，流式=%s，参考音频=%s\n' \
  "$port" "$max_calls" "$speed" "$stream" "${reference:-默认}"
exec "${command[@]}"
