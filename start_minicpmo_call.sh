#!/usr/bin/env bash
set -euo pipefail
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec bash start_voice_call.sh --backend minicpmo-local --retrieval-metric cosine --port 7865 "$@"
