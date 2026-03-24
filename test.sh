#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ENV_FILE="${ROOT_DIR}/.env"
PROXY_URL="${PROXY_URL:-http://192.168.49.2/ollama/v1/chat/completions}"
MODEL="${MODEL:-qwen3-coder:480b-cloud}"
PROMPT="${PROMPT:-Reply with LOGTEST only.}"
MAX_TOKENS="${MAX_TOKENS:-8}"

if [ ! -f "${ENV_FILE}" ]; then
  echo ".env file not found: ${ENV_FILE}" >&2
  exit 1
fi

TOKEN=$(ENV_FILE="${ENV_FILE}" python3 - <<'PY'
import os
from pathlib import Path
for line in Path(os.environ['ENV_FILE']).read_text().splitlines():
    if line.startswith('PROXY_BEARER_TOKEN='):
        print(line.split('=', 1)[1])
        break
else:
    raise SystemExit('PROXY_BEARER_TOKEN not found in .env')
PY
)

curl -sS -X POST "${PROXY_URL}" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d "$(MODEL="${MODEL}" PROMPT="${PROMPT}" MAX_TOKENS="${MAX_TOKENS}" python3 - <<'PY'
import os
import json
print(json.dumps({
    'model': os.environ['MODEL'],
    'messages': [{'role': 'user', 'content': os.environ['PROMPT']}],
    'max_tokens': int(os.environ['MAX_TOKENS']),
}))
PY
)"
