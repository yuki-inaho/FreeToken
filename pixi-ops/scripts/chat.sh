#!/usr/bin/env bash
# 簡単なチャット疎通 (pixi run chat [プロンプト])
set -euo pipefail
PORT="${FREETOKEN_PORT:-1919}"
MODEL="${FREETOKEN_MODEL_NAME:-DeepSeek-V4-Flash-0731}"
PROMPT="${1:-Say hello in exactly five words.}"
curl -s -m 600 "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "$(python - <<PY
import json
print(json.dumps({"model": "$MODEL",
                  "messages": [{"role": "user", "content": "$PROMPT"}],
                  "max_tokens": 64, "temperature": 0}))
PY
)" | python -c "import json,sys; r=json.load(sys.stdin); print(r['choices'][0]['message']['content']); print('usage:', r.get('usage'))"
