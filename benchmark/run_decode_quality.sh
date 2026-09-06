#!/usr/bin/env bash
# Historical Qwen3.8 quality gates against an already-running server; never boots/stops it.
# bash benchmark/run_decode_quality.sh OUTPUT_DIR [BASE_URL] [MODEL]
set -euo pipefail

if [[ "${1:-}" == --help || "${1:-}" == -h || $# -eq 0 || $# -gt 3 ]]; then
  cat <<'USAGE'
Usage: bash benchmark/run_decode_quality.sh OUTPUT_DIR [BASE_URL] [MODEL]

Runs GSM8K-150 (8-shot CoT), MMLU-chat (10 per subject), then needle retrieval
at 200K requested tokens and five depths. All requests use concurrency 1.
BASE_URL defaults to http://127.0.0.1:1920/v1 (or FREETOKEN_API_BASE).
MODEL defaults to Qwen3.8-Flash-Next-NVFP4 (or FREETOKEN_MODEL_NAME).

OUTPUT_DIR must not already contain these gate results. Native lm_eval JSON,
logs, run.json, and summary.json are retained. Evaluation errors or scores below
the historical gates fail the command after writing the results.

Overrides: FREETOKEN_LM_EVAL_VENV, FREETOKEN_VENV, FREETOKEN_PORT.
USAGE
  [[ $# -gt 0 && $# -le 3 ]] || exit 2
  exit 0
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_VENV="${FREETOKEN_LM_EVAL_VENV:-/home/nuko/.freetoken/lm-eval-venv}"
VENV="${FREETOKEN_VENV:-/home/nuko/.freetoken/venv}"
LMEVAL="$EVAL_VENV/bin/lm_eval"
EVAL_PY="$EVAL_VENV/bin/python"
PY="$VENV/bin/python"
BASE="${2:-${FREETOKEN_API_BASE:-http://127.0.0.1:${FREETOKEN_PORT:-1920}/v1}}"
BASE="${BASE%/}"
MODEL="${3:-${FREETOKEN_MODEL_NAME:-Qwen3.8-Flash-Next-NVFP4}}"
OUT="$1"
for executable in "$LMEVAL" "$EVAL_PY" "$PY"; do
  [[ -x "$executable" ]] || { printf 'Missing executable: %s\n' "$executable" >&2; exit 1; }
done
for result in gsm8k mmlu needle.log run.json summary.json; do
  [[ ! -e "$OUT/$result" ]] || { printf 'Result already exists: %s/%s\n' "$OUT" "$result" >&2; exit 1; }
done
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
export TOKENIZERS_PARALLELISM=false

"$EVAL_PY" - "$OUT" "$BASE" "$MODEL" <<'PY'
import datetime, json, pathlib, sys
import lm_eval
out, base, model = sys.argv[1:]
record = {
    "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "base_url": base, "model": model, "lm_eval_version": lm_eval.__version__,
    "concurrency": 1, "temperature": 0,
    "gsm8k": {"task": "gsm8k_cot_llama", "limit": 150, "num_fewshot": 8,
              "apply_chat_template": True, "fewshot_as_multiturn": True},
    "mmlu": {"task": "mmlu_chat", "limit_per_subject": 10, "num_fewshot": 0,
             "apply_chat_template": True, "max_gen_toks": 48},
    "needle": {"requested_tokens": 200000, "depths": [0.1, 0.3, 0.5, 0.7, 0.9]},
    "historical_reference": {"gsm8k_flexible_exact_match": 133 / 150,
                             "mmlu_exact_match": 498 / 570, "needle_hits": 5},
}
(pathlib.Path(out) / "run.json").write_text(json.dumps(record, indent=2) + "\n")
PY

MODEL_ARGS="model=$MODEL,base_url=$BASE/chat/completions,num_concurrent=1,max_retries=3,tokenized_requests=False,timeout=900"
COMMON=(--model local-chat-completions --model_args "$MODEL_ARGS"
        --gen_kwargs temperature=0 --batch_size 1 --seed 0,1234,1234,1234)

printf 'GSM8K-150: %s\n' "$(date -Is)"
mkdir -p "$OUT/gsm8k"
"$LMEVAL" "${COMMON[@]}" --tasks gsm8k_cot_llama --limit 150 \
  --apply_chat_template --fewshot_as_multiturn --output_path "$OUT/gsm8k" \
  >"$OUT/gsm8k.log" 2>&1

printf 'MMLU-chat 570 questions: %s\n' "$(date -Is)"
mkdir -p "$OUT/mmlu"
"$LMEVAL" "${COMMON[@]}" --include_path "$REPO/benchmark/quality_tasks" \
  --tasks mmlu_chat --limit 10 --apply_chat_template --output_path "$OUT/mmlu" \
  >"$OUT/mmlu.log" 2>&1

printf 'Needle retrieval, five depths: %s\n' "$(date -Is)"
"$PY" "$REPO/scripts/needle-check.py" --base "$BASE" --model "$MODEL" \
  --tokens 200000 --depths 0.1,0.3,0.5,0.7,0.9 | tee "$OUT/needle.log"

"$EVAL_PY" - "$OUT" <<'PY'
import datetime, json, pathlib, re, sys
out = pathlib.Path(sys.argv[1])

def result_file(name):
    files = sorted((out / name).rglob("results_*.json"))
    if len(files) != 1:
        raise RuntimeError(f"Expected one {name} result JSON, found {files}")
    return files[0], json.loads(files[0].read_text())

gsm_path, gsm = result_file("gsm8k")
mmlu_path, mmlu = result_file("mmlu")
needle_text = (out / "needle.log").read_text()
needle_match = re.search(r"^retrieved (\d+)/(\d+)$", needle_text, re.MULTILINE)
if needle_match is None:
    raise RuntimeError("Needle run did not produce its completion summary")
hits, total = map(int, needle_match.groups())
if total != 5:
    raise RuntimeError(f"Expected five needle depths, found {total}")
summary = {
    "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "gsm8k": {"results": gsm["results"]["gsm8k_cot_llama"],
              "result_file": str(gsm_path.relative_to(out))},
    "mmlu": {"results": mmlu["results"]["mmlu_chat"],
             "result_file": str(mmlu_path.relative_to(out))},
    "needle": {"hits": hits, "total": total, "passed": hits == total,
               "depth_results": [line for line in needle_text.splitlines() if line.startswith("depth ")]},
}
summary["gates"] = {
    "gsm8k": gsm["results"]["gsm8k_cot_llama"]["exact_match,flexible-extract"] >= 133 / 150,
    "mmlu": mmlu["results"]["mmlu_chat"]["exact_match,get_letter"] >= 498 / 570,
    "needle": hits == total,
}
summary["passed"] = all(summary["gates"].values())
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
sys.exit(0 if summary["passed"] else 1)
PY
