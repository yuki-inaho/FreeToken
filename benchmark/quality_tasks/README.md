# Historical Qwen3.8 quality tasks

`mmlu_chat/` preserves the custom task definitions used by the September 2, 2026
fp16-state quality run: 498/570 correct (87.3684%). There are 57 subjects, ten test
questions each, zero few-shot examples, greedy generation capped at 48 tokens,
and the original standalone-letter extraction expression. The subject YAML,
template, and weighted group definition are unchanged from that run.

Use `bash benchmark/run_decode_quality.sh OUTPUT_DIR [BASE_URL] [MODEL]` with an
already-running server. The runner also repeats the historical GSM8K gate with
`gsm8k_cot_llama`, eight examples as chat turns, and 150 test questions. Compare
its `exact_match,flexible-extract` score with 133/150 (88.6667%). The recorded
baseline used lm_eval 0.4.12; the runner records the installed version in
`run.json`. It retains the native result JSON and logs, plus `summary.json`.

The final gate calls the existing `scripts/needle-check.py` unchanged, requesting
200,000 tokens at depths 0.1, 0.3, 0.5, 0.7, and 0.9. The actual historical prompt
length was approximately 191,404 tokens; five of five answers were retrieved.
This checks the same prompt builder and answer matcher as the original run.

These are the historical gates, not the newer `gsm8k_cot` task or
`scripts/quality-gate.py` proxy. Keep their results separate. The historical run
took about 4m17s for GSM8K, 10m34s for MMLU, and 4m41s for needle retrieval;
allow roughly 20 minutes for all three on the 32 GiB profile.
