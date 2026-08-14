#!/usr/bin/env bash
# Example: score a free-text generation run with the LLM judge.
#
# Free-text answers carry no A-E letter, so a run produced with --generate is
# judged before it can be scored. The judge makes two calls per subject:
#
#   1. Content pass  -- scores the content turns 0-2 against the question, with
#      the reference correct answer from questions.csv as a hint and the
#      discharge summaries from notes.csv as the authoritative source.
#   2. Source pass   -- scores the source-citation turns 0-2 against the
#      MODEL'S OWN content answer. The verbatim content-pass output is injected
#      as [Content Pass Output]; the judge forces a source score of 0 when the
#      paired content question scored 0.
#
# Output: <answers-dir>/gpt_evaluated/<input-name>.csv (or gemini_evaluated/ for
# a Gemini judge), with the judge text in a gpt_eval / gemini_eval column. The
# input CSV is never modified. Each row is checkpointed and rows that already
# carry a judgement are skipped, so this can be interrupted and resumed.
#
# Credentials are read from environment variables; export them in your shell
# before running (see the "API credentials" section of README.md).

set -euo pipefail
cd "$(dirname "$0")/.."

# Directory holding notes.csv and questions.csv.
DATASET_DIR="data/release"

# The generation-mode answer CSV to judge, e.g. the output of
#   bash scripts/run_evaluate_local.sh   (with --generate)
GENERATE_CSV="data/eval_local_generate/answers_DeepSeek-R1-Distill-Qwen-32B_generate.csv"

# Judge model. The paper used gpt-5.4. A name containing "gemini" is routed to
# Gemini on Vertex AI instead of Azure OpenAI.
JUDGE_MODEL="gpt-5.4"

python evaluation/multiturn_gpt_evaluate_generation_eval.py \
    --input-csv "$GENERATE_CSV" \
    --data-dir  "$DATASET_DIR" \
    --model     "$JUDGE_MODEL"

# Then aggregate the judged CSVs into the per-model summary:
python scoring/multiturn_qa_scoring.py \
    --input-dir     "$(dirname "$GENERATE_CSV")/gpt_evaluated" \
    --questions-csv "$DATASET_DIR/questions.csv" \
    --generate

# If scoring excludes a model because some rows' judge output was unusable, it
# prints the offending row indices. Re-judge only those rows with:
#
#   python evaluation/multiturn_gpt_evaluate_generation_eval.py \
#       --input-csv "$GENERATE_CSV" \
#       --data-dir  "$DATASET_DIR" \
#       --model     "$JUDGE_MODEL" \
#       --rows 4 7 12
