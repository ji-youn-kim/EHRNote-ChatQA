#!/usr/bin/env bash
# Example: score every per-model answer CSV in a directory.
#
# The scoring script:
#   1. Loads the correct answers from the released questions.csv
#      (--questions-csv, or notes.csv/questions.csv next to --input-dir, or
#      <DATA_DIR>/questions.csv).
#   2. Reads every answer CSV in --input-dir, extracts the model name from each
#      filename (the evaluation scripts write answers_<model_name>.csv; a model
#      name is also matched against the canonical list in core/model_ordering.py,
#      or strip a custom prefix with --csv-prefix).
#   3. Computes content / source / paired accuracy per model:
#         - QA-level (averaged across all QA pairs)
#         - Sample-level (mean per-sample correctness ratio)
#         - Sample-level (0/1: every turn correct)
#   4. Writes a per-model summary CSV to <input-dir>/multiturn_qa_scoring_summary.csv
#      (or to --output-csv if specified).
#   5. Writes <input-dir>/scoring_unparseable/<model_name>.txt for any model
#      whose outputs could not be parsed into a letter A-E.

set -euo pipefail
cd "$(dirname "$0")/.."

# Directory holding notes.csv and questions.csv.
DATASET_DIR="data/release"

# Score the local-vLLM answer CSVs:
python scoring/multiturn_qa_scoring.py \
    --input-dir     data/eval_local \
    --questions-csv "$DATASET_DIR/questions.csv"

# Score the API answer CSVs:
python scoring/multiturn_qa_scoring.py \
    --input-dir     data/eval_api \
    --questions-csv "$DATASET_DIR/questions.csv"

# If your answer CSVs follow a different naming convention, pass --csv-prefix
# to strip a known prefix when extracting the model name. E.g.:
#   python scoring/multiturn_qa_scoring.py \
#       --input-dir     data/eval_local \
#       --questions-csv "$DATASET_DIR/questions.csv" \
#       --csv-prefix    answers_

# Free-text generation runs (--generate) have no A-E letter, so they must first
# be judged by evaluation/multiturn_gpt_evaluate_generation_eval.py (see
# scripts/run_generation_eval.sh). Then score the judge's output directory:
#   python scoring/multiturn_qa_scoring.py \
#       --input-dir     data/eval_local_generate/gpt_evaluated \
#       --questions-csv "$DATASET_DIR/questions.csv" \
#       --generate
