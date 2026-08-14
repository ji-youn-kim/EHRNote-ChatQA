#!/usr/bin/env bash
# Example: evaluate a locally-served open-weight model on EHRNote-ChatQA via vLLM.
#
# The paper evaluates 19 open-weight models using vLLM on NVIDIA A6000 GPUs.
# Set CUDA_VISIBLE_DEVICES and --tensor-parallel-size to match your hardware.
#
# Input: the released dataset, i.e. notes.csv + questions.csv in --data-dir.
# Output: <output-dir>/answers_<model_name>.csv, one row per subject with the
# per-turn model outputs as a JSON blob in `model_answer`, plus a stats file at
# <output-dir>/<model_name>.txt summarising unparseable (non A-E) answers.

set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=0,1

# Pick any HuggingFace model id or absolute path to a local model directory.
# Examples: meta-llama/Llama-4-Scout-17B-16E-Instruct,
#           Qwen/Qwen3-Next-80B-A3B-Instruct,
#           deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,
#           google/medgemma-27b-it
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

# Directory holding notes.csv and questions.csv (step 5 of the pipeline, or the
# files as distributed on PhysioNet).
DATASET_DIR="data/release"

python evaluation/multiturn_gpt_evaluate.py \
    --data-dir   "$DATASET_DIR" \
    --output-dir data/eval_local \
    --model      "$MODEL" \
    --tensor-parallel-size 2

# Free-text generation instead of multiple choice: add --generate. The
# multi-turn protocol is identical; the model answers in prose and source
# turns ask which notes/headers support its own previous answer. Output goes
# to <output-dir>/answers_<model_name>_generate.csv and no unparseable-answer
# stats file is written (there is no A-E letter to parse).
#
# python evaluation/multiturn_gpt_evaluate.py \
#     --data-dir   "$DATASET_DIR" \
#     --output-dir data/eval_local_generate \
#     --model      "$MODEL" \
#     --tensor-parallel-size 2 \
#     --generate
#
# The two tables can also be pointed at individually:
#   --notes-csv path/to/notes.csv --questions-csv path/to/questions.csv
