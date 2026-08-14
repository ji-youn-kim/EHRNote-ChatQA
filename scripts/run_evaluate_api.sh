#!/usr/bin/env bash
# Example: evaluate a closed-source model on EHRNote-ChatQA via HTTP API.
#
# Backend is auto-detected from --model:
#   - names containing "gemini" go to Gemini on Vertex AI
#   - all others go to Azure OpenAI (GPT)
#
# Credentials are read from environment variables; export them in your
# shell before running this script (see the "API credentials" section of
# README.md for the full list).
#
# Use of MIMIC-derived content with proprietary APIs should run through
# HIPAA-compliant deployments (Vertex AI for Gemini, Azure OpenAI for GPT),
# consistent with the MIMIC-IV data use agreement.
#
# Input: the released dataset, i.e. notes.csv + questions.csv in --data-dir.
# Output: <output-dir>/answers_<model_name>.csv, checkpointed after every
# subject; rerunning skips subjects that already have an answer.

set -euo pipefail
cd "$(dirname "$0")/.."

# Pick one of the three proprietary models used in the paper, or substitute
# any other model name the configured provider supports.
MODEL="gpt-5.4"               # or: gpt-5.4-mini, gemini-3-flash-preview

# Directory holding notes.csv and questions.csv (step 5 of the pipeline, or the
# files as distributed on PhysioNet).
DATASET_DIR="data/release"

python evaluation/multiturn_gpt_evaluate_api.py \
    --data-dir   "$DATASET_DIR" \
    --output-dir data/eval_api \
    --model      "$MODEL"

# Free-text generation instead of multiple choice: add --generate. The
# multi-turn protocol is identical; the model answers in prose and source
# turns ask which notes/headers support its own previous answer. Output goes
# to <output-dir>/answers_<model_name>_generate.csv and no unparseable-answer
# stats file is written (there is no A-E letter to parse).
#
# python evaluation/multiturn_gpt_evaluate_api.py \
#     --data-dir   "$DATASET_DIR" \
#     --output-dir data/eval_api_generate \
#     --model      "$MODEL" \
#     --generate
#
# The two tables can also be pointed at individually:
#   --notes-csv path/to/notes.csv --questions-csv path/to/questions.csv
