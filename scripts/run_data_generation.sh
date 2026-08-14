#!/usr/bin/env bash
# Example: run the full multi-turn QA data-generation pipeline.
#
# The pipeline produces, for each patient row, a structured multi-turn QA
# sample with content questions, source-location (evidence-grounding)
# questions, and five answer choices per question.
#
# Steps that call the LLM API: 1, 2, 3, 4.
# Steps that are offline transforms: 2_1, 4_preprocess, 4_postprocess, 5.
#
# Step 5 converts the wide step-4 output into the released two-table dataset
# (notes.csv + questions.csv) that every evaluation and scoring script reads.
# Include the optional `note_seqs` and `storetimes` columns in the input CSV to
# have note_id / note_seq / storetime populated in notes.csv; without them those
# three columns are written empty and everything else is unaffected.
#
# The paper uses gemini-2.5-pro for all four LLM steps. Override with
# --model <other_model_name> if you want to regenerate with a different
# model. Credentials are read from environment variables; export them in
# your shell before running this script (see the "API credentials"
# section of README.md for the full list).
#
# Input data: the pipeline expects an input CSV located at
#   data/sampled_df_grouped_with_cat.csv
# with columns: subject_id, qa_category, num_notes, charttimes, hadm_ids,
# note_1, note_2, ..., note_{num_notes}.

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="gemini-2.5-pro"

python data_generation/multiturn_gpt_step1.py             --model "$MODEL"
python data_generation/multiturn_gpt_step2.py             --model "$MODEL"
python data_generation/multiturn_gpt_step2_1.py
python data_generation/multiturn_gpt_step3.py             --model "$MODEL"
python data_generation/multiturn_gpt_step4_preprocess.py
python data_generation/multiturn_gpt_step4.py             --model "$MODEL"
python data_generation/multiturn_gpt_step4_postprocess.py
python data_generation/multiturn_gpt_step5_release_tables.py \
    --input-csv  data/sampled_df_grouped_step4_post.csv \
    --output-dir data/release

echo "Done. Released dataset at data/release/{notes.csv,questions.csv}"
