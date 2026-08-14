#!/usr/bin/env python
# coding: utf-8

"""
Generate source-location multiple-choice questions for each turn.

For each row:
1. Load the step_4_pre column (contains the merged step_1 + step_3 content)
2. Build discharge summaries from the row's note columns
3. Fill placeholders in prompt/multiturn_gpt_step4.txt
4. Call the LLM (default: gemini-2.5-pro) to generate source-location MCQs
5. Save the raw model output to the step_4 column
"""

import argparse
import pandas as pd
from pathlib import Path

# Make sibling subfolders importable when this script is launched directly.
import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _sub in ("core", "data_generation", "evaluation", "scoring", ""):
    _p = str(_PROJECT_ROOT / _sub) if _sub else str(_PROJECT_ROOT)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from multiturn_gpt import (
    DATA_DIR,
    PROMPT_DIR,
    load_text,
    build_discharge_summaries_generate,
)
from llm_client import chat_completion


def main(input_file=None, output_file=None, model="gemini-2.5-pro", rows=None):
    if input_file is None:
        input_file = DATA_DIR / "sampled_df_grouped_step4_pre.csv"
    else:
        input_file = Path(input_file)
        if not input_file.is_absolute():
            input_file = DATA_DIR / input_file

    if output_file is None:
        output_file = DATA_DIR / "sampled_df_grouped_step4.csv"
    else:
        output_file = Path(output_file)
        if not output_file.is_absolute():
            output_file = DATA_DIR / output_file

    print(f"Loading input file: {input_file}")
    df = pd.read_csv(input_file)
    print(df)

    base_prompt_template = load_text(PROMPT_DIR / "multiturn_gpt_step4.txt")

    output_col = "step_4"
    df[output_col] = ""

    # If --rows is specified, clear step_4 for those rows so they get re-run
    if rows is not None:
        for r in rows:
            if r < 0 or r >= len(df):
                print(f"Warning: row index {r} is out of range (0-{len(df)-1}), skipping...")
                continue
            df.at[r, output_col] = ""
            print(f"Cleared {output_col} for row {r} (subject_id={df.at[r, 'subject_id']}, qa_category={df.at[r, 'qa_category']}) — will re-run API")

    output_path = output_file

    cnt = 0
    for idx, row in df.iterrows():
        discharge_summaries = build_discharge_summaries_generate(row)

        step4_pre_col = "step_4_pre"

        if step4_pre_col not in df.columns:
            print(f"Warning: {step4_pre_col} column not found for row {idx}, skipping...")
            continue

        step4_pre_qa = df.at[idx, step4_pre_col]

        if pd.isna(step4_pre_qa) or (isinstance(step4_pre_qa, str) and not step4_pre_qa.strip()):
            print(f"Skipping row {idx}: step_4_pre QA is empty")
            continue

        # Skip if already populated (e.g., from a previous run)
        if isinstance(df.at[idx, output_col], str) and df.at[idx, output_col].strip():
            continue

        prompt_text = base_prompt_template
        prompt_text = prompt_text.replace(
            "{discharge_summaries}", discharge_summaries
        )
        prompt_text = prompt_text.replace(
            "{multiturn_qa_step4_pre}", str(step4_pre_qa)
        )

        answer = chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a clinical NLP assistant that generates source-location multiple choice questions based on discharge summaries and initial multi-turn QA.",
                },
                {"role": "user", "content": prompt_text},
            ],
            model=model,
            temperature=0.7,
        )
        df.at[idx, output_col] = answer

        cnt += 1
        print(f"Row {cnt} / {len(df)}")
        print(answer)
        print(row['subject_id'], row['qa_category'])

        # Save after each API call (for safety)
        df.to_csv(output_path, index=False)

        if cnt % 10 == 0:
            print(f"Processed {cnt} / {len(df)} rows")

    df.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate step_4 source-location multiple choice questions from step_4_pre data"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Input CSV file path (relative to data/ directory or absolute path). Default: data/sampled_df_grouped_step4_pre.csv",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output CSV file path (relative to data/ directory or absolute path). Default: data/sampled_df_grouped_step4.csv",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-pro",
        help=(
            "Model name for LLM calls. The paper uses 'gemini-2.5-pro' for "
            "all data-generation steps; pass any other model name to override."
        ),
    )
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=None,
        help="Row indices to clear and re-run (e.g., --rows 4 7 12). Clears step_4 for these rows so the API is called again.",
    )

    args = parser.parse_args()
    main(input_file=args.input_file, output_file=args.output_file, model=args.model, rows=args.rows)
