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
        input_file = DATA_DIR / "sampled_df_grouped_step2_1.csv"
    else:
        input_file = Path(input_file)
        if not input_file.is_absolute():
            input_file = DATA_DIR / input_file

    if output_file is None:
        output_file = DATA_DIR / "sampled_df_grouped_step3.csv"
    else:
        output_file = Path(output_file)
        if not output_file.is_absolute():
            output_file = DATA_DIR / output_file

    print(f"Loading input file: {input_file}")
    df = pd.read_csv(input_file)

    base_prompt_template = load_text(PROMPT_DIR / "multiturn_gpt_step3.txt")

    if "step_3" not in df.columns:
        df["step_3"] = ""

    # If --rows is specified, clear step_3 for those rows so they get re-run
    if rows is not None:
        for r in rows:
            if r < 0 or r >= len(df):
                print(f"Warning: row index {r} is out of range (0-{len(df)-1}), skipping...")
                continue
            df.at[r, "step_3"] = ""
            print(f"Cleared step_3 for row {r} (subject_id={df.at[r, 'subject_id']}, qa_category={df.at[r, 'qa_category']}) — will re-run API")

    output_path = output_file

    cnt = 0
    for idx, row in df.iterrows():
        discharge_summaries = build_discharge_summaries_generate(row)

        step2_1_col = "step_2_1"

        if step2_1_col not in df.columns:
            print(f"Warning: {step2_1_col} column not found for row {idx}, skipping...")
            continue

        step2_qa = df.at[idx, step2_1_col]

        if pd.isna(step2_qa) or (isinstance(step2_qa, str) and not step2_qa.strip()):
            print(f"Skipping row {idx}: step2_1 QA is empty")
            continue

        # Skip if already populated (e.g., from a previous run)
        if isinstance(df.at[idx, "step_3"], str) and df.at[idx, "step_3"].strip():
            continue

        prompt_text = base_prompt_template
        prompt_text = prompt_text.replace(
            "{discharge_summaries}", discharge_summaries
        )
        prompt_text = prompt_text.replace(
            "{multiturn_qa_step2}", str(step2_qa)
        )
        answer = chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a clinical NLP assistant that generates multiple-choice QA options (one correct and four difficult distractors) based on discharge summaries and prior multi-turn QA.",
                },
                {"role": "user", "content": prompt_text},
            ],
            model=model,
            temperature=0.7,
        )
        df.at[idx, "step_3"] = answer

        cnt += 1
        print(f"Row {cnt} / {len(df)}")
        print(answer)

        df.to_csv(output_path, index=False)

        if cnt % 10 == 0:
            print(f"Processed {cnt} / {len(df)} rows")

    df.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate step_3 multiple-choice QA options from step_2_1 data"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Input CSV file path (relative to data/ directory or absolute path). Default: data/sampled_df_grouped_step2_1.csv",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output CSV file path (relative to data/ directory or absolute path). Default: data/sampled_df_grouped_step3.csv",
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
        help="Row indices to clear and re-run (e.g., --rows 4 7 12). Clears step_3 for these rows so the API is called again.",
    )

    args = parser.parse_args()
    main(input_file=args.input_file, output_file=args.output_file, model=args.model, rows=args.rows)