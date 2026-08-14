import argparse
import pandas as pd
import re
import random
import sys
from pathlib import Path
import time

# Make sibling subfolders importable when this script is launched directly.
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
    QA_TEMPLATE_CATEGORIES,
)
from llm_client import chat_completion


def main(input_file=None, output_file=None, model="gemini-2.5-pro"):
    if input_file is None:
        input_file = DATA_DIR / "sampled_df_grouped_with_cat.csv"
    else:
        input_file = Path(input_file)
        if not input_file.is_absolute():
            input_file = DATA_DIR / input_file
    
    if output_file is None:
        output_file = DATA_DIR / "sampled_df_grouped_step1.csv"
    else:
        output_file = Path(output_file)
        if not output_file.is_absolute():
            output_file = DATA_DIR / output_file
    
    print(f"Loading input file: {input_file}")
    df = pd.read_csv(input_file)
    print(df)
    print(df['qa_category'].value_counts())

    qa_template_categories = QA_TEMPLATE_CATEGORIES

    base_prompt_template_singlenote = load_text(PROMPT_DIR / "multiturn_gpt_step1_singlenote.txt")
    base_prompt_template_multinote = load_text(PROMPT_DIR / "multiturn_gpt_step1_multinote.txt")

    # Pre-load both singlenote and multinote templates for "other examples"
    qa_template_examples_singlenote = {}
    qa_template_examples_multinote = {}
    for cat in qa_template_categories:
        singlenote_path = PROMPT_DIR / f"template_ex_singlenote_{cat}.txt"
        multinote_path = PROMPT_DIR / f"template_ex_multinote_{cat}.txt"
        qa_template_examples_singlenote[cat] = load_text(singlenote_path)
        qa_template_examples_multinote[cat] = load_text(multinote_path)

    # Heading map for category display names
    heading_map = {
        "diagnosis": "Diagnosis",
        "symptom": "Symptom",
        "procedure": "Procedure",
        "medication": "Medication",
        "microbiology": "Microbiology",
        "clinical_assessment": "Clinical Assessment",
        "clinical_outcome": "Clinical Outcome",
        "disch_plan": "Discharge Planning",
    }

    # Set random seed for reproducible sampling
    random.seed(42)

    if "step_1" not in df.columns:
        df["step_1"] = ""

    col_name = "step_1"
    
    # Find the first row where step_1 is empty
    start_idx = None
    for idx, row in df.iterrows():
        step1_value = df.at[idx, col_name]
        if pd.isna(step1_value) or (isinstance(step1_value, str) and not step1_value.strip()):
            start_idx = idx
            break
    
    if start_idx is None:
        print("All rows already have step_1 values. Nothing to process.")
        return
    
    print(f"Starting processing from row {start_idx} (first row with empty step_1)")
    
    cnt = 0
    # Loop over each patient row starting from the first empty step_1
    for idx, row in df.iterrows():
        if idx < start_idx:
            continue
            
        discharge_summaries = build_discharge_summaries_generate(row)

        category = row.get("qa_category")
        if category not in qa_template_categories:
            print(f"Skipping row {idx}: unknown qa_category '{category}'")
            continue

        # Skip if already populated (shouldn't happen after start_idx, but check anyway)
        step1_value = df.at[idx, col_name]
        if isinstance(step1_value, str) and step1_value.strip():
            print(f"Skipping row {idx}: step_1 already populated")
            continue

        # Get num_notes for this row to determine which template to use
        num_notes = row.get("num_notes", 1)
        if pd.isna(num_notes):
            num_notes = 1
        else:
            num_notes = int(num_notes)
        
        if num_notes == 1:
            template_path = PROMPT_DIR / f"template_ex_singlenote_{category}.txt"
        else:
            template_path = PROMPT_DIR / f"template_ex_multinote_{category}.txt"
        
        current_category_examples = load_text(template_path).strip()
        
        # Build examples_other: for each category except current, sample one template
        # Use the same template type (singlenote/multinote) as the current row's num_notes
        other_examples_parts = []
        category_num = 1
        for other_cat in qa_template_categories:
            if other_cat == category:
                continue
            
            if num_notes == 1:
                template_text = qa_template_examples_singlenote[other_cat]
            else:
                template_text = qa_template_examples_multinote[other_cat]
            
            # Split by [Template X - Description] markers
            # Pattern matches: [Template 1 - Description], [Template 2 - Another Description], etc.
            templates = re.split(r'\[Template \d+[^\]]*\]', template_text)
            templates = [t.strip() for t in templates if t.strip()]
            
            if templates:
                sampled_template = random.choice(templates)
                heading = heading_map.get(other_cat, other_cat.replace("_", " ").title())
                other_examples_parts.append(f"{category_num}. {heading}")
                other_examples_parts.append(sampled_template)
                category_num += 1
        
        other_examples_block = "\n\n".join(other_examples_parts)

        if num_notes == 1:
            base_prompt_template = base_prompt_template_singlenote
        else:
            base_prompt_template = base_prompt_template_multinote

        prompt_text = base_prompt_template
        prompt_text = prompt_text.replace("{qa_template_category}", heading_map[category])
        prompt_text = prompt_text.replace("{qa_template_examples}", current_category_examples)
        prompt_text = prompt_text.replace("{qa_template_examples_other}", other_examples_block)
        prompt_text = prompt_text.replace("{discharge_summaries}", discharge_summaries)

        answer = chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a clinical NLP assistant that generates multi-turn QA based on discharge summaries.",
                },
                {"role": "user", "content": prompt_text},
            ],
            model=model,
            temperature=1,
        )
        df.at[idx, col_name] = answer
        cnt += 1
        print(f"Row {cnt} / {len(df)}, category: {category}")
        print(answer)
        time.sleep(1)

        df.to_csv(output_file, index=False)

        if cnt % 10 == 0:
            print(f"Processed {cnt} / {len(df)} rows")

    df.to_csv(output_file, index=False)
    print(f"Saved results to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate step_1 QA data from discharge summaries"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Input CSV file path (relative to data/ directory or absolute path). Default: data/sampled_df_grouped_with_cat.csv",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output CSV file path (relative to data/ directory or absolute path). Default: data/sampled_df_grouped_step1.csv",
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

    args = parser.parse_args()
    main(input_file=args.input_file, output_file=args.output_file, model=args.model)
