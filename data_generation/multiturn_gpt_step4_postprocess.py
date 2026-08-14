#!/usr/bin/env python
# coding: utf-8

"""
Post-process step_3 and step_4 columns to create structured QA dictionaries.

For each row:
1. Parse step_3 QAs (keys: 1, 2, 3, ...)
2. Parse step_4 QAs (keys: 1-1, 2-1, 3-1, ...)
3. Shuffle answers for both sets
4. Combine into a single dictionary with keys: 1, 1-1, 2, 2-1, 3, 3-1, ...
"""

import re
import random
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import argparse
import pandas as pd

# Make sibling subfolders importable when this script is launched directly.
import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _sub in ("core", "data_generation", "evaluation", "scoring", ""):
    _p = str(_PROJECT_ROOT / _sub) if _sub else str(_PROJECT_ROOT)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from multiturn_gpt import DATA_DIR

# Module-level regex used by the parsers below.
# The per-question A/I sub-patterns are built on the fly inside each parser
# (they need the question number interpolated), so we only keep the
# question-marker patterns at module scope.
_Q_PATTERN = re.compile(r"Q(\d+):\s*(.*?)(?=Q\d+:|$)", re.DOTALL)
_Q4_PATTERN = re.compile(r"Q(\d+)-1:\s*(.*?)(?=Q\d+-1:|$)", re.DOTALL)  # step_4: Q1-1:, Q2-1:, etc.


def parse_qa_text(text: str) -> Dict[str, Dict[str, object]]:
    """
    Parse step_3 text to extract questions, answers, and incorrect choices.
    
    Format (with reasoning):
        Q1: ...
        A1: ...
        A1-Reasoning: ...
        I1: ...
        I1-Reasoning: ...
        ...
    
    Returns a dictionary with string keys:
        {
          qnum_str: {
            'question': str,
            'answer': str,
            'answer_reasoning': str,
            'incorrect': List[str],
            'incorrect_reasoning': List[str],
          }
        }
    """
    if pd.isna(text) or not text.strip():
        return {}
    
    # Find all question markers
    q_matches = list(_Q_PATTERN.finditer(text))
    if not q_matches:
        return {}
    
    qas = {}
    
    for i, q_match in enumerate(q_matches):
        qnum = int(q_match.group(1))
        qnum_str = str(qnum)
        
        # Get the section for this question (from Q to next Q or end)
        start_pos = q_match.start()
        end_pos = q_matches[i + 1].start() if i + 1 < len(q_matches) else len(text)
        question_section = text[start_pos:end_pos]
        
        # Extract question text (after Q{num}:)
        question_match = re.match(rf"Q{qnum}:\s*(.*?)(?=A{qnum}:|$)", question_section, re.DOTALL)
        question_text = question_match.group(1).strip() if question_match else ""
        
        # Extract answer (A{qnum}:) and its reasoning (A{qnum}-Reasoning:)
        answer_match = re.search(
            rf"A{qnum}:\s*(.*?)(?=\nA{qnum}-Reasoning:|\nI1:|\nQ\d+:|$)",
            question_section,
            re.DOTALL,
        )
        answer = answer_match.group(1).strip() if answer_match else ""

        answer_reason_match = re.search(
            rf"A{qnum}-Reasoning:\s*(.*?)(?=\nI1:|\nQ\d+:|$)",
            question_section,
            re.DOTALL,
        )
        answer_reasoning = (
            answer_reason_match.group(1).strip() if answer_reason_match else ""
        )
        
        # Extract incorrect choices (I1, I2, I3, I4) and their reasoning
        incorrect_choices = []
        incorrect_reasoning = []
        for i_num in range(1, 5):
            choice_match = re.search(
                rf"I{i_num}:\s*(.*?)(?=\nI{i_num}-Reasoning:|\nI\d+:|\nQ\d+:|$)",
                question_section,
                re.DOTALL,
            )
            if choice_match:
                incorrect_choices.append(choice_match.group(1).strip())
                reason_match = re.search(
                    rf"I{i_num}-Reasoning:\s*(.*?)(?=\nI\d+:|\nQ\d+:|$)",
                    question_section,
                    re.DOTALL,
                )
                incorrect_reasoning.append(
                    reason_match.group(1).strip() if reason_match else ""
                )
        
        if question_text and answer:
            qas[qnum_str] = {
                'question': question_text,
                'answer': answer,
                'answer_reasoning': answer_reasoning,
                'incorrect': incorrect_choices[:4],
                'incorrect_reasoning': incorrect_reasoning[:4],
            }
    
    return qas


def create_qa_dict(parsed_qas: Dict, correct_answer_letters: List[str], seed: Optional[int] = None) -> Dict:
    """
    Create the final QA dictionary structure with shuffled choices.
    
    Args:
        parsed_qas: Dictionary from parse_qa_text (keys can be int or str like "1-1")
        correct_answer_letters: Pre-assigned list of correct answer letters (A-E) for each question
        seed: Optional random seed for reproducibility (used for shuffling incorrect choices)
    
    Returns:
        Dictionary with structure:
        {
          key: {
            'Question': str,
            'A': str, ..., 'E': str,
            'A_reasoning': str, ..., 'E_reasoning': str,
            'correct_answer': str,
          }
        }
    """
    if seed is not None:
        random.seed(seed)
    
    result = {}
    
    if not parsed_qas:
        return result
    
    # Sort questions: pure numbers first (step_3: "1", "2", "3"), then "x-1" format (step_4: "1-1", "2-1")
    qnums = sorted(parsed_qas.keys(), key=lambda x: (
        int(x.split('-')[0]) if '-' in x else int(x),
        1 if '-' in x else 0,  # step_3 before step_4
        int(x.split('-')[1]) if '-' in x else 0
    ))
    
    if len(qnums) != len(correct_answer_letters):
        raise ValueError(f"Number of questions ({len(qnums)}) doesn't match number of answer letters ({len(correct_answer_letters)})")
    
    for i, qnum in enumerate(qnums):
        qa_data = parsed_qas[qnum]
        answer = qa_data["answer"]
        incorrect = qa_data["incorrect"]
        answer_reasoning = qa_data.get("answer_reasoning", "")
        incorrect_reasoning = qa_data.get("incorrect_reasoning", [])
        correct_letter = correct_answer_letters[i]

        # Map letter to index (A=0, B=1, C=2, D=3, E=4)
        letter_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
        correct_idx = letter_to_idx[correct_letter]
        
        # Ensure we have exactly 4 incorrect options and reasoning (pad with empty strings if needed)
        incorrect_choices = incorrect[:4]
        while len(incorrect_choices) < 4:
            incorrect_choices.append("")
        incorrect_reasoning_list = list(incorrect_reasoning[:4])
        while len(incorrect_reasoning_list) < 4:
            incorrect_reasoning_list.append("")
        
        # Pair incorrect choices with their reasoning and shuffle them together
        incorrect_pairs = list(zip(incorrect_choices, incorrect_reasoning_list))
        random.shuffle(incorrect_pairs)

        # Build shuffled lists with correct answer at the assigned position
        shuffled = [""] * 5
        shuffled_reasoning = [""] * 5
        shuffled[correct_idx] = answer
        shuffled_reasoning[correct_idx] = answer_reasoning

        incorrect_iter = iter(incorrect_pairs)
        for j in range(5):
            if j == correct_idx:
                continue
            choice_text, choice_reason = next(incorrect_iter, ("", ""))
            shuffled[j] = choice_text
            shuffled_reasoning[j] = choice_reason
        
        result[qnum] = {
            "Question": qa_data["question"],
            "A": shuffled[0],
            "B": shuffled[1],
            "C": shuffled[2],
            "D": shuffled[3],
            "E": shuffled[4],
            "A_reasoning": shuffled_reasoning[0],
            "B_reasoning": shuffled_reasoning[1],
            "C_reasoning": shuffled_reasoning[2],
            "D_reasoning": shuffled_reasoning[3],
            "E_reasoning": shuffled_reasoning[4],
            "correct_answer": correct_letter,
        }
    
    return result


def load_qa_dict(json_str: str) -> Dict[str, Dict[str, str]]:
    """
    Load a QA dictionary from JSON string. All keys are strings.
    
    The JSON string is stored as a dictionary.
    Keys are strings: step_3 uses "1", "2", "3", ... and step_4 uses "1-1", "2-1", ...
    """
    if not json_str or not json_str.strip():
        return {}
    
    try:
        qa_dict = json.loads(json_str)
        return {str(k): v for k, v in qa_dict.items()}
    except (json.JSONDecodeError, ValueError, TypeError):
        return {}


def parse_step4_qa_text(text: str) -> Dict[str, Dict[str, object]]:
    """
    Parse step_4 text to extract questions, answers, and incorrect choices.
    
    Format (with reasoning):
        Q1-1: ...
        A1-1: ...
        A1-1-Reasoning: ...
        I1: ...
        I1-Reasoning: ...
        ...
    
    Returns a dictionary with string keys like "1-1", "2-1", etc.: 
        {
          qnum_str: {
            'question': str,
            'answer': str,
            'answer_reasoning': str,
            'incorrect': List[str],
            'incorrect_reasoning': List[str],
          }
        }
    """
    if pd.isna(text) or not text.strip():
        return {}
    
    # Find all question markers (Q1-1:, Q2-1:, etc.)
    q_matches = list(_Q4_PATTERN.finditer(text))
    if not q_matches:
        return {}
    
    qas = {}
    
    for i, q_match in enumerate(q_matches):
        qnum = int(q_match.group(1))
        # Step_4 QAs are paired with step_3 QAs, so use "qnum-1" as key
        qnum_str = f"{qnum}-1"
        
        # Get the section for this question (from Q to next Q or end)
        start_pos = q_match.start()
        end_pos = q_matches[i + 1].start() if i + 1 < len(q_matches) else len(text)
        question_section = text[start_pos:end_pos]
        
        # Extract question text (after Q{num}-1:)
        question_match = re.match(rf"Q{qnum}-1:\s*(.*?)(?=A{qnum}-1:|$)", question_section, re.DOTALL)
        question_text = question_match.group(1).strip() if question_match else ""
        
        # Extract answer (A{num}-1:) and its reasoning (A{num}-1-Reasoning:)
        answer_match = re.search(
            rf"A{qnum}-1:\s*(.*?)(?=\nA{qnum}-1-Reasoning:|\nI1:|\nQ\d+-1:|$)",
            question_section,
            re.DOTALL,
        )
        answer = answer_match.group(1).strip() if answer_match else ""

        answer_reason_match = re.search(
            rf"A{qnum}-1-Reasoning:\s*(.*?)(?=\nI1:|\nQ\d+-1:|$)",
            question_section,
            re.DOTALL,
        )
        answer_reasoning = (
            answer_reason_match.group(1).strip() if answer_reason_match else ""
        )
        
        # Extract incorrect choices (I1, I2, I3, I4) and their reasoning
        incorrect_choices = []
        incorrect_reasoning = []
        for i_num in range(1, 5):
            choice_match = re.search(
                rf"I{i_num}:\s*(.*?)(?=\nI{i_num}-Reasoning:|\nI\d+:|\nQ\d+-1:|$)",
                question_section,
                re.DOTALL,
            )
            if choice_match:
                incorrect_choices.append(choice_match.group(1).strip())
                reason_match = re.search(
                    rf"I{i_num}-Reasoning:\s*(.*?)(?=\nI\d+:|\nQ\d+-1:|$)",
                    question_section,
                    re.DOTALL,
                )
                incorrect_reasoning.append(
                    reason_match.group(1).strip() if reason_match else ""
                )
        
        if question_text and answer:
            qas[qnum_str] = {
                'question': question_text,
                'answer': answer,
                'answer_reasoning': answer_reasoning,
                'incorrect': incorrect_choices[:4],
                'incorrect_reasoning': incorrect_reasoning[:4],
            }
    
    return qas


_Q1_MARKER_RE = re.compile(r"(?m)^\s*Q(\d+):")


def _split_step1_into_blocks(text: str) -> List[Tuple[int, str]]:
    """Split step_1 text into (qnum, block_text) tuples."""
    if pd.isna(text) or not str(text).strip():
        return []
    text = str(text)
    matches = list(_Q1_MARKER_RE.finditer(text))
    if not matches:
        return []
    blocks: List[Tuple[int, str]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        qnum = int(m.group(1))
        blocks.append((qnum, text[start:end].rstrip()))
    return blocks


@dataclass(frozen=True)
class Mismatch:
    row_idx: int
    subject_id: Optional[str]
    qa_category: Optional[str]
    issue: str
    details: str


def main(input_file: Optional[str] = None, output_file: Optional[str] = None) -> None:
    """
    Post-process step_3 and step_4 columns to create structured QA dictionaries.

    Parameters mirror multiturn_gpt_step1.py:
      - input_file: CSV path (relative to DATA_DIR or absolute).
      - output_file: CSV path (relative to DATA_DIR or absolute).
    """
    if input_file is None:
        input_path = DATA_DIR / "sampled_df_grouped_step4.csv"
    else:
        input_path = Path(input_file)
        if not input_path.is_absolute():
            input_path = DATA_DIR / input_path

    if output_file is None:
        output_path = DATA_DIR / "sampled_df_grouped_step4_post.csv"
    else:
        output_path = Path(output_file)
        if not output_path.is_absolute():
            output_path = DATA_DIR / output_path

    # Global seed for reproducibility
    global_seed = 42
    random.seed(global_seed)

    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)
    
    print(f"Found {len(df)} rows")
    
    # First pass: Count total questions across all rows
    print("Counting questions across all rows...")
    row_question_counts = []
    step3_col = "step_3"
    step4_col = "step_4"
    
    for idx, row in df.iterrows():
        step3_text = row[step3_col] if step3_col in df.columns else None
        step4_text = row[step4_col] if step4_col in df.columns else None
        
        parsed_step3_qas = {}
        if pd.notna(step3_text) and str(step3_text).strip():
            parsed_step3_qas = parse_qa_text(str(step3_text))
        
        parsed_step4_qas = {}
        if pd.notna(step4_text) and str(step4_text).strip():
            parsed_step4_qas = parse_step4_qa_text(str(step4_text))
        
        if not parsed_step3_qas and not parsed_step4_qas:
            row_question_counts.append(0)
            continue
        
        step3_keys = set(parsed_step3_qas.keys())
        step4_base_keys = set([k.split('-')[0] for k in parsed_step4_qas.keys()])
        common_keys = step3_keys.intersection(step4_base_keys)
        
        if common_keys:
            # Count: step_3 QAs + step_4 QAs (each step_3 QA has a corresponding step_4 QA)
            num_questions = len(common_keys) * 2
        else:
            num_questions = 0
        
        row_question_counts.append(num_questions)
    
    total_questions = sum(row_question_counts)
    print(f"Total questions across all rows: {total_questions}")

    # Generate global list of answer letters (A-E) evenly distributed for qas
    letters = ['A', 'B', 'C', 'D', 'E']
    base_count = total_questions // 5
    remainder = total_questions % 5

    global_answer_letters = []
    for i, letter in enumerate(letters):
        count = base_count + (1 if i < remainder else 0)
        global_answer_letters.extend([letter] * count)

    random.shuffle(global_answer_letters)
    print(f"Generated {len(global_answer_letters)} answer letters globally for qas")

    global_letter_idx = 0

    mismatches: List[Mismatch] = []

    qas_col = "qas"
    if qas_col not in df.columns:
        df[qas_col] = ""
    
    # Second pass: Process each row and assign letters
    for idx, row in df.iterrows():
        if (idx + 1) % 10 == 0:
            print(f"Processing row {idx + 1} / {len(df)}...")
        
        num_questions = row_question_counts[idx]
        if num_questions == 0:
            df.at[idx, qas_col] = ""
            continue
        
        # Sample letters for this row from global pool
        row_letters = global_answer_letters[global_letter_idx:global_letter_idx + num_questions]
        global_letter_idx += num_questions
        
        # Shuffle the order of letters within this row (with row-specific seed)
        row_seed = hash((global_seed, idx, row.get('subject_id', ''))) % (2**32)
        random.seed(row_seed)
        shuffled_row_letters = row_letters.copy()
        random.shuffle(shuffled_row_letters)
        
        subject_id = (
            str(row["subject_id"])
            if "subject_id" in df.columns and not pd.isna(row.get("subject_id"))
            else None
        )
        qa_category = (
            str(row["qa_category"])
            if "qa_category" in df.columns and not pd.isna(row.get("qa_category"))
            else None
        )

        step3_text = row[step3_col] if step3_col in df.columns else None
        parsed_step3_qas = {}
        if pd.notna(step3_text) and str(step3_text).strip():
            parsed_step3_qas = parse_qa_text(str(step3_text))

        step4_text = row[step4_col] if step4_col in df.columns else None
        parsed_step4_qas = {}
        if pd.notna(step4_text) and str(step4_text).strip():
            parsed_step4_qas = parse_step4_qa_text(str(step4_text))

        # --- Validate step_4 against step_1 ---
        # (step_3 vs step_1 is already validated in step4_preprocess)
        step1_text = row.get("step_1") if "step_1" in df.columns else None
        if pd.notna(step1_text) and str(step1_text).strip():
            step1_blocks = _split_step1_into_blocks(str(step1_text))
            step1_map = {qnum: block for qnum, block in step1_blocks}
            step1_qnums = sorted(step1_map.keys())

            # Check step_4 question count vs step_1
            step4_qnums = sorted([int(k.split('-')[0]) for k in parsed_step4_qas.keys()])
            if len(step4_qnums) != len(step1_qnums):
                mismatches.append(Mismatch(
                    row_idx=int(idx), subject_id=subject_id, qa_category=qa_category,
                    issue="step4_question_count_mismatch",
                    details=f"step1 has {len(step1_qnums)} Qs ({step1_qnums}), "
                            f"step4 has {len(step4_qnums)} Qs ({step4_qnums})",
                ))

        step3_keys = set(parsed_step3_qas.keys())
        step4_base_keys = set([k.split('-')[0] for k in parsed_step4_qas.keys()])
        common_keys = step3_keys.intersection(step4_base_keys)

        # Error if there are base content questions without matching source questions
        missing_source_keys = sorted(step3_keys - step4_base_keys, key=lambda x: int(x))
        if missing_source_keys:
            subject_id = (
                str(row.get("subject_id", ""))
                if "subject_id" in df.columns
                else ""
            )
            raise ValueError(
                f"Row {idx} (subject_id={subject_id}): base content question(s) "
                f"{missing_source_keys} do not have matching source questions "
                "(missing corresponding Qx-1 in step_4)."
            )

        if not common_keys:
            print(f"Warning: Row {idx}: No matching QAs between step_3 and step_4, skipping")
            df.at[idx, qas_col] = ""
            continue
        
        filtered_step3_qas = {k: v for k, v in parsed_step3_qas.items() if k in common_keys}
        filtered_step4_qas = {k: v for k, v in parsed_step4_qas.items() if k.split('-')[0] in common_keys}
        
        # Combine step_3 and step_4 QAs into a single dict for processing.
        # We iterate by sorted step_3 keys and inline-attach the matching
        # step_4 follow-up so that step_4 keys are not used standalone.
        combined_parsed_qas = {}
        step3_keys_sorted = sorted(filtered_step3_qas.keys(), key=lambda x: int(x))

        for qnum_str in step3_keys_sorted:
            combined_parsed_qas[qnum_str] = filtered_step3_qas[qnum_str]
            step4_key = f"{qnum_str}-1"
            if step4_key in filtered_step4_qas:
                combined_parsed_qas[step4_key] = filtered_step4_qas[step4_key]
        
        # Create QA dict with pre-assigned letters (content + source separately)
        qa_dict = create_qa_dict(combined_parsed_qas, shuffled_row_letters, seed=row_seed)

        # For source questions (step_4 keys like "1-1", "2-1", ...), overwrite the
        # question text with the unified wording, while keeping base (content)
        # questions ("1", "2", ...) unchanged.
        for key in list(qa_dict.keys()):
            if "-" in str(key):
                qa_dict[key]["Question"] = (
                    "What are the minimal exact sources from the patient's "
                    "discharge summaries that fully contain information for "
                    "your previous answer?"
                )

        df.at[idx, qas_col] = json.dumps(qa_dict, ensure_ascii=False)
    
    print(f"\nSaving to {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"Done! Saved {len(df)} rows with unified qas column.")
    print("\nNote: Combined dictionaries have string keys like '1', '1-1', '2', '2-1', etc.")
    print("  step_3 QAs use string keys ('1', '2', '3', ...)")
    print("  step_4 QAs use string keys ('1-1', '2-1', '3-1', ...)")

    # Report mismatches
    if mismatches:
        mismatch_df = pd.DataFrame([m.__dict__ for m in mismatches])
        mismatch_path = output_path.with_name(output_path.stem + "_mismatches.csv")
        mismatch_df.to_csv(mismatch_path, index=False)

        print(f"\nWARNING: Found {len(mismatches)} mismatches. Details saved to {mismatch_path}")
        print("First 10 mismatches:")
        print(mismatch_df.head(10).to_string(index=False))
    else:
        print("\nAll rows passed validation (question counts and texts match step_1).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post-process step_3 and step_4 columns to create structured QA dictionaries"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help=(
            "Input CSV file path (relative to data/ directory or absolute path). "
            "Default: data/sampled_df_grouped_step4.csv"
        ),
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help=(
            "Output CSV file path (relative to data/ directory or absolute path). "
            "Default: data/sampled_df_grouped_step4_post.csv"
        ),
    )

    args = parser.parse_args()
    main(input_file=args.input_file, output_file=args.output_file)
