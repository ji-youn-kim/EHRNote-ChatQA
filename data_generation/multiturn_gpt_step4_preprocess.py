#!/usr/bin/env python
# coding: utf-8

"""
Post-process step_3 column by adding answer evidence and source location from step_1.

For each row:
1. Parse step_3 to extract only questions (Qx:) and answers (Ax:)
2. Load step_1 and swap headers:
   - [Entities, Attributes, Relationships] -> [Answer Evidence]
   - [Source Location] -> [Answer Source]
3. For each QA from step_3, add the corresponding answer evidence and source location from step_1
"""

import re
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Make sibling subfolders importable when this script is launched directly.
import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _sub in ("core", "data_generation", "evaluation", "scoring", ""):
    _p = str(_PROJECT_ROOT / _sub) if _sub else str(_PROJECT_ROOT)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from multiturn_gpt import DATA_DIR

# Regex patterns for parsing
_Q_PATTERN = re.compile(r"Q(\d+):\s*(.*?)(?=Q\d+:|$)", re.DOTALL)
_A_PATTERN = re.compile(r"A(\d+):\s*(.*?)(?=A\d+:|I\d+:|Q\d+:|$)", re.DOTALL)


def swap_step1_headers(text: str) -> str:
    """
    Swap headers in step_1 text:
    - [Entities, Attributes, Relationships] -> [Answer Evidence]
    - [Source Location] -> [Answer Source]

    Uses regex to handle LLM output variations (spacing, case, typos).
    """
    if pd.isna(text) or not text.strip():
        return text

    text = str(text)
    text = re.sub(
        r"\[['\"]?Entit\w*['\"]?[,\s]+['\"]?\w*tt\w*['\"]?[,\s]+['\"]?[Rr]elationships['\"]?\]",
        "[Answer Evidence]",
        text,
    )
    text = text.replace("[Source Location]", "[Answer Source]")
    return text


def split_into_question_blocks(text: str) -> List[Tuple[int, str]]:
    """
    Split a multi-question string into ordered (qnum, block_text) tuples.
    Questions must be marked like "Q1:" at the start of a line or anywhere in text.
    """
    if pd.isna(text) or not text.strip():
        return []
    
    matches = list(_Q_PATTERN.finditer(text))
    if not matches:
        return []
    
    blocks: List[Tuple[int, str]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        qnum = int(m.group(1))
        block = text[start:end].rstrip()
        blocks.append((qnum, block))
    
    return blocks


def extract_answer_evidence_and_source(step1_block: str) -> Tuple[str, str]:
    """
    Extract [Answer Evidence] and [Answer Source] sections from a step_1 question block.
    
    Note: The headers should already be swapped to [Answer Evidence] and [Answer Source].
    
    Format expected:
    Q1: question
    A1: 
    [Answer Evidence]
    ...content...
    [Answer Source]
    ...content...
    
    Returns:
        Tuple of (answer_evidence_text, answer_source_text)
    """
    if not step1_block:
        return "", ""
    
    # Pattern to match [Answer Evidence] section (everything after the header until [Answer Source] or end)
    evidence_pattern = re.compile(
        r"\[Answer Evidence\]\s*\n?(.*?)(?=\[Answer Source\]|$)",
        re.DOTALL
    )
    
    # Pattern to match [Answer Source] section (everything after the header until [Answer] header, next Q, or end)
    source_pattern = re.compile(
        r"\[Answer Source\]\s*\n?(.*?)(?=\[Answer\]|Q\d+:|$)",
        re.DOTALL,
    )
    
    evidence_match = evidence_pattern.search(step1_block)
    source_match = source_pattern.search(step1_block)
    
    answer_evidence = evidence_match.group(1).strip() if evidence_match else ""
    answer_source = source_match.group(1).strip() if source_match else ""
    
    return answer_evidence, answer_source


def parse_step3_qa_only(text: str) -> Dict[int, Dict[str, str]]:
    """
    Parse step_3 text to extract only questions and answers.
    
    Format (step_3 includes reasoning lines):
        Q1: question
        A1: answer
        A1-Reasoning: ...
        I1: ...
        I1-Reasoning: ...
        ...
        Q2: ...
    
    Returns a dictionary: {qnum: {'question': str, 'answer': str}}
    """
    if pd.isna(text) or not text.strip():
        return {}
    
    text = str(text)
    
    # Find all question markers
    q_matches = list(_Q_PATTERN.finditer(text))
    if not q_matches:
        return {}
    
    qas = {}
    
    for i, q_match in enumerate(q_matches):
        qnum = int(q_match.group(1))
        
        # Get the section for this question (from Q to next Q or end)
        start_pos = q_match.start()
        end_pos = q_matches[i + 1].start() if i + 1 < len(q_matches) else len(text)
        question_section = text[start_pos:end_pos]
        
        # Extract question text (after Q{num}:)
        question_match = re.match(rf"Q{qnum}:\s*(.*?)(?=A{qnum}:|$)", question_section, re.DOTALL)
        question_text = question_match.group(1).strip() if question_match else ""
        
        # Extract answer (A{qnum}:), stopping before:
        #   - the corresponding reasoning block (A{qnum}-Reasoning:)
        #   - any distractor (I\d+:)
        #   - the next question (Q\d+:)
        #   - or end of text
        a_pattern = re.compile(
            rf"A{qnum}:\s*(.*?)(?=\nA{qnum}-Reasoning:|\nI\d+:|\nQ\d+:|$)",
            re.DOTALL,
        )
        a_match = a_pattern.search(question_section)
        answer = a_match.group(1).strip() if a_match else ""
        
        if question_text and answer:
            qas[qnum] = {
                'question': question_text,
                'answer': answer
            }
    
    return qas


def _extract_question_text_from_step1_block(block: str) -> str:
    """Extract just the question text (first line only) from a step_1 block.

    Later pipeline steps may insert extra content (distractor analysis, MCQ
    options, etc.) between the question line and the A{n}: answer line.
    Taking only the first line avoids false-positive text mismatches.
    """
    m = re.match(r"Q\d+:\s*(.*?)(?:\n|$)", block)
    return m.group(1).strip() if m else ""


@dataclass(frozen=True)
class Mismatch:
    row_idx: int
    subject_id: Optional[str]
    qa_category: Optional[str]
    issue: str
    details: str


def merge_step3_with_step1(
    step3_qas: Dict[int, Dict[str, str]],
    step1_blocks: List[Tuple[int, str]]
) -> str:
    """
    Merge step_3 QAs with answer evidence and source location from step_1.
    
    Args:
        step3_qas: Dictionary from parse_step3_qa_only
        step1_blocks: List of (qnum, block_text) tuples from step_1
    
    Returns:
        Merged text with format:
        Q1: question
        A1: answer
        [Answer Evidence]
        ...
        [Answer Source]
        ...
        Q2: question
        A2: answer
        ...
    """
    step1_map = {qnum: block for qnum, block in step1_blocks}
    
    merged_lines = []
    
    for qnum in sorted(step3_qas.keys()):
        qa = step3_qas[qnum]
        question = qa['question']
        answer = qa['answer']
        
        merged_lines.append(f"Q{qnum}: {question}")
        merged_lines.append(f"A{qnum}: {answer}")
        
        step1_block = step1_map.get(qnum, "")
        if step1_block:
            answer_evidence, answer_source = extract_answer_evidence_and_source(step1_block)
            
            if answer_evidence:
                merged_lines.append("")
                merged_lines.append("[Answer Evidence]")
                merged_lines.append(answer_evidence)
            
            if answer_source:
                merged_lines.append("")
                merged_lines.append("[Answer Source]")
                merged_lines.append(answer_source)
        
        merged_lines.append("")
    
    return "\n".join(merged_lines).strip()


def main(input_file: Optional[str] = None, output_file: Optional[str] = None) -> None:
    """
    Merge step_3 with answer evidence/source from step_1 into step_4_pre.

    Parameters mirror multiturn_gpt_step1.py:
      - input_file: CSV path (relative to DATA_DIR or absolute).
      - output_file: CSV path (relative to DATA_DIR or absolute).
    """
    if input_file is None:
        input_path = DATA_DIR / "sampled_df_grouped_step3.csv"
    else:
        input_path = Path(input_file)
        if not input_path.is_absolute():
            input_path = DATA_DIR / input_path

    if output_file is None:
        output_path = DATA_DIR / "sampled_df_grouped_step4_pre.csv"
    else:
        output_path = Path(output_file)
        if not output_path.is_absolute():
            output_path = DATA_DIR / output_path

    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)

    print(f"Found {len(df)} rows")

    output_col = "step_4_pre"
    if output_col not in df.columns:
        df[output_col] = ""

    mismatches: List[Mismatch] = []

    for idx, row in df.iterrows():
        if (idx + 1) % 10 == 0:
            print(f"Processing row {idx + 1} / {len(df)}...")

        subject_id = (
            str(row["subject_id"])
            if "subject_id" in df.columns and not pd.isna(row["subject_id"])
            else None
        )
        qa_category = (
            str(row["qa_category"])
            if "qa_category" in df.columns and not pd.isna(row["qa_category"])
            else None
        )

        step3_col = "step_3"
        step1_col = "step_1"

        if step3_col not in df.columns or step1_col not in df.columns:
            print(f"Warning: Missing {step3_col} or {step1_col} column for row {idx}, skipping...")
            continue

        step3_text = row[step3_col]
        step1_text = row[step1_col]

        if pd.isna(step3_text) or not str(step3_text).strip():
            df.at[idx, output_col] = ""
            continue

        if pd.isna(step1_text) or not str(step1_text).strip():
            # If step1 is empty, just use step3 QAs without evidence
            step3_qas = parse_step3_qa_only(str(step3_text))
            merged_text = "\n".join([
                f"Q{qnum}: {qa['question']}\nA{qnum}: {qa['answer']}"
                for qnum, qa in sorted(step3_qas.items())
            ])
            df.at[idx, output_col] = merged_text
            continue

        step3_qas = parse_step3_qa_only(str(step3_text))
        if not step3_qas:
            df.at[idx, output_col] = ""
            continue

        step1_text_swapped = swap_step1_headers(str(step1_text))

        step1_blocks = split_into_question_blocks(step1_text_swapped)
        step1_map = {qnum: block for qnum, block in step1_blocks}
        step1_qnums = sorted(step1_map.keys())
        step3_qnums = sorted(step3_qas.keys())

        # --- Validation ---
        # 1. Question count mismatch
        if len(step3_qnums) != len(step1_qnums):
            mismatches.append(Mismatch(
                row_idx=int(idx), subject_id=subject_id, qa_category=qa_category,
                issue="question_count_mismatch",
                details=f"step1 has {len(step1_qnums)} Qs ({step1_qnums}), "
                        f"step3 has {len(step3_qnums)} Qs ({step3_qnums})",
            ))

        # 2. Question number mismatch
        missing_in_step1 = sorted(set(step3_qnums) - set(step1_qnums))
        if missing_in_step1:
            mismatches.append(Mismatch(
                row_idx=int(idx), subject_id=subject_id, qa_category=qa_category,
                issue="qnums_missing_in_step1",
                details=f"Q{missing_in_step1} in step3 but not in step1",
            ))

        # 3. Question text mismatch
        for qnum in step3_qnums:
            if qnum not in step1_map:
                continue
            step1_q_text = _extract_question_text_from_step1_block(step1_map[qnum])
            step3_q_text = step3_qas[qnum]["question"]
            if step1_q_text != step3_q_text:
                mismatches.append(Mismatch(
                    row_idx=int(idx), subject_id=subject_id, qa_category=qa_category,
                    issue="question_text_mismatch",
                    details=f"Q{qnum}: step1={step1_q_text[:80]!r}... "
                            f"vs step3={step3_q_text[:80]!r}...",
                ))

        # 4. Missing answer evidence or answer source
        for qnum in step3_qnums:
            step1_block = step1_map.get(qnum, "")
            if not step1_block:
                continue
            evidence, source = extract_answer_evidence_and_source(step1_block)
            missing_parts = []
            if not evidence:
                missing_parts.append("answer_evidence")
            if not source:
                missing_parts.append("answer_source")
            if missing_parts:
                mismatches.append(Mismatch(
                    row_idx=int(idx), subject_id=subject_id, qa_category=qa_category,
                    issue="missing_evidence_or_source",
                    details=f"Q{qnum}: missing {', '.join(missing_parts)}",
                ))

        merged_text = merge_step3_with_step1(step3_qas, step1_blocks)

        df.at[idx, output_col] = merged_text

    print(f"\nSaving to {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"Done! Saved {len(df)} rows with unified step_4_pre column.")

    # Report mismatches
    if mismatches:
        mismatch_df = pd.DataFrame([m.__dict__ for m in mismatches])
        mismatch_path = output_path.with_name(output_path.stem + "_mismatches.csv")
        mismatch_df.to_csv(mismatch_path, index=False)

        print(f"\nWARNING: Found {len(mismatches)} mismatches. Details saved to {mismatch_path}")
        print("Mismatches:")
        print(mismatch_df.to_string(index=False))
    else:
        print("\nAll rows passed validation (question counts, texts, evidence, and source all match).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge step_3 QA with answer evidence/source from step_1 into step_4_pre"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help=(
            "Input CSV file path (relative to data/ directory or absolute path). "
            "Default: data/sampled_df_grouped_step3.csv"
        ),
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help=(
            "Output CSV file path (relative to data/ directory or absolute path). "
            "Default: data/sampled_df_grouped_step4_pre.csv"
        ),
    )

    args = parser.parse_args()
    main(input_file=args.input_file, output_file=args.output_file)
