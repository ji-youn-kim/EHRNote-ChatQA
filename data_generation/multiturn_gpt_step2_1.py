"""
Post-process after `multiturn_gpt_step2.py` when using unified columns.

Adds a single `step_2_1` column by merging, for each row:
- `step_1` (with header swaps)
- the per-question distractor guidance parsed from `step_2`

Rules:
1) `step_1` text:
   - swap "[Entities, Attributes, Relationships]" -> "[Answer Evidence]"
   - swap "[Source Location]" -> "[Answer Source]"
2) `step_2` text:
   - for each question, extract from "Effective Distractor Type 1:" up to
     (but not including) the next question ("Q{n}:") or end of text.
3) Append that extracted distractor block to the corresponding question block
   in the (modified) step_1 text.

If question counts (or numbering) mismatch between step_1 and step_2, the script
prints a warning and writes a mismatch report CSV.
"""

from __future__ import annotations

import argparse
import re
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


_Q_MARKER_RE = re.compile(r"(?m)^\s*Q(\d+):")
# In step-2 CSV, lines start with "- Effective Distractor Type X: ...".
# Allow an optional leading "-" (with surrounding whitespace) before the phrase.
_DISTRACTOR_START_RE = re.compile(r"(?m)^\s*-?\s*Effective Distractor Type 1:")
# For Q2+ questions, the step-2 prompt adds Referential Ambiguity Analysis and
# Cross-Turn Error Path Analysis sections before the distractor types.
_REFERENTIAL_AMBIGUITY_RE = re.compile(r"(?m)^\s*-?\s*Referential Ambiguity Analysis:")
_CROSS_TURN_ERROR_RE = re.compile(r"(?m)^\s*-?\s*Cross-Turn Error Path Analysis:")
_EFFECTIVE_DISTRACTOR_HEADER = "[Effective Distractor Construction Content]"
# In step-1 text, each question block ends with an [Answer] section we want to drop.
_ANSWER_HEADER_RE = re.compile(r"(?m)^\s*\[Answer\]\s*$")


def _swap_step1_headers(text: str) -> str:
    # Rename the step-1 section headers to the wording used from step 2_1 on.
    return (
        text.replace("[Entities, Attributes, Relationships]", "[Answer Evidence]")
        .replace("[Source Location]", "[Answer Source]")
    )


def _split_into_question_blocks(text: str) -> List[Tuple[int, str]]:
    """
    Split a multi-question string into ordered (qnum, block_text) tuples.
    Requires questions to be marked like "Q1:" at the start of a line.
    """
    matches = list(_Q_MARKER_RE.finditer(text))
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


def _extract_distractor_block_from_question(step2_question_block: str) -> str:
    """
    From a single question block in step_2, extract the distractor-related content.

    For Q2+ questions the step-2 prompt produces:
      - Referential Ambiguity Analysis: ...
      - Cross-Turn Error Path Analysis: ...
      - Effective Distractor Type 1: ...
      ...
    For Q1 it starts directly at Effective Distractor Type 1.

    We extract from whichever section appears first (Referential Ambiguity,
    Cross-Turn Error Path, or Effective Distractor Type 1) to the end of the block.
    """
    # Find the earliest starting position among the three possible section headers.
    candidates = []
    for pattern in (_REFERENTIAL_AMBIGUITY_RE, _CROSS_TURN_ERROR_RE, _DISTRACTOR_START_RE):
        m = pattern.search(step2_question_block)
        if m:
            candidates.append(m.start())

    if not candidates:
        return ""

    start = min(candidates)
    extracted = step2_question_block[start:].strip()
    return _EFFECTIVE_DISTRACTOR_HEADER + "\n" + extracted


@dataclass(frozen=True)
class Mismatch:
    row_idx: int
    subject_id: Optional[str]
    qa_category: Optional[str]
    step1_qnums: str
    step2_qnums: str
    missing_in_step2: str
    extra_in_step2: str
    no_distractor_in_step2: str
    question_text_mismatch: str


def _extract_question_text(block: str) -> str:
    """Extract just the question text (first line only) from a question block.

    In step2 output, distractor analysis lines (e.g. '- Effective Distractor
    Type ...', '- Referential Ambiguity Analysis: ...') appear between the
    question line and the A{n}: answer line. Taking only the first line after
    Q{n}: keeps them out of the text comparison, which would otherwise report
    false-positive mismatches.
    """
    m = re.match(r"Q\d+:\s*(.*?)(?:\n|$)", block)
    return m.group(1).strip() if m else ""


def _strip_answer_from_step1_block(block: str) -> str:
    """
    Remove the [Answer] section and everything after it from a single
    step_1 question block. Keeps Q*, [Answer Evidence], and [Answer Source].
    """
    m = _ANSWER_HEADER_RE.search(block)
    if not m:
        return block.rstrip()
    return block[: m.start()].rstrip()


def _merge_step1_and_step2(step1_text: str, step2_text: str) -> Tuple[str, List[str], List[str], List[str]]:
    """
    Returns:
      merged_text,
      step1_qnums (string list),
      step2_qnums (string list),
      qnums_with_no_distractor (string list)
    """
    step1_blocks = _split_into_question_blocks(step1_text)
    step2_blocks = _split_into_question_blocks(step2_text)

    step1_qnums = [q for q, _ in step1_blocks]
    step2_qnums = [q for q, _ in step2_blocks]

    # Map step2 qnum -> extracted distractor guidance
    step2_distractors: Dict[int, str] = {}
    qnums_with_no_distractor: List[int] = []
    for qnum, block in step2_blocks:
        dist = _extract_distractor_block_from_question(block)
        if not dist:
            qnums_with_no_distractor.append(qnum)
        step2_distractors[qnum] = dist

    merged_blocks: List[str] = []
    for qnum, block in step1_blocks:
        # Drop the [Answer] section from the step_1 block; we only want
        # the question, answer evidence, and answer source in step_2_1.
        base_block = _strip_answer_from_step1_block(block)
        dist = step2_distractors.get(qnum, "")
        if dist:
            merged_blocks.append(base_block + "\n\n" + dist.strip())
        else:
            merged_blocks.append(base_block)

    merged_text = "\n\n".join(b for b in merged_blocks if b.strip()).strip()
    return (
        merged_text,
        [str(x) for x in step1_qnums],
        [str(x) for x in step2_qnums],
        [str(x) for x in qnums_with_no_distractor],
    )


def main(input_file: Optional[str] = None, output_file: Optional[str] = None) -> None:
    """
    Merge step_1 and step_2 into step_2_1.

    Parameters mirror multiturn_gpt_step1.py:
      - input_file: CSV path (relative to DATA_DIR or absolute).
      - output_file: CSV path (relative to DATA_DIR or absolute).
    """
    if input_file is None:
        input_path = DATA_DIR / "sampled_df_grouped_step2.csv"
    else:
        input_path = Path(input_file)
        if not input_path.is_absolute():
            input_path = DATA_DIR / input_path

    if output_file is None:
        output_path = DATA_DIR / "sampled_df_grouped_step2_1.csv"
    else:
        output_path = Path(output_file)
        if not output_path.is_absolute():
            output_path = DATA_DIR / output_path

    print(f"Loading input file: {input_path}")
    df = pd.read_csv(input_path)

    out_col = "step_2_1"
    if out_col not in df.columns:
        df[out_col] = ""

    mismatches: List[Mismatch] = []

    for idx, row in df.iterrows():
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

        step1_raw = row.get("step_1")
        step2_raw = row.get("step_2")

        if (
            pd.isna(step1_raw)
            or not isinstance(step1_raw, str)
            or not step1_raw.strip()
        ):
            continue
        if (
            pd.isna(step2_raw)
            or not isinstance(step2_raw, str)
            or not step2_raw.strip()
        ):
            continue

        # Skip if already populated (and already contains the new distractor header)
        existing = row.get(out_col)
        if isinstance(existing, str) and existing.strip() and (
            _EFFECTIVE_DISTRACTOR_HEADER in existing
        ):
            continue

        step1_text = _swap_step1_headers(step1_raw)
        step2_text = step2_raw

        step1_blocks = _split_into_question_blocks(step1_text)
        step2_blocks = _split_into_question_blocks(step2_text)

        merged_text, step1_qnums, step2_qnums, no_dist_qnums = _merge_step1_and_step2(
            step1_text, step2_text
        )

        # Detect mismatches:
        s1_set = set(step1_qnums)
        s2_set = set(step2_qnums)
        missing_in_step2 = sorted(s1_set - s2_set, key=lambda x: int(x))
        extra_in_step2 = sorted(s2_set - s1_set, key=lambda x: int(x))

        step1_block_map = {q: b for q, b in step1_blocks}
        step2_block_map = {q: b for q, b in step2_blocks}
        text_mismatch_qnums: List[str] = []
        for qnum_str in step1_qnums:
            qnum_int = int(qnum_str)
            if qnum_int in step2_block_map:
                s1_q = _extract_question_text(step1_block_map[qnum_int])
                s2_q = _extract_question_text(step2_block_map[qnum_int])
                if s1_q != s2_q:
                    text_mismatch_qnums.append(qnum_str)

        if (
            (len(step1_blocks) != len(step2_blocks))
            or (step1_qnums != step2_qnums)
            or missing_in_step2
            or extra_in_step2
            or no_dist_qnums
            or text_mismatch_qnums
        ):
            mismatches.append(
                Mismatch(
                    row_idx=int(idx),
                    subject_id=subject_id,
                    qa_category=qa_category,
                    step1_qnums=",".join(step1_qnums),
                    step2_qnums=",".join(step2_qnums),
                    missing_in_step2=",".join(missing_in_step2),
                    extra_in_step2=",".join(extra_in_step2),
                    no_distractor_in_step2=",".join(no_dist_qnums),
                    question_text_mismatch=",".join(text_mismatch_qnums),
                )
            )

        df.at[idx, out_col] = merged_text

    df.to_csv(output_path, index=False)
    print(f"Saved updated CSV (with step_2_1_* columns) to {output_path}")

    if mismatches:
        mismatch_df = pd.DataFrame([m.__dict__ for m in mismatches])
        mismatch_path = output_path.with_name(output_path.stem + "_mismatches.csv")
        mismatch_df.to_csv(mismatch_path, index=False)

        print(f"WARNING: Found {len(mismatches)} row/category mismatches. Details saved to {mismatch_path}")
        print("First 10 mismatches:")
        print(mismatch_df.head(10).to_string(index=False))
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge step_1 and step_2 into step_2_1 using unified columns"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help=(
            "Input CSV file path (relative to data/ directory or absolute path). "
            "Default: data/sampled_df_grouped_step2.csv"
        ),
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help=(
            "Output CSV file path (relative to data/ directory or absolute path). "
            "Default: data/sampled_df_grouped_step2_1.csv"
        ),
    )

    args = parser.parse_args()
    main(input_file=args.input_file, output_file=args.output_file)

