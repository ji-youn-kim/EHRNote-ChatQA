#!/usr/bin/env python
# coding: utf-8

"""
Score multi-turn QA model answers against correct answers.

Gold answers come from the released ``questions.csv``; model outputs come from
the per-model answer CSVs written by the evaluation scripts (one row per
subject, with the per-turn outputs as a JSON blob in ``model_answer``).

Multiple-choice runs (default). For each subject:
1. Parse model answers from the model_answer column
2. Extract the letter (A, B, C, D, or E) by parsing the model answer text
3. Compare with the correct answers for that subject in questions.csv
4. Calculate overall accuracy and sample-level accuracy

Free-text generation runs (``--generate``). Free-text answers carry no letter,
so they are scored from the LLM-judge output produced by
``evaluation/multiturn_gpt_evaluate_generation_eval.py``: this script parses the
``Score{key}: {0-2}`` lines out of the judge column and normalises them to the
same accuracy fields the multiple-choice path reports.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Make sibling subfolders importable when this script is launched directly.
import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _sub in ("core", "data_generation", "evaluation", "scoring", ""):
    _p = str(_PROJECT_ROOT / _sub) if _sub else str(_PROJECT_ROOT)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from multiturn_gpt import DATA_DIR
from release_data import load_release_samples
from model_answer_parsing import extract_answer_letter
from model_ordering import MODEL_ORDER, MODEL_ORDER_MAP


# ---------------------------------------------------------------------------
# Generation-mode (LLM judge) scoring
# ---------------------------------------------------------------------------

# Strict pattern for the judge output, matching the exact format specified in
# prompt/multiturn_gpt_evaluate_generate_eval_content.txt and
# prompt/multiturn_gpt_evaluate_generate_eval_source.txt:
#
#   Score1: 2          (content question, key 1, score 0-2)
#   Score1-1: 2        (source question, key 1-1, score 0-2)
#
# Requirements (all enforced):
#   - The line must START with "Score" (after optional leading whitespace).
#     The line is anchored with ^[ \t]* + [ \t]*$ to reject mid-sentence matches.
#   - No curly braces around the key  (rejects "Score{1}: 2").
#   - No "Q" prefix                   (rejects "Score Q1: 2").
#   - No markdown bold                (rejects "**Score1:** 2").
#   - No whitespace between "Score" and the key.
#   - Mandatory colon after the key.
#   - Inner whitespace is [ \t]* (horizontal only) so the value cannot live on a
#     separate line from the key -- `\s*` would swallow `\n` and parse
#     `Score1:\n3` as score=3 for key 1.
#   - The value is one or more digits; range validation (0-2) happens in
#     parse_gpt_eval_scores. Out-of-range values are skipped and surface as a
#     missing key, which excludes the row.
_GPT_EVAL_SCORE_PATTERN = re.compile(
    r"^[ \t]*Score(\d+(?:-\d+)?):[ \t]*(\d+)[ \t]*$",
    re.MULTILINE,
)

# Max scores for the multi-point rubrics (both are 0-2).
CONTENT_MAX_SCORE = 2  # Content questions scored 0-2
SOURCE_MAX_SCORE = 2   # Source-citation questions scored 0-2


def parse_gpt_eval_scores(eval_text: str) -> Dict[str, int]:
    """
    Parse LLM-judge output into per-question scores under the strict
    line-anchored format specified by the eval prompts.

    Expected lines (each on its own line, no markdown, no curly braces):
        Score1: 2          (content question 1, integer 0-2)
        Score2-1: 2        (source question 2-1, integer 0-2)

    Returns dict mapping turn key (e.g. "1", "1-1") to integer score.

    Out-of-range values are silently skipped -- the score for that key will be
    missing from the returned dict, which causes the missing-key detector in
    evaluate_single_csv_generate() to flag the row for investigation instead of
    silently scoring it as 0. Duplicate keys: the LAST occurrence wins.
    """
    if not eval_text or not str(eval_text).strip():
        return {}
    scores: Dict[str, int] = {}
    for m in _GPT_EVAL_SCORE_PATTERN.finditer(eval_text):
        key = m.group(1)
        val = int(m.group(2))
        max_score = SOURCE_MAX_SCORE if "-" in key else CONTENT_MAX_SCORE
        if val < 0 or val > max_score:
            continue
        scores[key] = val
    return scores


def _sort_qa_keys(keys: List[str]) -> List[str]:
    """Sort QA keys in order: 1, 1-1, 2, 2-1, ..."""
    return sorted(
        keys,
        key=lambda x: (
            int(x.split("-")[0]) if "-" in x else int(x),
            1 if "-" in x else 0,
            int(x.split("-")[1]) if "-" in x else 0,
        ),
    )


def score_qa_answers_individual(
    qa_dict: Dict[str, Dict[str, str]],
    model_answers_dict: Dict[str, str],
    is_deepseek: bool,
    is_gemini: bool = False,
) -> Dict[str, bool]:
    """
    Score model answers against correct answers (individual QAs only - step_3 QAs).
    
    Args:
        qa_dict: Dictionary from qas_{category} with structure {key: {'correct_answer': 'A', ...}}
                 Keys are strings like "1", "1-1", "2", "2-1", etc.
        model_answers_dict: Dictionary from model_answer_{category} with structure {key: 'model_answer_text'}
    
    Returns:
        Dictionary mapping key to whether the answer is correct: {key: True/False}
        Only includes step_3 QAs (keys without "-1")
    """
    scores = {}
    
    # Only score step_3 QAs (keys without "-1")
    for key in qa_dict.keys():
        if '-' in key:  # Skip step_4 QAs (keys like "1-1", "2-1")
            continue
            
        if key not in model_answers_dict:
            scores[key] = False
            continue
        
        correct_answer = qa_dict[key].get('correct_answer', '').upper()
        if not correct_answer:
            scores[key] = False
            continue
        
        model_answer_text = model_answers_dict[key]
        model_answer_letter = extract_answer_letter(model_answer_text, is_deepseek, is_gemini)
        
        is_correct = (model_answer_letter == correct_answer)
        scores[key] = is_correct
    
    return scores


def score_qa_answers_source_individual(
    qa_dict: Dict[str, Dict[str, str]],
    model_answers_dict: Dict[str, str],
    is_deepseek: bool,
    is_gemini: bool = False,
) -> Dict[str, bool]:
    """
    Score model answers against correct answers (source QAs only - step_4 QAs).
    
    Args:
        qa_dict: Dictionary from qas_{category} with structure {key: {'correct_answer': 'A', ...}}
                 Keys are strings like "1", "1-1", "2", "2-1", etc.
        model_answers_dict: Dictionary from model_answer_{category} with structure {key: 'model_answer_text'}
    
    Returns:
        Dictionary mapping key to whether the answer is correct: {key: True/False}
        Only includes step_4 QAs (keys with "-", like "1-1", "2-1")
    """
    scores = {}
    
    # Only score step_4 QAs (keys with "-")
    for key in qa_dict.keys():
        if '-' not in key:  # Skip step_3 QAs (keys like "1", "2")
            continue
            
        if key not in model_answers_dict:
            scores[key] = False
            continue
        
        correct_answer = qa_dict[key].get('correct_answer', '').upper()
        if not correct_answer:
            scores[key] = False
            continue
        
        model_answer_text = model_answers_dict[key]
        model_answer_letter = extract_answer_letter(model_answer_text, is_deepseek, is_gemini)
        
        is_correct = (model_answer_letter == correct_answer)
        scores[key] = is_correct
    
    return scores


def score_qa_answers_paired(
    qa_dict: Dict[str, Dict[str, str]],
    model_answers_dict: Dict[str, str],
    is_deepseek: bool,
    is_gemini: bool = False,
) -> Dict[str, bool]:
    """
    Score model answers in pairs: both N and N-1 must be correct.
    
    Args:
        qa_dict: Dictionary from qas_{category} with structure {key: {'correct_answer': 'A', ...}}
                 Keys are strings like "1", "1-1", "2", "2-1", etc.
        model_answers_dict: Dictionary from model_answer_{category} with structure {key: 'model_answer_text'}
    
    Returns:
        Dictionary mapping base number to whether the pair is correct: {"1": True/False, "2": True/False, ...}
        A pair is correct only if both "N" and "N-1" are correct.
    """
    paired_scores = {}
    
    step3_keys = [key for key in qa_dict.keys() if '-' not in key]
    
    for base_key in step3_keys:
        step4_key = f"{base_key}-1"
        
        if base_key not in qa_dict or base_key not in model_answers_dict:
            paired_scores[base_key] = False
            continue
        
        if step4_key not in qa_dict or step4_key not in model_answers_dict:
            paired_scores[base_key] = False
            continue
        
        correct_answer_3 = qa_dict[base_key].get('correct_answer', '').upper()
        model_answer_text_3 = model_answers_dict[base_key]
        model_answer_letter_3 = extract_answer_letter(model_answer_text_3, is_deepseek, is_gemini)
        is_correct_3 = (model_answer_letter_3 == correct_answer_3)
        
        correct_answer_4 = qa_dict[step4_key].get('correct_answer', '').upper()
        model_answer_text_4 = model_answers_dict[step4_key]
        model_answer_letter_4 = extract_answer_letter(model_answer_text_4, is_deepseek, is_gemini)
        is_correct_4 = (model_answer_letter_4 == correct_answer_4)
        
        paired_scores[base_key] = is_correct_3 and is_correct_4
    
    return paired_scores


def evaluate_single_csv_generate(
    csv_path: Path,
    gold_by_subject: Dict[int, Dict[str, Dict[str, str]]],
    eval_column: str = "gpt_eval",
) -> Dict:
    """
    Evaluate a single free-text generation CSV that carries an LLM-judge column
    (e.g. gpt_eval or gemini_eval), written by
    evaluation/multiturn_gpt_evaluate_generation_eval.py.

    *gold_by_subject* maps subject_id to that subject's turn dict, loaded from
    the released questions.csv; it defines which turns must carry a score.

    Parses ``Score1: 0-2``, ``Score1-1: 0-2``, ... and returns metrics in the
    same shape as evaluate_single_csv() so both answer formats share one summary.

    Content and source metrics use partial credit (score / max_score), while the
    paired metric is a strict AND: a turn pair counts only when BOTH the content
    question and its source follow-up earn full marks. That matches
    score_qa_answers_paired() on the multiple-choice path, so the paired column
    means the same thing in both modes.

    A row is EXCLUDED from aggregation (and reported) when its judge output is
    empty, unparseable, missing an expected key, or carries an unexpected key --
    otherwise a judge failure would silently register as a score of 0.
    """
    df = pd.read_csv(csv_path)
    if eval_column not in df.columns:
        for fallback in ("gpt_eval", "gemini_eval"):
            if fallback in df.columns:
                eval_column = fallback
                break
    if eval_column not in df.columns:
        raise ValueError(
            f"CSV must contain '{eval_column}', 'gpt_eval', or 'gemini_eval' "
            "column for generate-mode scoring."
        )
    if "subject_id" not in df.columns:
        raise ValueError("CSV must contain a 'subject_id' column.")

    print(f"\nEvaluating (generate mode) {csv_path.name} using column '{eval_column}'")
    print(f"  Loaded {len(df)} rows")

    invalid_eval_row_count = 0
    invalid_eval_details: List[Dict[str, str]] = []
    has_invalid_eval = False
    # Rows that parsed cleanly and contributed to the metric.
    valid_row_count = 0

    total_qa_count_content_individual = 0
    correct_qa_count_content_individual = 0.0
    sample_level_correct_content_individual = 0
    sample_level_total_content_individual = 0
    sample_ratio_sum_content_individual = 0.0
    sample_ratio_count_content_individual = 0

    total_qa_count_source_individual = 0
    correct_qa_count_source_individual = 0.0
    sample_level_correct_source_individual = 0
    sample_level_total_source_individual = 0
    sample_ratio_sum_source_individual = 0.0
    sample_ratio_count_source_individual = 0

    total_qa_count_paired = 0
    correct_qa_count_paired = 0.0
    sample_level_correct_paired = 0
    sample_level_total_paired = 0
    sample_ratio_sum_paired = 0.0
    sample_ratio_count_paired = 0

    for idx, row in df.iterrows():
        if pd.isna(row.get("subject_id")):
            continue
        subject_id = int(row["subject_id"])
        subject_id_val = str(subject_id)
        qa_category_val = (
            str(row["qa_category"])
            if "qa_category" in df.columns and not pd.isna(row.get("qa_category"))
            else ""
        )

        qa_dict = gold_by_subject.get(subject_id)
        if not qa_dict:
            print(f"  Warning: subject {subject_id} is not in questions.csv; skipping row {idx}.")
            continue

        eval_val = row.get(eval_column, "")
        if pd.isna(eval_val) or not str(eval_val).strip():
            invalid_eval_row_count += 1
            has_invalid_eval = True
            invalid_eval_details.append({
                "row_index": str(idx),
                "subject_id": subject_id_val,
                "qa_category": qa_category_val,
                "reason": "empty_or_nan_eval",
                "eval_value": "" if pd.isna(eval_val) else str(eval_val),
            })
            continue

        scores = parse_gpt_eval_scores(str(eval_val))
        if not scores:
            invalid_eval_row_count += 1
            has_invalid_eval = True
            invalid_eval_details.append({
                "row_index": str(idx),
                "subject_id": subject_id_val,
                "qa_category": qa_category_val,
                "reason": "no_scores_parsed",
                "eval_value": str(eval_val),
            })
            continue

        sorted_keys = _sort_qa_keys(list(qa_dict.keys()))
        content_keys = [k for k in sorted_keys if "-" not in k]
        source_keys = [k for k in sorted_keys if "-" in k]

        # Partial parse: the judge did not score every question in qa_dict.
        # Without this check the missing keys would be silently counted as 0.
        missing_content_keys = [k for k in content_keys if k not in scores]
        missing_source_keys = [k for k in source_keys if k not in scores]
        if missing_content_keys or missing_source_keys:
            invalid_eval_row_count += 1
            has_invalid_eval = True
            invalid_eval_details.append({
                "row_index": str(idx),
                "subject_id": subject_id_val,
                "qa_category": qa_category_val,
                "reason": "partial_parse_failure",
                "missing_content_keys": ",".join(missing_content_keys),
                "missing_source_keys": ",".join(missing_source_keys),
                "expected_content_keys": ",".join(content_keys),
                "expected_source_keys": ",".join(source_keys),
                "parsed_keys": ",".join(sorted(scores.keys())),
                "eval_value": str(eval_val),
            })
            continue

        # Extra keys: the judge scored a question that does not exist (e.g.
        # wrote Score10 when there were only 9 questions).
        expected_set = set(content_keys) | set(source_keys)
        extra_keys = [k for k in sorted(scores.keys()) if k not in expected_set]
        if extra_keys:
            invalid_eval_row_count += 1
            has_invalid_eval = True
            invalid_eval_details.append({
                "row_index": str(idx),
                "subject_id": subject_id_val,
                "qa_category": qa_category_val,
                "reason": "extra_keys_parsed",
                "extra_keys": ",".join(extra_keys),
                "expected_content_keys": ",".join(content_keys),
                "expected_source_keys": ",".join(source_keys),
                "parsed_keys": ",".join(sorted(scores.keys())),
                "eval_value": str(eval_val),
            })
            continue

        valid_row_count += 1

        # Normalize to a 0-1 range: both rubrics are 0-2, so a score of 1
        # contributes 0.5 (partial credit).
        num_correct_content = sum(
            scores[k] / CONTENT_MAX_SCORE for k in content_keys
        )
        num_qas_content = len(content_keys)
        num_correct_source = sum(
            scores[k] / SOURCE_MAX_SCORE for k in source_keys
        )
        num_qas_source = len(source_keys)

        # Paired: strict AND -- a pair counts only if BOTH the content answer
        # and its source follow-up earn full marks (Score 2). This mirrors
        # score_qa_answers_paired() on the multiple-choice path. Do NOT use a
        # partial-credit mean of the two: a mean is bounded by min/max of the
        # content and source accuracies, so it could never fall below source
        # accuracy and would not be comparable to the MCQ paired metric.
        scores_paired: Dict[str, float] = {}
        for base in content_keys:
            follow = f"{base}-1"
            if follow in qa_dict:
                content_correct = scores.get(base, 0) >= CONTENT_MAX_SCORE
                source_correct = scores.get(follow, 0) >= SOURCE_MAX_SCORE
                scores_paired[base] = 1.0 if (content_correct and source_correct) else 0.0
        num_qas_paired = len(scores_paired)
        num_correct_paired = sum(scores_paired.values())

        total_qa_count_content_individual += num_qas_content
        correct_qa_count_content_individual += num_correct_content
        sample_level_total_content_individual += 1
        # Perfect sample: every normalized score is 1.0.
        if num_qas_content > 0 and abs(num_correct_content - num_qas_content) < 1e-9:
            sample_level_correct_content_individual += 1
        if num_qas_content > 0:
            sample_ratio_sum_content_individual += num_correct_content / num_qas_content
            sample_ratio_count_content_individual += 1

        total_qa_count_source_individual += num_qas_source
        correct_qa_count_source_individual += num_correct_source
        sample_level_total_source_individual += 1
        if num_qas_source > 0 and abs(num_correct_source - num_qas_source) < 1e-9:
            sample_level_correct_source_individual += 1
        if num_qas_source > 0:
            sample_ratio_sum_source_individual += num_correct_source / num_qas_source
            sample_ratio_count_source_individual += 1

        total_qa_count_paired += num_qas_paired
        correct_qa_count_paired += num_correct_paired
        sample_level_total_paired += 1
        if num_qas_paired > 0 and abs(num_correct_paired - num_qas_paired) < 1e-9:
            sample_level_correct_paired += 1
        if num_qas_paired > 0:
            sample_ratio_sum_paired += num_correct_paired / num_qas_paired
            sample_ratio_count_paired += 1

    qa_acc_content_individual = (
        correct_qa_count_content_individual / total_qa_count_content_individual
        if total_qa_count_content_individual > 0 else 0.0
    )
    qa_acc_source_individual = (
        correct_qa_count_source_individual / total_qa_count_source_individual
        if total_qa_count_source_individual > 0 else 0.0
    )
    qa_acc_paired = (
        correct_qa_count_paired / total_qa_count_paired
        if total_qa_count_paired > 0 else 0.0
    )
    sample_acc_content_individual = (
        sample_level_correct_content_individual / sample_level_total_content_individual
        if sample_level_total_content_individual > 0 else 0.0
    )
    sample_acc_content_individual_mean = (
        sample_ratio_sum_content_individual / sample_ratio_count_content_individual
        if sample_ratio_count_content_individual > 0 else 0.0
    )
    sample_acc_source_individual = (
        sample_level_correct_source_individual / sample_level_total_source_individual
        if sample_level_total_source_individual > 0 else 0.0
    )
    sample_acc_source_individual_mean = (
        sample_ratio_sum_source_individual / sample_ratio_count_source_individual
        if sample_ratio_count_source_individual > 0 else 0.0
    )
    sample_acc_paired = (
        sample_level_correct_paired / sample_level_total_paired
        if sample_level_total_paired > 0 else 0.0
    )
    sample_acc_paired_mean = (
        sample_ratio_sum_paired / sample_ratio_count_paired
        if sample_ratio_count_paired > 0 else 0.0
    )

    print(f"  QA-level Acc (Content, Individual): {qa_acc_content_individual:.4f}")
    print(f"  QA-level Acc (Source, Individual):  {qa_acc_source_individual:.4f}")
    print(f"  QA-level Acc (Paired):              {qa_acc_paired:.4f}")
    print(f"  Sample-level Acc (Content, Ind.):   {sample_acc_content_individual:.4f}")
    print(f"  Sample-level Acc (Source, Ind.):    {sample_acc_source_individual:.4f}")
    print(f"  Sample-level Acc (Paired):          {sample_acc_paired:.4f}")
    print(f"  Sample-level Acc (Content, Ind., per-sample mean): {sample_acc_content_individual_mean:.4f}")
    print(f"  Sample-level Acc (Source, Ind., per-sample mean):  {sample_acc_source_individual_mean:.4f}")
    print(f"  Sample-level Acc (Paired, per-sample mean):        {sample_acc_paired_mean:.4f}")
    print(f"  Rows scored:                        {valid_row_count}")
    print(f"  Rows with invalid judge output:     {invalid_eval_row_count}")

    return {
        "qa_acc_content_individual": qa_acc_content_individual,
        "qa_acc_source_individual": qa_acc_source_individual,
        "qa_acc_paired": qa_acc_paired,
        "sample_acc_content_individual": sample_acc_content_individual,
        "sample_acc_source_individual": sample_acc_source_individual,
        "sample_acc_paired": sample_acc_paired,
        "sample_acc_content_individual_mean": sample_acc_content_individual_mean,
        "sample_acc_source_individual_mean": sample_acc_source_individual_mean,
        "sample_acc_paired_mean": sample_acc_paired_mean,
        # No letter parsing happens in generation mode.
        "unparseable_qa_count": 0,
        "unparseable_row_count": 0,
        "unparseable_details": [],
        # Generate-mode specific: used to exclude a model from the summary.
        "has_invalid_eval": has_invalid_eval,
        "invalid_eval_row_count": invalid_eval_row_count,
        "invalid_eval_details": invalid_eval_details,
        "valid_row_count": valid_row_count,
    }


def evaluate_single_csv(
    csv_path: Path,
    gold_by_subject: Dict[int, Dict[str, Dict[str, str]]],
) -> Dict[str, float]:
    """
    Evaluate QA and sample-level accuracies for a single model-answer CSV.

    *gold_by_subject* maps subject_id to that subject's turn dict, loaded from
    the released questions.csv.

    Scores standard MCQ runs (A-E options):
      - content individual (step_3)
      - source individual (step_4)
      - paired (both N and N-1 correct)
    """
    is_deepseek = "DeepSeek-R1-Distill-" in csv_path.name
    is_gemini = "gemini" in csv_path.name.lower()
    print(f"\nEvaluating {csv_path.name} (DeepSeek: {is_deepseek}, Gemini: {is_gemini})")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows")

    total_qa_count_content_individual = 0
    correct_qa_count_content_individual = 0
    sample_level_correct_content_individual = 0
    sample_level_total_content_individual = 0
    # For "soft" sample-level accuracy: mean of per-sample correct ratios
    sample_ratio_sum_content_individual = 0.0
    sample_ratio_count_content_individual = 0
    
    total_qa_count_source_individual = 0
    correct_qa_count_source_individual = 0
    sample_level_correct_source_individual = 0
    sample_level_total_source_individual = 0
    # For "soft" sample-level accuracy: mean of per-sample correct ratios
    sample_ratio_sum_source_individual = 0.0
    sample_ratio_count_source_individual = 0
    
    total_qa_count_paired = 0
    correct_qa_count_paired = 0
    sample_level_correct_paired = 0
    sample_level_total_paired = 0
    # For "soft" sample-level accuracy: mean of per-sample correct ratios
    sample_ratio_sum_paired = 0.0
    sample_ratio_count_paired = 0
    
    unparseable_qa_count = 0
    unparseable_row_count = 0  # Number of rows with at least one unparseable answer
    unparseable_details: List[Dict[str, str]] = []

    for idx, row in df.iterrows():
        model_answer_col = "model_answer"

        if "subject_id" not in df.columns or model_answer_col not in df.columns:
            continue

        if pd.isna(row.get("subject_id")):
            continue
        subject_id_int = int(row["subject_id"])

        model_answer_json = row[model_answer_col]
        if pd.isna(model_answer_json) or not str(model_answer_json).strip():
            continue

        try:
            # Gold answers for this subject come from the released questions.csv
            qa_dict = gold_by_subject.get(subject_id_int)
            if not qa_dict:
                print(
                    f"  Warning: subject {subject_id_int} is not in questions.csv; "
                    f"skipping row {idx}."
                )
                continue

            try:
                model_answers_dict = json.loads(str(model_answer_json))
                model_answers_dict = {str(k): v for k, v in model_answers_dict.items()}
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
            
            if not model_answers_dict:
                continue
            
            row_has_unparseable = False
            
            subject_id = str(subject_id_int)

            # Check all QAs in qa_dict for unparseable answers
            for key in qa_dict.keys():
                if key not in model_answers_dict:
                    # Missing key counts as unparseable
                    unparseable_qa_count += 1
                    row_has_unparseable = True
                    unparseable_details.append({
                        "subject_id": subject_id,
                        "question_key": str(key),
                        "model_answer": "(missing key)",
                    })
                    continue
                
                model_answer_text = model_answers_dict[key]
                if not model_answer_text or (isinstance(model_answer_text, str) and not model_answer_text.strip()):
                    unparseable_qa_count += 1
                    row_has_unparseable = True
                    unparseable_details.append({
                        "subject_id": subject_id,
                        "question_key": str(key),
                        "model_answer": "(empty value)",
                    })
                    continue
                
                model_answer_letter = extract_answer_letter(model_answer_text, is_deepseek, is_gemini)
                if not model_answer_letter or model_answer_letter not in ("A", "B", "C", "D", "E"):
                    unparseable_qa_count += 1
                    row_has_unparseable = True
                    unparseable_details.append({
                        "subject_id": subject_id,
                        "question_key": str(key),
                        "model_answer": str(model_answer_text),
                    })
            
            if row_has_unparseable:
                unparseable_row_count += 1
            
            scores_content_individual = score_qa_answers_individual(qa_dict, model_answers_dict, is_deepseek, is_gemini)
            
            scores_source_individual = score_qa_answers_source_individual(qa_dict, model_answers_dict, is_deepseek, is_gemini)
            
            scores_paired = score_qa_answers_paired(qa_dict, model_answers_dict, is_deepseek, is_gemini)
            
            num_qas_content_individual = len(scores_content_individual)
            num_correct_content_individual = sum(scores_content_individual.values())
            
            total_qa_count_content_individual += num_qas_content_individual
            correct_qa_count_content_individual += num_correct_content_individual
            
            # Sample-level for content individual: 1 if all step_3 QAs are correct, 0 otherwise
            sample_level_total_content_individual += 1
            
            if num_correct_content_individual == num_qas_content_individual and num_qas_content_individual > 0:
                sample_level_correct_content_individual += 1

            # "Soft" sample-level: per-sample ratio of correct step_3 QAs
            if num_qas_content_individual > 0:
                sample_ratio_sum_content_individual += (
                    num_correct_content_individual / num_qas_content_individual
                )
                sample_ratio_count_content_individual += 1
            
            num_qas_source_individual = len(scores_source_individual)
            num_correct_source_individual = sum(scores_source_individual.values())
            
            total_qa_count_source_individual += num_qas_source_individual
            correct_qa_count_source_individual += num_correct_source_individual
            
            # Sample-level for source individual: 1 if all step_4 QAs are correct, 0 otherwise
            sample_level_total_source_individual += 1
            
            if num_correct_source_individual == num_qas_source_individual and num_qas_source_individual > 0:
                sample_level_correct_source_individual += 1

            # "Soft" sample-level: per-sample ratio of correct step_4 QAs
            if num_qas_source_individual > 0:
                sample_ratio_sum_source_individual += (
                    num_correct_source_individual / num_qas_source_individual
                )
                sample_ratio_count_source_individual += 1
            
            num_qas_paired = len(scores_paired)
            num_correct_paired = sum(scores_paired.values())
            
            total_qa_count_paired += num_qas_paired
            correct_qa_count_paired += num_correct_paired
            
            # Sample-level for paired: 1 if all pairs are correct, 0 otherwise
            sample_level_total_paired += 1
            
            if num_correct_paired == num_qas_paired and num_qas_paired > 0:
                sample_level_correct_paired += 1
            
            # "Soft" sample-level: per-sample ratio of correct pairs
            if num_qas_paired > 0:
                sample_ratio_sum_paired += (
                    num_correct_paired / num_qas_paired
                )
                sample_ratio_count_paired += 1
            
        except Exception as e:
            print(f"  Error processing row {idx}: {e}")
            continue
    
    # Compute overall accuracies with safe division
    qa_acc_content_individual = (
        correct_qa_count_content_individual / total_qa_count_content_individual
        if total_qa_count_content_individual > 0 else 0.0
    )
    qa_acc_source_individual = (
        correct_qa_count_source_individual / total_qa_count_source_individual
        if total_qa_count_source_individual > 0 else 0.0
    )
    qa_acc_paired = (
        correct_qa_count_paired / total_qa_count_paired
        if total_qa_count_paired > 0 else 0.0
    )
    sample_acc_content_individual = (
        sample_level_correct_content_individual / sample_level_total_content_individual
        if sample_level_total_content_individual > 0 else 0.0
    )
    sample_acc_content_individual_mean = (
        sample_ratio_sum_content_individual / sample_ratio_count_content_individual
        if sample_ratio_count_content_individual > 0 else 0.0
    )
    sample_acc_source_individual = (
        sample_level_correct_source_individual / sample_level_total_source_individual
        if sample_level_total_source_individual > 0 else 0.0
    )
    sample_acc_source_individual_mean = (
        sample_ratio_sum_source_individual / sample_ratio_count_source_individual
        if sample_ratio_count_source_individual > 0 else 0.0
    )
    sample_acc_paired = (
        sample_level_correct_paired / sample_level_total_paired
        if sample_level_total_paired > 0 else 0.0
    )
    sample_acc_paired_mean = (
        sample_ratio_sum_paired / sample_ratio_count_paired
        if sample_ratio_count_paired > 0 else 0.0
    )
    
    print(f"  QA-level Acc (Content, Individual): {qa_acc_content_individual:.4f}")
    print(f"  QA-level Acc (Source, Individual):  {qa_acc_source_individual:.4f}")
    print(f"  QA-level Acc (Paired):              {qa_acc_paired:.4f}")
    print(f"  Sample-level Acc (Content, Ind.):   {sample_acc_content_individual:.4f}")
    print(f"  Sample-level Acc (Source, Ind.):    {sample_acc_source_individual:.4f}")
    print(f"  Sample-level Acc (Paired):          {sample_acc_paired:.4f}")
    print(f"  Sample-level Acc (Content, Ind., per-sample mean): {sample_acc_content_individual_mean:.4f}")
    print(f"  Sample-level Acc (Source, Ind., per-sample mean):  {sample_acc_source_individual_mean:.4f}")
    print(f"  Sample-level Acc (Paired, per-sample mean):        {sample_acc_paired_mean:.4f}")
    print(f"  Unparseable QAs:                     {unparseable_qa_count}")
    print(f"  Rows with unparseable QAs:          {unparseable_row_count}")
    
    return {
        "qa_acc_content_individual": qa_acc_content_individual,
        "qa_acc_source_individual": qa_acc_source_individual,
        "qa_acc_paired": qa_acc_paired,
        "sample_acc_content_individual": sample_acc_content_individual,
        "sample_acc_source_individual": sample_acc_source_individual,
        "sample_acc_paired": sample_acc_paired,
        "sample_acc_content_individual_mean": sample_acc_content_individual_mean,
        "sample_acc_source_individual_mean": sample_acc_source_individual_mean,
        "sample_acc_paired_mean": sample_acc_paired_mean,
        "unparseable_qa_count": unparseable_qa_count,
        "unparseable_row_count": unparseable_row_count,
        "unparseable_details": unparseable_details,
    }


def main():
    parser = argparse.ArgumentParser(description="Score multi-turn QA model answers")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help=(
            "Directory containing per-model answer CSVs to score. "
            "The directory must contain one CSV per model, each with a "
            "'subject_id' column and a 'model_answer' column."
        ),
    )
    parser.add_argument(
        "--questions-csv",
        type=str,
        default=None,
        help=(
            "questions.csv from the released dataset, supplying the correct "
            "answers. If not provided, defaults to <input-dir>/questions.csv, "
            "then <DATA_DIR>/questions.csv."
        ),
    )
    parser.add_argument(
        "--notes-csv",
        type=str,
        default=None,
        help=(
            "notes.csv from the released dataset. Scoring does not read note "
            "text, but the loader validates the two tables against each other. "
            "If not provided, defaults to the questions.csv sibling."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help=(
            "Output CSV file with per-model scoring summary. "
            "If not provided, defaults to "
            "<input-dir>/multiturn_qa_scoring_summary.csv."
        ),
    )
    parser.add_argument(
        "--csv-prefix",
        type=str,
        default=None,
        help=(
            "Optional filename prefix to strip when extracting the model "
            "name from each CSV file. E.g., if your CSVs are named "
            "'answers_{model_name}.csv', pass --csv-prefix answers_. "
            "If not provided, the model name is matched against the canonical "
            "list in core/model_ordering.py (longest-suffix match); when no "
            "match is found, the full filename stem is used as the model name."
        ),
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help=(
            "Score free-text generation runs instead of multiple-choice runs. "
            "Expects CSVs judged by "
            "evaluation/multiturn_gpt_evaluate_generation_eval.py (i.e. carrying "
            "a gpt_eval / gemini_eval column) -- point --input-dir at that "
            "script's output directory. Per-question judge scores (0-2) are "
            "normalised to the same accuracy fields the multiple-choice path "
            "reports."
        ),
    )
    parser.add_argument(
        "--eval-column",
        type=str,
        default=None,
        help=(
            "Judge column to read when using --generate. Default: auto-detected "
            "from the input directory name ('gemini_eval' if it contains "
            "'gemini', else 'gpt_eval'), falling back to whichever of "
            "gpt_eval / gemini_eval is present in the CSV."
        ),
    )
    args = parser.parse_args()

    # Determine input directory (resolved against the current working
    # directory if a relative path is provided, matching the eval scripts).
    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = input_dir.expanduser().resolve()

    if args.output_csv:
        output_path = Path(args.output_csv)
    else:
        output_path = input_dir / "multiturn_qa_scoring_summary.csv"

    # Auto-detect the judge column from the input directory name when scoring
    # free-text runs; evaluate_single_csv_generate() falls back to whichever of
    # gpt_eval / gemini_eval the CSV actually has.
    if args.generate and args.eval_column is None:
        args.eval_column = (
            "gemini_eval" if "gemini" in input_dir.name.lower() else "gpt_eval"
        )

    suffix = ".csv"

    if not input_dir.exists():
        print(f"Input directory does not exist: {input_dir}")
        return

    # Locate the released tables that hold the correct answers.
    if args.questions_csv:
        questions_csv = Path(args.questions_csv).expanduser().resolve()
    elif (input_dir / "questions.csv").exists():
        questions_csv = input_dir / "questions.csv"
    else:
        questions_csv = Path(DATA_DIR) / "questions.csv"
    notes_csv = (
        Path(args.notes_csv).expanduser().resolve()
        if args.notes_csv
        else questions_csv.parent / "notes.csv"
    )
    if not questions_csv.exists() or not notes_csv.exists():
        print(
            f"Released dataset not found (looked for {notes_csv} and "
            f"{questions_csv}).\nPass --questions-csv / --notes-csv explicitly."
        )
        return

    print(f"Loading gold answers from {questions_csv}")
    gold_by_subject = {
        s["subject_id"]: s["qas"]
        for s in load_release_samples(notes_csv, questions_csv)
    }
    print(f"  {len(gold_by_subject)} subjects")

    # Pick up any CSV in the directory except our own summary output and the
    # released tables themselves, if they happen to live alongside the answers.
    dataset_names = {"notes.csv", "questions.csv", "data_dictionary.csv"}
    csv_paths = sorted(
        p
        for p in input_dir.glob("*.csv")
        if p.name != output_path.name
        and p.name.endswith(suffix)
        and p.name not in dataset_names
    )
    if not csv_paths:
        print(f"No CSV files found under {input_dir}")
        return

    print(f"Found {len(csv_paths)} CSV files to evaluate under {input_dir}")

    summary_rows = []

    unparseable_dir = output_path.parent / "scoring_unparseable"
    unparseable_dir.mkdir(parents=True, exist_ok=True)

    # Generate mode only: models dropped from the summary because at least one
    # row's judge output could not be used, and CSVs that carry no judge column.
    excluded_generate_models: List[Dict[str, str]] = []
    skipped_missing_eval_columns: List[Dict[str, str]] = []

    def _extract_model_name(name: str) -> str:
        """Derive the model name from an answer-CSV filename.

        The evaluation scripts write ``answers_<model_name>.csv`` (plus a
        ``_generate`` suffix in free-text mode). Strategy:
          1. If ``--csv-prefix`` was provided and ``name`` starts with it,
             strip the prefix and the ``.csv`` suffix.
          2. Otherwise, drop a trailing ``_generate`` marker, then look for any
             model name from ``MODEL_ORDER`` that appears as a suffix. Longest
             match wins so e.g. ``DeepSeek-R1-Distill-Qwen-32B`` is preferred
             over a partial match.
          3. Fall back to the stem with a leading ``answers_`` stripped.
        """
        stem = name[:-len(suffix)] if name.endswith(suffix) else name
        if args.csv_prefix and stem.startswith(args.csv_prefix):
            return stem[len(args.csv_prefix):]
        if stem.endswith("_generate"):
            stem = stem[: -len("_generate")]
        matches = [m for m in MODEL_ORDER if stem.endswith(m)]
        if matches:
            return max(matches, key=len)
        return stem[len("answers_"):] if stem.startswith("answers_") else stem

    for csv_path in csv_paths:
        model_name = _extract_model_name(csv_path.name)

        if args.generate:
            try:
                metrics = evaluate_single_csv_generate(
                    csv_path, gold_by_subject, eval_column=args.eval_column
                )
            except ValueError as e:
                msg = str(e)
                if "column for generate-mode scoring" in msg:
                    skipped_missing_eval_columns.append(
                        {"model_name": model_name, "csv_path": str(csv_path), "reason": msg}
                    )
                    print(
                        f"  [generate] Skipping '{model_name}': no judge column "
                        f"(run evaluation/multiturn_gpt_evaluate_generation_eval.py first)."
                    )
                    continue
                raise
        else:
            metrics = evaluate_single_csv(csv_path, gold_by_subject)

        # Generate mode, strict exclusion: if ANY row's judge output was
        # unusable, the model is dropped from the summary entirely rather than
        # reporting a metric that silently mixes scored and unscored rows. The
        # offending rows are logged so they can be re-judged with --rows.
        if args.generate and metrics.get("has_invalid_eval"):
            invalid_rows = int(metrics.get("invalid_eval_row_count", 0) or 0)
            valid_rows_for_model = int(metrics.get("valid_row_count", 0) or 0)
            excluded_generate_models.append({
                "model_name": model_name,
                "invalid_eval_row_count": str(invalid_rows),
                "csv_path": str(csv_path),
            })

            invalid_eval_details = metrics.get("invalid_eval_details", []) or []
            if invalid_eval_details:
                invalid_file = unparseable_dir / f"{model_name}_generate_invalid_eval.txt"
                with open(invalid_file, "w", encoding="utf-8") as f:
                    f.write(f"Invalid / unparsable judge rows for model: {model_name}\n")
                    f.write(f"Source CSV: {csv_path}\n")
                    f.write(f"Total invalid rows: {invalid_rows}\n")
                    f.write("=" * 80 + "\n\n")
                    for i, detail in enumerate(invalid_eval_details, 1):
                        f.write(f"[{i}] row_index: {detail.get('row_index', '')}\n")
                        if detail.get("subject_id"):
                            f.write(f"    subject_id: {detail.get('subject_id', '')}\n")
                        if detail.get("qa_category"):
                            f.write(f"    qa_category: {detail.get('qa_category', '')}\n")
                        f.write(f"    reason: {detail.get('reason', '')}\n")
                        for field in (
                            "missing_content_keys",
                            "missing_source_keys",
                            "extra_keys",
                            "expected_content_keys",
                            "expected_source_keys",
                            "parsed_keys",
                        ):
                            if detail.get(field):
                                f.write(f"    {field}: {detail.get(field)}\n")
                        f.write(f"    eval_value: {detail.get('eval_value', '')}\n")
                        f.write("-" * 80 + "\n")
                print(f"  Saved {len(invalid_eval_details)} invalid judge rows to {invalid_file}")

            reason_counts: Dict[str, int] = {}
            for d in invalid_eval_details:
                r = str(d.get("reason", "unknown"))
                reason_counts[r] = reason_counts.get(r, 0) + 1
            reason_breakdown = ", ".join(f"{r}={n}" for r, n in sorted(reason_counts.items()))
            print(
                f"  [generate] Excluding '{model_name}' from the summary: "
                f"{invalid_rows} row(s) had invalid judge output ({reason_breakdown}); "
                f"{valid_rows_for_model} row(s) parsed cleanly."
            )
            if invalid_eval_details:
                row_indices = " ".join(
                    str(d.get("row_index", "")) for d in invalid_eval_details
                )
                print(
                    f"  Re-judge those rows with: "
                    f"python evaluation/multiturn_gpt_evaluate_generation_eval.py "
                    f"--input-csv <generate CSV> --rows {row_indices}"
                )
            continue

        unparseable_details = metrics.get("unparseable_details", [])
        if unparseable_details:
            unparseable_file = unparseable_dir / f"{model_name}.txt"
            with open(unparseable_file, "w", encoding="utf-8") as f:
                f.write(f"Unparseable answers for model: {model_name}\n")
                f.write(f"Total unparseable QAs: {metrics.get('unparseable_qa_count', 0)}\n")
                f.write(f"Rows with unparseable QAs: {metrics.get('unparseable_row_count', 0)}\n")
                f.write("=" * 80 + "\n\n")
                for i, detail in enumerate(unparseable_details, 1):
                    f.write(f"[{i}] subject_id: {detail['subject_id']}\n")
                    f.write(f"    question_key: {detail['question_key']}\n")
                    f.write(f"    model_answer: {detail['model_answer']}\n")
                    f.write("-" * 80 + "\n")
            print(f"  Saved {len(unparseable_details)} unparseable answers to {unparseable_file}")
        
        summary_rows.append({
            "Model Name": model_name,
            "QA-level Acc (Content, Individual)": round(metrics["qa_acc_content_individual"] * 100.0, 2),
            "QA-level Acc (Source, Individual)": round(metrics["qa_acc_source_individual"] * 100.0, 2),
            "Sample-level Acc (Content, Individual)": round(metrics["sample_acc_content_individual"] * 100.0, 2),
            "Sample-level Acc (Source, Individual)": round(metrics["sample_acc_source_individual"] * 100.0, 2),
            "Sample-level Acc (Content, Individual, Mean-per-sample)": round(metrics["sample_acc_content_individual_mean"] * 100.0, 2),
            "Sample-level Acc (Source, Individual, Mean-per-sample)": round(metrics["sample_acc_source_individual_mean"] * 100.0, 2),
            "QA-level Acc (Paired)": round(metrics["qa_acc_paired"] * 100.0, 2),
            "Sample-level Acc (Paired)": round(metrics["sample_acc_paired"] * 100.0, 2),
            "Sample-level Acc (Paired, Mean-per-sample)": round(metrics["sample_acc_paired_mean"] * 100.0, 2),
        })

    # Generate mode: report everything that did not make it into the summary,
    # so a silently short summary is never mistaken for full coverage.
    if args.generate and (excluded_generate_models or skipped_missing_eval_columns):
        print("\n[generate] Models not included in the summary:")
        for entry in excluded_generate_models:
            print(
                f"  - {entry['model_name']}: "
                f"{entry['invalid_eval_row_count']} row(s) with invalid judge output"
            )
        for entry in skipped_missing_eval_columns:
            print(f"  - {entry['model_name']}: no judge column in {entry['csv_path']}")

    summary_df = pd.DataFrame(summary_rows)

    # Bail out gracefully if no eligible CSVs produced metrics.
    if summary_df.empty:
        if args.generate and (excluded_generate_models or skipped_missing_eval_columns):
            print(
                "\nNo models to summarize — every CSV in the input directory was "
                "excluded (see the list above).\nExiting without writing summary CSVs."
            )
        else:
            print(
                "\nNo models to summarize — no eligible CSVs were found in the input "
                "directory.\nExiting without writing summary CSVs or plots."
            )
        return

    # Reorder models according to the canonical order defined in core/model_ordering.py.
    def _base_model_name(name: str) -> str:
        if not isinstance(name, str):
            return name
        return name

    if not summary_df.empty:
        summary_df["__base"] = summary_df["Model Name"].map(_base_model_name)
        summary_df["__order"] = summary_df["__base"].map(
            lambda n: MODEL_ORDER_MAP.get(n, len(MODEL_ORDER))
        )
        summary_df = summary_df.sort_values(
            by=["__order", "__base", "Model Name"], kind="mergesort"
        ).drop(columns=["__order", "__base"])

        # Rename metric columns to requested wording and enforce column order.
        col_rename = {
            "QA-level Acc (Content, Individual)": "QA-level Acc\n(content)",
            "QA-level Acc (Source, Individual)": "QA-level Acc\n(source)",
            "QA-level Acc (Paired)": "QA-level Acc\n(paired)",
            "Sample-level Acc (Content, Individual, Mean-per-sample)": "Sample-level Acc\n(content, \nmean-per-sample)",
            "Sample-level Acc (Content, Individual)": "Sample-level\n(content, 0/1)",
            "Sample-level Acc (Source, Individual, Mean-per-sample)": "Sample-level\n(source,\nmean-per-sample)",
            "Sample-level Acc (Source, Individual)": "Sample-level\n(source, 0/1)",
            "Sample-level Acc (Paired, Mean-per-sample)": "Sample-level\n(paired,\nmean-per-sample)",
            "Sample-level Acc (Paired)": "Sample-level\n(paired, 0/1)",
        }
        summary_df = summary_df.rename(columns=col_rename)

        metric_cols_order = [
            "QA-level Acc\n(content)",
            "QA-level Acc\n(source)",
            "QA-level Acc\n(paired)",
            "Sample-level Acc\n(content, \nmean-per-sample)",
            "Sample-level\n(content, 0/1)",
            "Sample-level\n(source,\nmean-per-sample)",
            "Sample-level\n(source, 0/1)",
            "Sample-level\n(paired,\nmean-per-sample)",
            "Sample-level\n(paired, 0/1)",
        ]
        summary_df = summary_df[["Model Name"] + metric_cols_order]
    
    # Pretty-print with % sign for accuracies, but keep raw numeric values in the CSV.
    display_df = summary_df.copy()
    display_df["Model Name"] = display_df["Model Name"].map(_base_model_name)
    for col in display_df.columns:
        if col != "Model Name":
            display_df[col] = display_df[col].map(
                lambda v: f"{v:.2f}%" if isinstance(v, (int, float)) else v
            )
    print("Per-model summary:")
    print(display_df)

    csv_df = summary_df.copy()
    csv_df["Model Name"] = csv_df["Model Name"].map(_base_model_name)
    csv_df.to_csv(output_path, index=False)
    print(f"\nSaved summary for {len(summary_rows)} models to {output_path}")


if __name__ == "__main__":
    main()
