#!/usr/bin/env python
# coding: utf-8

"""
Score free-text generation outputs with an LLM judge.

Reads a model-answer CSV produced by ``--generate`` (columns ``subject_id`` and
``model_answer``) together with the released ``notes.csv`` / ``questions.csv``,
and scores every turn against a 0-2 rubric: 0-2 for content questions and 0-2
for source-citation questions.

Scoring is TWO-PASS (two judge calls per row):

  Pass 1 (content) -- prompt/multiturn_gpt_evaluate_generate_eval_content.txt
    Scores the content keys (1, 2, 3, ...). The reference correct answer is
    supplied as a hint, but the question defines what is essential and the
    discharge summaries are the authoritative source.

  Pass 2 (source) -- prompt/multiturn_gpt_evaluate_generate_eval_source.txt
    Scores the source-citation keys (1-1, 2-1, ...) relative to the MODEL'S
    OWN content answer rather than the reference answer. The verbatim Pass 1
    output is injected as the ``[Content Pass Output]`` section, and the judge
    is instructed to force a source score of 0 whenever the paired content
    question scored 0. There is no Python-side parsing or auto-zero shortcut,
    so the forcing rule is applied by the judge that can read the reasoning.

The per-row eval text written to the output column is the concatenation of the
Pass 1 and Pass 2 raw responses. ``scoring/multiturn_qa_scoring.py --generate``
parses the ``Score{key}: {0-2}`` lines out of it.

The backend is auto-detected from --model by ``core/llm_client.py``: names
containing "gemini" go to Gemini on Vertex AI, all others to Azure OpenAI.
Credentials are read from environment variables (see README.md).
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# Make sibling subfolders importable when this script is launched directly.
import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _sub in ("core", "data_generation", "evaluation", "scoring", ""):
    _p = str(_PROJECT_ROOT / _sub) if _sub else str(_PROJECT_ROOT)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from multiturn_gpt import DATA_DIR, PROMPT_DIR, load_text
from release_data import load_release_samples, sort_turn_keys
from llm_client import chat_completion


# Single, rubric-agnostic system message shared by every judge call. The actual
# scoring criteria (scale, hallucination policy, source rules, output format)
# live solely in the prompt templates under prompt/, so there is only one source
# of truth to keep in sync.
EVAL_SYSTEM_MESSAGE = (
    "You are a careful clinical NLP evaluator. Follow the scoring instructions in "
    "the user message exactly, and always provide your reasoning before each score."
)


_sort_qa_keys = sort_turn_keys


def _get_reference_text(qa: Dict) -> str:
    """Resolve the reference correct answer text from a qa entry's correct_answer letter."""
    correct_letter = qa.get("correct_answer", "").upper()
    if correct_letter in ("A", "B", "C", "D", "E"):
        return qa.get(correct_letter, "")
    return ""


def build_content_qas_block(
    qa_dict: Dict[str, Dict],
    model_answers: Dict[str, str],
) -> str:
    """
    Build the qas block for the content-only scoring pass. Includes only content
    keys (no "-"). Each entry shows the question, the reference correct answer,
    and the model output.
    """
    lines: List[str] = []
    content_keys = [k for k in _sort_qa_keys(list(qa_dict.keys())) if "-" not in k]
    for key in content_keys:
        qa = qa_dict[key]
        question_text = qa.get("Question", "")
        model_text = model_answers.get(key, "")
        lines.append(f"Q{key}: {question_text}")
        lines.append(f"Reference Correct Answer {key}: {_get_reference_text(qa)}")
        lines.append(f"Model Output {key}: {model_text}")
        lines.append("")
    return "\n".join(lines).strip()


def build_source_qas_block(
    qa_dict: Dict[str, Dict],
    model_answers: Dict[str, str],
    source_keys: List[str],
) -> str:
    """
    Build the qas block for the source-only scoring pass. For each source key
    (e.g., "1-1"), include:
      - The source question text
      - The reference correct content answer for the paired content question (hint)
      - The reference correct source answer (hint)
      - The MODEL's content answer for the paired content question (anchor)
      - The MODEL's source output

    The judge looks up the paired content question's score directly in the
    [Content Pass Output] section of the source prompt; no per-key score is
    injected here.
    """
    lines: List[str] = []
    for key in _sort_qa_keys(source_keys):
        content_key = key.split("-")[0]
        source_qa = qa_dict.get(key, {})
        content_qa = qa_dict.get(content_key, {})

        source_question = source_qa.get("Question", "")
        ref_content_answer = _get_reference_text(content_qa)
        ref_source_answer = _get_reference_text(source_qa)
        model_content = model_answers.get(content_key, "")
        model_source = model_answers.get(key, "")

        lines.append(f"Q{key}: {source_question}")
        lines.append(f"Reference Correct Content Answer for Q{content_key}: {ref_content_answer}")
        lines.append(f"Reference Correct Source Answer for Q{key}: {ref_source_answer}")
        lines.append(f"Model Content Answer for Q{content_key}: {model_content}")
        lines.append(f"Model Source Output for Q{key}: {model_source}")
        lines.append("")
    return "\n".join(lines).strip()


def split_qa_keys(keys: List[str]) -> Tuple[List[str], List[str]]:
    """Split keys into (content_keys, source_keys)."""
    content_keys = [k for k in keys if "-" not in k]
    source_keys = [k for k in keys if "-" in k]
    return content_keys, source_keys


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Score free-text generation outputs with an LLM judge (two-pass "
            "content + source rubric, 0-2 per turn)."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help=(
            "Model-answer CSV produced by an evaluation run with --generate, "
            "containing 'subject_id' and 'model_answer' columns. A relative "
            "path is resolved against --data-dir."
        ),
    )
    parser.add_argument(
        "--notes-csv",
        type=str,
        default=None,
        help=(
            "notes.csv from the released dataset. If not provided, defaults to "
            "<data-dir>/notes.csv."
        ),
    )
    parser.add_argument(
        "--questions-csv",
        type=str,
        default=None,
        help=(
            "questions.csv from the released dataset (supplies the reference "
            "answers the judge sees as hints). If not provided, defaults to "
            "<data-dir>/questions.csv."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=(
            "Directory holding notes.csv / questions.csv, also used to resolve "
            "a relative --input-csv. If not provided, defaults to DATA_DIR "
            "from multiturn_gpt."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help=(
            "Output CSV path. If not provided, writes to "
            "<input-dir>/<provider>_evaluated/<input-name>.csv, where provider "
            "is 'gpt' or 'gemini' depending on --model. The input CSV is never "
            "modified."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.4",
        help=(
            "Judge model. Names containing 'gemini' are routed to Gemini on "
            "Vertex AI; all others to Azure OpenAI. The paper used gpt-5.4."
        ),
    )
    parser.add_argument(
        "--eval-column",
        type=str,
        default=None,
        help=(
            "Column to store the judge output in. Default: auto-detected from "
            "--model ('gemini_eval' for Gemini judges, 'gpt_eval' otherwise). "
            "Must match --eval-column in scoring/multiturn_qa_scoring.py."
        ),
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0,
        help="Seconds to sleep after each judge call (rate limiting). Default: 0.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Row indices to re-judge (e.g., --rows 4 7 12). Clears the eval "
            "column for those rows so the judge is called again. When --rows is "
            "given and the output CSV already exists, it is loaded instead of "
            "the input CSV so previously-saved scores for the other rows are "
            "preserved. scoring/multiturn_qa_scoring.py --generate prints the "
            "exact row indices to pass here for any row it could not parse."
        ),
    )
    args = parser.parse_args()

    if args.eval_column is None:
        args.eval_column = "gemini_eval" if "gemini" in args.model.lower() else "gpt_eval"

    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    input_path = Path(args.input_csv)
    if not input_path.is_absolute():
        input_path = data_dir / args.input_csv
    input_path = input_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    notes_csv = args.notes_csv or str(data_dir / "notes.csv")
    questions_csv = args.questions_csv or str(data_dir / "questions.csv")

    if args.output_csv:
        output_path = Path(args.output_csv).expanduser().resolve()
    else:
        provider = "gemini" if "gemini" in args.model.lower() else "gpt"
        output_dir = input_path.parent / f"{provider}_evaluated"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / input_path.name

    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    content_template = load_text(
        PROMPT_DIR / "multiturn_gpt_evaluate_generate_eval_content.txt"
    )
    source_template = load_text(
        PROMPT_DIR / "multiturn_gpt_evaluate_generate_eval_source.txt"
    )

    # When --rows is specified and the output CSV (which holds prior eval data)
    # already exists, load from the output path so previously-saved scores for
    # the OTHER rows are preserved. Otherwise load from the raw input.
    if args.rows is not None and output_path.exists():
        source_path = output_path
        print(f"--rows specified and {output_path} exists; loading from output for re-evaluation.")
    else:
        source_path = input_path

    print(f"Loading {source_path}...")
    df = pd.read_csv(source_path)
    total = len(df)
    if "subject_id" not in df.columns or "model_answer" not in df.columns:
        raise ValueError(
            "Input CSV must contain 'subject_id' and 'model_answer' columns "
            "(the output of an evaluation run with --generate)."
        )

    print(f"Loading {notes_csv} + {questions_csv}...")
    samples_by_subject = {
        s["subject_id"]: s for s in load_release_samples(notes_csv, questions_csv)
    }

    print("[Judge config]")
    print(f"  judge model : {args.model}")
    print(f"  backend     : "
          f"{'Gemini (Vertex)' if 'gemini' in args.model.lower() else 'Azure OpenAI'}")
    print(f"  eval column : {args.eval_column}")
    print(f"  output      : {output_path}")

    eval_col = args.eval_column
    if eval_col not in df.columns:
        df[eval_col] = ""

    # If --rows is specified, clear eval_col for those rows so they are re-run
    # by the judge (the skip-if-present check below would otherwise skip them).
    if args.rows is not None:
        for r in args.rows:
            if r < 0 or r >= len(df):
                print(f"Warning: row index {r} is out of range (0-{len(df)-1}), skipping...")
                continue
            df.at[r, eval_col] = ""
            sid = df.at[r, "subject_id"] if "subject_id" in df.columns else "?"
            cat = df.at[r, "qa_category"] if "qa_category" in df.columns else "?"
            print(
                f"Cleared {eval_col} for row {r} "
                f"(subject_id={sid}, qa_category={cat}) -- will re-run the judge"
            )

    processed = 0
    for idx, row in df.iterrows():
        # Resume: skip rows that already carry a judgement so long runs can be
        # restarted without re-billing completed rows.
        existing = row.get(eval_col, "")
        if isinstance(existing, str) and existing.strip():
            print(f"Skipping row {idx} (already has {eval_col})")
            continue
        if not pd.isna(existing) and not isinstance(existing, str):
            continue

        model_answer_json = row["model_answer"]
        if pd.isna(model_answer_json) or not str(model_answer_json).strip():
            continue

        subject_id = int(row["subject_id"])
        sample = samples_by_subject.get(subject_id)
        if sample is None:
            print(
                f"Warning: subject {subject_id} (row {idx}) is not in "
                f"{Path(questions_csv).name}; skipping."
            )
            continue
        qa_dict = sample["qas"]
        if not qa_dict:
            continue
        try:
            model_answers = json.loads(str(model_answer_json))
            model_answers = {str(k): (v or "") for k, v in model_answers.items()}
        except (json.JSONDecodeError, TypeError):
            continue

        discharge_summaries = sample["discharge_summaries"]

        # Two-pass scoring: content first, then source. The entire content
        # response is injected into the source prompt as [Content Pass Output];
        # nothing about it is parsed here.
        _, source_keys = split_qa_keys(list(qa_dict.keys()))

        # ---------- Pass 1: content ----------
        content_block = build_content_qas_block(qa_dict, model_answers)
        content_prompt = content_template.replace("{discharge summaries}", discharge_summaries)
        content_prompt = content_prompt.replace("{qas}", content_block)

        content_response = chat_completion(
            messages=[
                {"role": "system", "content": EVAL_SYSTEM_MESSAGE},
                {"role": "user", "content": content_prompt},
            ],
            model=args.model,
            temperature=0,
        ).strip()

        # ---------- Pass 2: source (all source keys, always) ----------
        source_response = ""
        if source_keys:
            source_block = build_source_qas_block(qa_dict, model_answers, source_keys)
            source_prompt = source_template.replace("{discharge summaries}", discharge_summaries)
            source_prompt = source_prompt.replace("{content_pass_output}", content_response)
            source_prompt = source_prompt.replace("{qas}", source_block)

            source_response = chat_completion(
                messages=[
                    {"role": "system", "content": EVAL_SYSTEM_MESSAGE},
                    {"role": "user", "content": source_prompt},
                ],
                model=args.model,
                temperature=0,
            ).strip()

        # ---------- Merge content + source into a single eval text ----------
        parts: List[str] = [content_response]
        if source_response:
            parts.append(source_response)
        final_eval = "\n\n".join(p for p in parts if p)

        df.at[idx, eval_col] = final_eval
        processed += 1
        print(f"Row {processed} / {total} (idx={idx})")
        if args.sleep > 0:
            time.sleep(args.sleep)
        # Per-row checkpoint so judge runs are resumable.
        df.to_csv(output_path, index=False)

    df.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
