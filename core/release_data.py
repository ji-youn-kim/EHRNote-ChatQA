# coding: utf-8
"""
Reader for the released two-table dataset (``notes.csv`` + ``questions.csv``).

The benchmark is distributed as two tidy tables rather than one wide CSV:

  notes.csv      one row per discharge summary
                 (subject_id, note_num, note_id, hadm_id, note_seq,
                  chartdate, storetime, text)
  questions.csv  one row per conversation turn
                 (subject_id, qa_category, turn_order, turn_index, turn_type,
                  question, option_a..option_e, correct_answer)

Every consumer in this repository -- evaluation, the generation judge, and
scoring -- reads the dataset through :func:`load_release_samples`, so the two
files are the single source of truth for what a model is shown.

``load_release_samples`` reassembles, per subject:

  * ``discharge_summaries`` -- the notes joined in ``note_num`` order with the
    ``[note #N start] ... [note #N end]`` wrapper the prompts expect.
  * ``qas`` -- the per-turn dict keyed ``"1"``, ``"1-1"``, ``"2"``, ``"2-1"``,
    ... that the evaluation loops iterate over. The release tables encode turn
    identity as integers (``turn_index`` + ``turn_type``) because spreadsheet
    software silently converts ``"1-1"``-style keys to dates; the string keys
    are rebuilt here and are what appears in a model_answer JSON blob.
"""

from pathlib import Path
from typing import Dict, List, Sequence, Union

import pandas as pd

LETTERS = ["A", "B", "C", "D", "E"]

NOTES_REQUIRED_COLUMNS = ["subject_id", "note_num", "text"]
QUESTIONS_REQUIRED_COLUMNS = [
    "subject_id", "qa_category", "turn_order", "turn_index", "turn_type",
    "question", "option_a", "option_b", "option_c", "option_d", "option_e",
    "correct_answer",
]


def build_discharge_summaries(note_texts: Sequence[str]) -> str:
    """Join per-note text into the ``[Discharge Summaries]`` block body.

    *note_texts* must already be in ``note_num`` order (note 1 first). This is
    the single definition of that format: the ``Note #N`` references in every
    source-turn answer choice are resolved against these numbers.
    """
    parts = []
    for idx, text in enumerate(note_texts, start=1):
        parts.append(f"[note #{idx} start]\n{text}\n[note #{idx} end]")
    return "\n\n".join(parts)


def _turn_key(turn_index: int, turn_type: str, source_ordinal: int) -> str:
    """Rebuild the string turn key: "2" for content, "2-1" for its source turn."""
    if turn_type == "content":
        return str(turn_index)
    return f"{turn_index}-{source_ordinal}"


def _require_columns(df: pd.DataFrame, required: Sequence[str], path: Path) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required column(s): {', '.join(missing)}")


def load_release_samples(
    notes_csv: Union[str, Path],
    questions_csv: Union[str, Path],
) -> List[Dict]:
    """Load the release tables and return one record per subject.

    Each record is a dict with:
        subject_id           int
        qa_category          str
        num_notes            int
        discharge_summaries  str   -- ready to substitute into a prompt
        qas                  dict  -- {turn_key: {Question, A..E, correct_answer}}

    Subjects are returned in the order they first appear in ``questions.csv``,
    and each subject's turns follow ``turn_order``.
    """
    notes_path = Path(notes_csv)
    questions_path = Path(questions_csv)
    notes_df = pd.read_csv(notes_path)
    questions_df = pd.read_csv(questions_path)
    _require_columns(notes_df, NOTES_REQUIRED_COLUMNS, notes_path)
    _require_columns(questions_df, QUESTIONS_REQUIRED_COLUMNS, questions_path)

    notes_by_subject: Dict[int, List[str]] = {}
    for subject_id, group in notes_df.groupby("subject_id", sort=False):
        group = group.sort_values("note_num")
        note_nums = [int(n) for n in group["note_num"]]
        if note_nums != list(range(1, len(note_nums) + 1)):
            raise ValueError(
                f"{notes_path}: subject {subject_id} has non-contiguous note_num "
                f"{note_nums}; 'Note #N' references would be ambiguous."
            )
        notes_by_subject[int(subject_id)] = [str(t) for t in group["text"]]

    samples: List[Dict] = []
    for subject_id, group in questions_df.groupby("subject_id", sort=False):
        subject_id = int(subject_id)
        if subject_id not in notes_by_subject:
            raise ValueError(
                f"{questions_path}: subject {subject_id} has questions but no rows "
                f"in {notes_path.name}."
            )
        group = group.sort_values("turn_order")

        qas: Dict[str, Dict[str, str]] = {}
        source_counts: Dict[int, int] = {}
        for _, q in group.iterrows():
            turn_index = int(q["turn_index"])
            turn_type = str(q["turn_type"]).strip()
            if turn_type not in ("content", "source"):
                raise ValueError(
                    f"{questions_path}: subject {subject_id} turn_order "
                    f"{q['turn_order']} has unknown turn_type {turn_type!r}."
                )
            if turn_type == "source":
                source_counts[turn_index] = source_counts.get(turn_index, 0) + 1
                ordinal = source_counts[turn_index]
            else:
                ordinal = 0
            key = _turn_key(turn_index, turn_type, ordinal)

            entry = {"Question": str(q["question"])}
            for letter in LETTERS:
                entry[letter] = str(q[f"option_{letter.lower()}"])
            correct = str(q["correct_answer"]).strip().upper()
            if correct not in LETTERS:
                raise ValueError(
                    f"{questions_path}: subject {subject_id} turn {key} has "
                    f"correct_answer {q['correct_answer']!r}, expected one of A-E."
                )
            entry["correct_answer"] = correct
            qas[key] = entry

        note_texts = notes_by_subject[subject_id]
        samples.append({
            "subject_id": subject_id,
            "qa_category": str(group["qa_category"].iloc[0]),
            "num_notes": len(note_texts),
            "discharge_summaries": build_discharge_summaries(note_texts),
            "qas": qas,
        })

    return samples


def sort_turn_keys(keys: Sequence[str]) -> List[str]:
    """Sort turn keys in conversation order: 1, 1-1, 2, 2-1, ..."""
    return sorted(
        keys,
        key=lambda x: (
            int(str(x).split("-")[0]),
            1 if "-" in str(x) else 0,
            int(str(x).split("-")[1]) if "-" in str(x) else 0,
        ),
    )
