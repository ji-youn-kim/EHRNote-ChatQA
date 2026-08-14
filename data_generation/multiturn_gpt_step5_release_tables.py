#!/usr/bin/env python
# coding: utf-8

"""
Step 5 (offline): convert the wide step-4 output into the released two-table
dataset that evaluation and scoring consume.

Input
    ..._step4_post.csv
        One row per subject: wide note columns (note_1..note_5), the
        comma-delimited per-note metadata columns, and a `qas` JSON blob
        holding every turn.

Output (--output-dir)
    notes.csv             one row per discharge summary
    questions.csv         one row per conversation turn
    data_dictionary.csv   column-level documentation for both tables
    SHA256SUMS.txt        checksums for integrity verification

Design decisions baked in here:
  * `subject_id` is the primary key -- it is unique across all rows, so no
    synthetic conversation id is introduced.
  * Note text comes from the `note_*` columns (the preprocessed version the QA
    generation actually saw). The `Headers:` values in every source-turn answer
    are defined against this structure. `note_id` is emitted where the source
    metadata is available, so provenance back to MIMIC-IV-Note is recoverable.
  * The per-option `*_reasoning` fields are dropped: they are generation-time
    annotations, they state which option is correct, and nothing in evaluation
    reads them.
  * Turn identity is emitted as integers + an enum (`turn_order`, `turn_index`,
    `turn_type`) rather than "1-1"-style string keys, which spreadsheet
    software silently converts to dates. core/release_data.py rebuilds the
    string keys.

The note-provenance columns `note_seqs` and `storetimes` are optional. When the
input CSV does not carry them, `note_id`, `note_seq` and `storetime` are written
empty and a warning is printed; every other column, and everything evaluation
needs, is unaffected.

Usage:
    python data_generation/multiturn_gpt_step5_release_tables.py
    python data_generation/multiturn_gpt_step5_release_tables.py \\
        --input-csv data/sampled_df_grouped_step4_post.csv \\
        --output-dir data/release
"""

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

# Make sibling subfolders importable when this script is launched directly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _sub in ("core", "data_generation", "evaluation", "scoring", ""):
    _p = str(_PROJECT_ROOT / _sub) if _sub else str(_PROJECT_ROOT)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from multiturn_gpt import DATA_DIR

LETTERS = list("ABCDE")
MAX_NOTES = 5

# Comma-delimited per-note metadata columns. hadm_ids / charttimes are produced
# by the sampling step and required; note_seqs / storetimes carry MIMIC-IV-Note
# provenance and are optional.
DELIMITED_REQUIRED = ["hadm_ids", "charttimes"]
DELIMITED_OPTIONAL = ["note_seqs", "storetimes"]

NOTES_COLUMNS = [
    "subject_id", "note_num", "note_id", "hadm_id",
    "note_seq", "chartdate", "storetime", "text",
]
QUESTIONS_COLUMNS = [
    "subject_id", "qa_category", "turn_order", "turn_index", "turn_type",
    "question", "option_a", "option_b", "option_c", "option_d", "option_e",
    "correct_answer",
]

DATA_DICTIONARY = [
    ("notes.csv", "subject_id", "integer", "MIMIC-IV subject identifier. Foreign key to questions.csv."),
    ("notes.csv", "note_num", "integer", "Position of this note within the subject's chronologically ordered set (1-5). This is the N referenced as 'Note #N' in source-turn answer choices."),
    ("notes.csv", "note_id", "string", "MIMIC-IV-Note discharge summary identifier, formatted {subject_id}-DS-{note_seq}. Use to join back to the source note in MIMIC-IV-Note. Empty when the input CSV carries no note_seqs column."),
    ("notes.csv", "hadm_id", "integer", "MIMIC-IV hospital admission identifier for the admission this note documents."),
    ("notes.csv", "note_seq", "integer", "MIMIC-IV-Note sequence number for this subject's discharge summaries. Empty when the input CSV carries no note_seqs column."),
    ("notes.csv", "chartdate", "date (YYYY-MM-DD)", "Date the note was charted. Date-shifted by MIMIC-IV de-identification; only the year-to-year interval within a subject is meaningful."),
    ("notes.csv", "storetime", "timestamp (YYYY-MM-DD HH:MM:SS)", "Time the note was stored. Date component is shifted by MIMIC-IV; time of day is unshifted, per MIMIC-IV's de-identification scheme. Empty when the input CSV carries no storetimes column."),
    ("notes.csv", "text", "string", "Discharge summary text as presented to the QA generation pipeline: an 'Admission ID' / 'Chartdate' header prepended to the MIMIC-IV-Note body, with whitespace normalised. Section headers in this text are what source-turn answer choices refer to. De-identified placeholders ('___') are inherited from MIMIC-IV."),
    ("questions.csv", "subject_id", "integer", "MIMIC-IV subject identifier. Foreign key to notes.csv. Unique per conversation."),
    ("questions.csv", "qa_category", "string", "Clinical domain of the conversation. One of: clinical_assessment, clinical_outcome, diagnosis, disch_plan, medication, microbiology, procedure, symptom."),
    ("questions.csv", "turn_order", "integer", "Position of this turn in the conversation, starting at 1. Sort by this to reconstruct the dialogue in order."),
    ("questions.csv", "turn_index", "integer", "Index of the parent content question. A content turn and its follow-up source turn share the same turn_index."),
    ("questions.csv", "turn_type", "string", "'content' for a clinical question about the notes; 'source' for the follow-up asking which note sections support the previous answer. Every content turn is followed by exactly one source turn."),
    ("questions.csv", "question", "string", "Question text presented to the model."),
    ("questions.csv", "option_a", "string", "Answer choice A."),
    ("questions.csv", "option_b", "string", "Answer choice B."),
    ("questions.csv", "option_c", "string", "Answer choice C."),
    ("questions.csv", "option_d", "string", "Answer choice D."),
    ("questions.csv", "option_e", "string", "Answer choice E."),
    ("questions.csv", "correct_answer", "string", "Letter (A-E) of the correct answer."),
]


def turn_sort_key(key):
    """Order turn keys as 1, 1-1, 2, 2-1, ... (content before its source turn)."""
    s = str(key)
    if "-" in s:
        base, sub = s.split("-", 1)
        return (int(base), 1, int(sub))
    return (int(s), 0, 0)


def split_delimited(value, expected):
    parts = [p.strip() for p in str(value).split(",")]
    if len(parts) != expected:
        raise ValueError(f"expected {expected} values, got {len(parts)}: {value!r}")
    return parts


def build_notes(df):
    for col in DELIMITED_REQUIRED:
        if col not in df.columns:
            raise ValueError(f"input CSV is missing required column '{col}'")
    available = [c for c in DELIMITED_OPTIONAL if c in df.columns]
    missing = [c for c in DELIMITED_OPTIONAL if c not in df.columns]

    rows = []
    for _, r in df.iterrows():
        n = int(r["num_notes"])
        cols = {c: split_delimited(r[c], n) for c in DELIMITED_REQUIRED + available}
        for i in range(1, n + 1):
            text = r[f"note_{i}"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"subject {r['subject_id']}: note_{i} is empty but num_notes={n}")
            seq = cols["note_seqs"][i - 1] if "note_seqs" in cols else ""
            chartdate = cols["charttimes"][i - 1].split(" ")[0]
            rows.append({
                "subject_id": int(r["subject_id"]),
                "note_num": i,
                "note_id": f"{int(r['subject_id'])}-DS-{seq}" if seq else "",
                "hadm_id": int(cols["hadm_ids"][i - 1]),
                "note_seq": int(seq) if seq else "",
                "chartdate": chartdate,
                "storetime": cols["storetimes"][i - 1] if "storetimes" in cols else "",
                "text": text,
            })
        # any note column beyond num_notes must be empty
        for i in range(n + 1, MAX_NOTES + 1):
            extra = r.get(f"note_{i}")
            if isinstance(extra, str) and extra.strip():
                raise ValueError(f"subject {r['subject_id']}: note_{i} has text but num_notes={n}")

    if missing:
        print(
            f"  NOTE: input CSV has no {', '.join(missing)} column(s); "
            f"note_id / note_seq / storetime are written empty."
        )
    return pd.DataFrame(rows, columns=NOTES_COLUMNS)


def build_questions(df):
    rows = []
    for _, r in df.iterrows():
        qas = json.loads(r["qas"])
        for order, key in enumerate(sorted(qas.keys(), key=turn_sort_key), start=1):
            qa = qas[key]
            is_source = "-" in str(key)
            ca = (qa.get("correct_answer") or "").strip()
            if ca not in LETTERS:
                raise ValueError(f"subject {r['subject_id']} turn {key}: bad correct_answer {ca!r}")
            row = {
                "subject_id": int(r["subject_id"]),
                "qa_category": r["qa_category"],
                "turn_order": order,
                "turn_index": int(str(key).split("-")[0]),
                "turn_type": "source" if is_source else "content",
                "question": qa.get("Question") or "",
                "correct_answer": ca,
            }
            for L in LETTERS:
                val = qa.get(L)
                if not isinstance(val, str) or not val.strip():
                    raise ValueError(f"subject {r['subject_id']} turn {key}: option {L} empty")
                row[f"option_{L.lower()}"] = val
            rows.append(row)
    return pd.DataFrame(rows, columns=QUESTIONS_COLUMNS)


def validate(notes, questions, df):
    """Post-build checks. Returns (problems, duplicate-option findings)."""
    problems = []

    if not df["subject_id"].is_unique:
        problems.append("subject_id is not unique in the input")

    n_subj = df["subject_id"].nunique()
    if notes["subject_id"].nunique() != n_subj:
        problems.append("notes.csv is missing subjects")
    if questions["subject_id"].nunique() != n_subj:
        problems.append("questions.csv is missing subjects")

    # referential integrity both ways
    if set(notes["subject_id"]) != set(questions["subject_id"]):
        problems.append("subject_id sets differ between notes.csv and questions.csv")

    # note counts match num_notes
    expected = df.set_index("subject_id")["num_notes"].to_dict()
    actual = notes.groupby("subject_id").size().to_dict()
    bad = [s for s in expected if expected[s] != actual.get(s)]
    if bad:
        problems.append(f"{len(bad)} subjects have a note count != num_notes (e.g. {bad[:5]})")

    # note_num is a contiguous 1..n per subject
    for sid, grp in notes.groupby("subject_id"):
        if sorted(grp["note_num"]) != list(range(1, len(grp) + 1)):
            problems.append(f"subject {sid}: note_num not contiguous from 1")
            break

    # every content turn has exactly one source turn with the same turn_index
    for sid, grp in questions.groupby("subject_id"):
        c = set(grp[grp.turn_type == "content"]["turn_index"])
        s = set(grp[grp.turn_type == "source"]["turn_index"])
        if c != s:
            problems.append(f"subject {sid}: content/source turn_index mismatch {c ^ s}")
            break
        if sorted(grp["turn_order"]) != list(range(1, len(grp) + 1)):
            problems.append(f"subject {sid}: turn_order not contiguous from 1")
            break

    # source-turn answers must reference an existing note_num for that subject
    valid = notes.groupby("subject_id")["note_num"].apply(set).to_dict()
    dangling = 0
    for _, q in questions[questions.turn_type == "source"].iterrows():
        for L in LETTERS:
            for num in re.findall(r"Note #(\d+)", q[f"option_{L.lower()}"]):
                if int(num) not in valid[q["subject_id"]]:
                    dangling += 1
    if dangling:
        problems.append(f"{dangling} source-turn options reference a non-existent Note #N")

    # no reasoning leaked through
    leaked = [c for c in questions.columns if "reason" in c.lower()]
    if leaked:
        problems.append(f"reasoning columns present: {leaked}")

    # duplicate answer choices within a turn (data defect -- reported, not fatal)
    dups = []
    for _, q in questions.iterrows():
        seen = {}
        for L in LETTERS:
            t = " ".join(q[f"option_{L.lower()}"].split()).rstrip(".").lower()
            seen.setdefault(t, []).append(L)
        for letters in seen.values():
            if len(letters) > 1:
                dups.append((q["subject_id"], q["turn_order"], letters,
                             q["correct_answer"] in letters))
    return problems, dups


def write_csv(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL,
              lineterminator="\n", encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Convert the wide step-4 output into the released two-table dataset "
            "(notes.csv + questions.csv) that evaluation and scoring read."
        )
    )
    ap.add_argument(
        "--input-csv",
        type=Path,
        default=DATA_DIR / "sampled_df_grouped_step4_post.csv",
        help="Wide step-4 output CSV. Default: <DATA_DIR>/sampled_df_grouped_step4_post.csv",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR / "release",
        help="Directory to write notes.csv / questions.csv / data_dictionary.csv / "
             "SHA256SUMS.txt into. Default: <DATA_DIR>/release",
    )
    args = ap.parse_args()

    print(f"Reading {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    notes = build_notes(df)
    questions = build_questions(df)

    problems, dups = validate(notes, questions, df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    notes_path = args.output_dir / "notes.csv"
    questions_path = args.output_dir / "questions.csv"
    dict_path = args.output_dir / "data_dictionary.csv"

    write_csv(notes, notes_path)
    write_csv(questions, questions_path)
    write_csv(pd.DataFrame(DATA_DICTIONARY,
                           columns=["file", "column", "type", "description"]), dict_path)

    sums_path = args.output_dir / "SHA256SUMS.txt"
    with open(sums_path, "w", encoding="utf-8", newline="\n") as fh:
        for p in [notes_path, questions_path, dict_path]:
            fh.write(f"{sha256(p)}  {p.name}\n")

    print(f"\nWrote to {args.output_dir}")
    for p in [notes_path, questions_path, dict_path, sums_path]:
        print(f"  {p.name:22s} {p.stat().st_size/1e6:8.2f} MB")

    print(f"\nnotes.csv      {len(notes):6d} rows x {len(notes.columns)} cols "
          f"({notes.subject_id.nunique()} subjects)")
    print(f"questions.csv  {len(questions):6d} rows x {len(questions.columns)} cols "
          f"({(questions.turn_type=='content').sum()} content / "
          f"{(questions.turn_type=='source').sum()} source)")

    print("\n--- validation ---")
    if problems:
        for p in problems:
            print(f"  FAIL  {p}")
    else:
        print("  all structural checks passed")

    if dups:
        broken = [d for d in dups if d[3]]
        print(f"\n  WARNING: {len(dups)} turns contain duplicate answer choices; "
              f"{len(broken)} of these duplicate the CORRECT answer and are unscorable.")
        for sid, order, letters, is_ca in sorted(dups)[:40]:
            mark = "  <-- duplicates correct answer" if is_ca else ""
            print(f"    subject {sid} turn_order {order:<3} {letters}{mark}")
        print("  These are data defects in the input, not introduced by this conversion.")

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
