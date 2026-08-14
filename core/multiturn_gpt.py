import pandas as pd
from pathlib import Path


# ``core/multiturn_gpt.py`` lives one level below the project root.
# BASE_DIR points to the project root so DATA_DIR and PROMPT_DIR
# resolve to ``<project>/data`` and ``<project>/prompt``.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROMPT_DIR = BASE_DIR / "prompt"


def load_text(path: Path) -> str:
    """Load text content from a file."""
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def build_discharge_summaries_generate(row: pd.Series) -> str:
    """
    Build the discharge summaries block from the concatenated notes
    for a single patient (row in concat_df_grouped.csv or concat_df_grouped_step1.csv).
    """
    num_notes = int(row["num_notes"])
    charttimes = str(row["charttimes"]).split(",")
    hadm_ids = str(row["hadm_ids"]).split(",")

    note = ""
    for idx in range(num_notes):
        note_text = row[f"note_{idx+1}"]
        note = (
            note
            + f"[note #{idx+1} Admission ID: {hadm_ids[idx]}, Chartdate: {charttimes[idx][:-9]}]\n"
            + str(note_text)
            + f"\n[note #{idx+1} end]"
        )
        if idx < num_notes - 1:
            note = note + "\n\n"
    return note

def build_discharge_summaries_evaluate(row: pd.Series) -> str:
    """
    Build the discharge summaries block from the wide note columns of a single
    patient row (``num_notes`` + ``note_1``..``note_N``).

    Evaluation reads the released ``notes.csv`` instead of a wide CSV, so it
    calls ``release_data.build_discharge_summaries()`` directly; this wrapper
    delegates to the same function so the two paths can never diverge.
    """
    from release_data import build_discharge_summaries

    num_notes = int(row["num_notes"])
    return build_discharge_summaries(
        [str(row[f"note_{idx + 1}"]) for idx in range(num_notes)]
    )


# QA template categories used in both step1 and step2
QA_TEMPLATE_CATEGORIES = [
    "diagnosis",
    "symptom",
    "procedure",
    "medication",
    "microbiology",
    "clinical_assessment",
    "clinical_outcome",
    "disch_plan",
]
