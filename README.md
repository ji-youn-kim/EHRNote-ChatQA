# EHRNote-ChatQA

Official Code for [EHRNote-ChatQA: A Benchmark for Evidence-Grounded Multi-Turn Clinical Question Answering over Longitudinal Discharge Summaries](https://arxiv.org/abs/2606.15735).

The benchmark itself (967 multi-turn samples, 16,072 medical-expert-verified
QA pairs across 8 clinical categories) will be released separately on
PhysioNet under credentialed access, consistent with the MIMIC-IV data use
agreement; this repository contains only the code. The link will be added
here once the dataset is published.

---

## Repository layout

```
.
├── core/                          # shared utilities (no entry points)
│   ├── multiturn_gpt.py           #   project paths + helpers
│   ├── llm_client.py              #   Azure OpenAI / Vertex Gemini dispatcher
│   ├── release_data.py            #   reader for notes.csv + questions.csv
│   ├── multiturn_gpt_evaluate_common.py  # vLLM init, prompt building, chat formatting
│   ├── model_answer_parsing.py    #   parse A-E from model outputs
│   └── model_ordering.py          #   canonical model display order
├── evaluation/
│   ├── multiturn_gpt_evaluate.py      # open-weight models via vLLM
│   ├── multiturn_gpt_evaluate_api.py  # closed-source models via HTTP API
│   └── multiturn_gpt_evaluate_generation_eval.py  # LLM judge for --generate runs
├── scoring/
│   └── multiturn_qa_scoring.py    # per-model accuracy summary
├── data_generation/               # pipeline that produced the released tables
├── prompt/                        # evaluation, judge, and generation prompts
├── scripts/                       # example bash runners
├── requirements.txt
└── README.md
```

Subfolders are pure code/text only; no MIMIC data is shipped.

---

## Installation

Developed with Python 3.12:

```bash
pip install -r requirements.txt
```

Versions are pinned to the environment used in the paper (``pandas==2.3.3``,
``vllm==0.16.0``, ``transformers==5.2.0``, ``torch==2.9.1``,
``huggingface_hub==1.4.1``, ``openai==2.24.0``, ``google-genai==1.65.0``,
``google-cloud-aiplatform==1.143.0``, ``numpy==2.2.6``).

### API credentials

The API evaluation script and the generation judge call hosted LLMs through
``core/llm_client.py``, which reads every credential from environment
variables:

| Variable | Used when |
|---|---|
| `AZURE_API_KEY`, `AZURE_ENDPOINT`, `AZURE_API_VERSION` | GPT models on Azure OpenAI |
| `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` | Gemini, with Application Default Credentials (`gcloud auth application-default login`) |

Gemini is always called through Vertex AI, and GPT through Azure OpenAI —
the HIPAA-compliant deployments, as required for MIMIC-derived content.

---

## Dataset

The benchmark will be distributed as two tables. Every script here reads them
through ``core/release_data.py``; point ``--data-dir`` at the directory
holding both, or pass ``--notes-csv`` / ``--questions-csv``.

``notes.csv`` — one row per discharge summary:

| column | description |
|---|---|
| `subject_id` | MIMIC-IV subject id; foreign key to `questions.csv` |
| `note_num` | position within the subject's chronologically ordered notes (1-5). This is the N in `Note #N` in source-turn answer choices |
| `note_id`, `hadm_id`, `note_seq` | provenance back to MIMIC-IV-Note |
| `chartdate`, `storetime` | chart date and store time (date-shifted by MIMIC de-identification) |
| `text` | the discharge summary as the model sees it; its section headers are what source-turn answers refer to |

``questions.csv`` — one row per conversation turn:

| column | description |
|---|---|
| `subject_id` | foreign key to `notes.csv`; unique per conversation |
| `qa_category` | clinical domain of the conversation |
| `turn_order` | position in the conversation, from 1 |
| `turn_index` | index of the parent content question; a content turn and its source follow-up share it |
| `turn_type` | `content` or `source` |
| `question`, `option_a` … `option_e`, `correct_answer` | the turn itself |

Turn identity is stored as integers plus an enum rather than `"1-1"`-style
keys, because spreadsheet software converts those to dates.
``core/release_data.py`` rebuilds the string keys (`"1"`, `"1-1"`, `"2"`, …)
that appear in a `model_answer` JSON blob.

---

## Evaluation

For each sample, the first user message holds the patient's discharge
summaries plus the first question; later messages hold only the next
question. The model sees the conversation so far, with **its own prior
outputs** as the assistant turns, formatted by the tokenizer's native chat
template. Decoding is greedy, except that local runs use temperature 0.6 for
the DeepSeek-R1-Distill family per the official recommendation.

Past multiple-choice turns enter that history **without their A-E choices** —
the question alone, answered as `{letter}: {choice text}` — while the current
turn always shows all five. Replaying earlier turns in full would show later
turns every distractor written for earlier ones, which matters most for source
turns, whose options enumerate `Note #N ... Headers: ...` combinations.

Each run writes ``<output-dir>/answers_<model>.csv``, one row per subject,
with `subject_id`, `qa_category`, and `model_answer` — a JSON object mapping
turn key (`"1"`, `"1-1"`, …) to that turn's model output. Multiple-choice runs
also log answers that could not be matched to a letter A-E.

### Answer format

``--generate`` switches from multiple choice to free-text generation. The
multi-turn protocol is unchanged — same samples, same order, same history
from the model's own answers — only the turn presentation differs:

| | multiple choice (default) | free-text (``--generate``) |
|---|---|---|
| Content turn | question + A-E choices, answered with a letter | question only, answered in prose |
| Source turn | stored source question + A-E choices | "which sources support your previous answer?", answered as `Note #<n> Headers: <headers>` |
| Scoring | letter vs `correct_answer` | LLM judge, 0-2 per turn (see below) |

### Open-weight models (vLLM)

```bash
export CUDA_VISIBLE_DEVICES=0,1
python evaluation/multiturn_gpt_evaluate.py \
    --data-dir   data/release \
    --output-dir data/eval_local \
    --model      deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --tensor-parallel-size 2
```

Add ``--generate`` for free-text mode; the answer CSV then gets a
``_generate`` suffix.

Models evaluated in the paper: Llama-4-Scout-17B-16E-Instruct;
Qwen3-Next-80B-A3B-Instruct, Qwen3-30B-A3B-Instruct-2507,
Qwen3-4B-Instruct-2507; DeepSeek-R1-Distill-Llama-70B, ‑Llama-8B, ‑Qwen-32B,
‑Qwen-14B, ‑Qwen-7B; Ministral-3-14B/8B/3B-Instruct-2512; Phi-4-mini-instruct,
Phi-3.5-mini-instruct; medgemma-27b-it, medgemma-4b-it; MediPhi-Instruct,
MediPhi-Clinical, MediPhi-PubMed.

### Closed-source models (API)

The backend is auto-detected from ``--model``: names containing ``gemini`` go
to Gemini on Vertex AI, all others to Azure OpenAI.

```bash
python evaluation/multiturn_gpt_evaluate_api.py \
    --data-dir   data/release \
    --output-dir data/eval_api \
    --model      gpt-5.4        # or gpt-5.4-mini, gemini-3-flash-preview
```

``--generate`` works here too. Each subject is checkpointed to disk and a rerun skips subjects already
answered, so long runs can be interrupted and resumed without re-billing.
The paper evaluated ``gpt-5.4``, ``gpt-5.4-mini`` and
``gemini-3-flash-preview``, costing roughly \$120, \$40 and \$40 per model
over the 967 samples.

---

## Scoring

### Scoring multiple-choice runs

```bash
python scoring/multiturn_qa_scoring.py \
    --input-dir     data/eval_local \
    --questions-csv data/release/questions.csv
```

Every answer CSV in ``--input-dir`` is scored against the correct answers in
``questions.csv`` and aggregated into
``<input-dir>/multiturn_qa_scoring_summary.csv``, one row per model, ordered
by ``core/model_ordering.py``. Answers that could not be parsed are logged
under ``<input-dir>/scoring_unparseable/``.

Nine metrics per model: three correctness types (content turn, source turn,
paired = both correct) × three granularities:

- **QA-level** — averaged over all QA pairs in the dataset.
- **Sample-level (mean)** — per-sample correctness ratio, averaged over samples.
- **Sample-level (0/1)** — 1 only if every turn in the sample is correct.

### Scoring free-text runs

Free-text answers carry no letter, so a ``--generate`` run is judged first,
then scored:

```bash
# 1. judge: two calls per subject, 0-2 per turn
python evaluation/multiturn_gpt_evaluate_generation_eval.py \
    --input-csv data/eval_local_generate/answers_<model>_generate.csv \
    --data-dir  data/release \
    --model     gpt-5.4

# 2. score the judged CSVs
python scoring/multiturn_qa_scoring.py \
    --input-dir     data/eval_local_generate/gpt_evaluated \
    --questions-csv data/release/questions.csv \
    --generate
```

The **content pass** scores content turns against the question, with the
reference answer as a hint and the discharge summaries as the authority;
only a seriously dangerous hallucination forces a 0. The **source pass**
scores source turns against the model's *own* content answer rather than the
reference, and receives the verbatim content-pass output so it can force a 0
when the paired content answer scored 0. Both rubrics live in ``prompt/``.
The judge never modifies its input, checkpoints every row, and skips rows
already judged; ``--rows 4 7 12`` re-judges specific rows.

Judge scores map onto the same nine metrics: content and source turns get
partial credit (score / 2), while a pair counts only when **both** turns earn
full marks — the same strict AND as multiple choice, so the paired column is
comparable across formats.

If any row's judge output is unusable (empty, unparseable, or missing/extra
`Score` lines), that model is **excluded from the summary entirely** rather
than counting those turns as 0. The offending rows are logged and the script
prints the ``--rows`` command needed to re-judge them.

---

## Example end-to-end run

```bash
# 1. Install dependencies, export credentials for the providers you use
pip install -r requirements.txt

# 2. Obtain notes.csv + questions.csv from PhysioNet (once released) into data/release/

# 3. Evaluate open-weight models, API models, or both
bash scripts/run_evaluate_local.sh
bash scripts/run_evaluate_api.sh

# 4. Aggregate per-model accuracy into a summary table
bash scripts/run_scoring.sh

# 5. (Free-text runs only) judge the generations, then score them
bash scripts/run_generation_eval.sh
```

To make a new model appear in its preferred position in the summary tables,
add its name to ``MODEL_ORDER`` in ``core/model_ordering.py``. Unknown models
are still scored, and are listed after all known models.

---

## Citation

```bibtex
@misc{kim2026ehrnotechatqabenchmarkevidencegroundedmultiturn,
      title={EHRNote-ChatQA: A Benchmark for Evidence-Grounded Multi-Turn Clinical Question Answering over Longitudinal Discharge Summaries}, 
      author={Jiyoun Kim and Muhan Yeo and Eunhye Jang and Jeewon Yang and Hangyul Yoon and Su Ji Lee and Hee Jo Han and Hee-Jae Jung and Doyun Kwon and Jun young Lee and Jaehun Lee and Jung-Oh Lee and Sunjun Kweon and Jong Hak Moon and Daseul Kim and Minjae Cho and Edward Choi},
      year={2026},
      eprint={2606.15735},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2606.15735}, 
}
```

---

## Contact

For any inquiries, email jiyoun.kim@kaist.ac.kr.
