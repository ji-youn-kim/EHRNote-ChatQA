# coding: utf-8
"""
Shared logic for vLLM-based evaluation scripts: prompt building,
chat formatting, model loading, and sampling params.

HuggingFace cache location is controlled by the user's environment
(``$HF_HOME`` / ``$HF_HUB_CACHE``) or the ``--hf-cache-dir`` CLI flag;
this module does not set or override those.
"""

import argparse
import os
import re
from typing import Dict, List, Optional

from vllm import LLM, SamplingParams

from multiturn_gpt import PROMPT_DIR


def load_prompt_template(template_name: str = "multiturn_gpt_evaluate.txt") -> str:
    """Load the evaluation prompt template from PROMPT_DIR."""
    prompt_path = PROMPT_DIR / template_name
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Follow-up turn templates (no discharge summaries; used for turns 2+).
#
# A follow-up turn uses the SAME prompt file as the first turn, minus the
# ``[Discharge Summaries]`` block (the notes are already in the chat history).
# The files under prompt/ are the single source of truth: never duplicate their
# wording here, or the first turn and later turns silently drift apart.
# ---------------------------------------------------------------------------

_DISCHARGE_SUMMARIES_BLOCK_RE = re.compile(
    r"\A\[Discharge Summaries\][ \t]*\r?\n\{discharge summaries\}[ \t]*\r?\n\s*"
)


def strip_discharge_summaries_block(template: str) -> str:
    """Return *template* without its leading ``[Discharge Summaries]`` block."""
    return _DISCHARGE_SUMMARIES_BLOCK_RE.sub("", template).strip()


def load_followup_template(template_name: str) -> str:
    """Load a prompt file and strip its ``[Discharge Summaries]`` block.

    Used to derive the turns-2+ variant of a prompt from the same file the
    first turn uses, so the two can never diverge.
    """
    return strip_discharge_summaries_block(load_prompt_template(template_name))


FOLLOWUP_MCQ_TEMPLATE = load_followup_template("multiturn_gpt_evaluate.txt")

FOLLOWUP_GENERATE_CONTENT_TEMPLATE = load_followup_template(
    "multiturn_gpt_evaluate_generate_content.txt"
)

FOLLOWUP_GENERATE_SOURCE_TEMPLATE = load_followup_template(
    "multiturn_gpt_evaluate_generate_source.txt"
)

# Question text used for the source-location turns in generation mode. The
# stored source question is written for multiple choice ("which of the
# following..."), so free-text generation asks for the sources of the model's
# own previous answer instead.
GENERATE_SOURCE_QUESTION = (
    "What exact sources from the patient's discharge summaries "
    "contain information for your previous answer? Include all "
    "necessary sources, but only the minimal set required to "
    "fully support the answer."
)


def build_mcq_user_content(
    question: str,
    choice_a: str,
    choice_b: str,
    choice_c: str,
    choice_d: str,
    choice_e: str,
    discharge_summaries: str = "",
    prompt_template: Optional[str] = None,
) -> str:
    """Build user message content for an MCQ turn.

    If *discharge_summaries* is provided (first turn), uses *prompt_template*
    (the full template with ``[Discharge Summaries]``).  Otherwise uses the
    compact ``FOLLOWUP_MCQ_TEMPLATE``.
    """
    if prompt_template:
        tmpl = prompt_template
        if discharge_summaries:
            tmpl = tmpl.replace("{discharge summaries}", discharge_summaries)
        else:
            tmpl = strip_discharge_summaries_block(tmpl)
    else:
        tmpl = FOLLOWUP_MCQ_TEMPLATE
    tmpl = tmpl.replace("{question}", question)
    tmpl = tmpl.replace("{choice_a}", choice_a)
    tmpl = tmpl.replace("{choice_b}", choice_b)
    tmpl = tmpl.replace("{choice_c}", choice_c)
    tmpl = tmpl.replace("{choice_d}", choice_d)
    tmpl = tmpl.replace("{choice_e}", choice_e)
    return tmpl


def build_history_user_content(
    question: str,
    discharge_summaries: str = "",
) -> str:
    """Build the compact record of a past MCQ turn: the question, no A-E choices.

    This is how every past MCQ turn is replayed. Keeping all five choices
    shows later turns every distractor written for earlier ones, which is a
    problem for the source turns in particular: their options enumerate
    ``Note #N ... Headers: ...`` combinations, so an earlier turn's distractor
    can be a later turn's correct answer. Dropping the choices from the history
    removes that channel; the model's own selection is still replayed in full
    as ``{letter}: {choice text}``, so the exchange stays interpretable.

    The first turn keeps the discharge summaries, since every later turn needs
    them.
    """
    parts = []
    if discharge_summaries:
        parts.append(f"[Discharge Summaries]\n{discharge_summaries}")
    parts.append(f"Question: {question}")
    return "\n\n".join(parts)


def build_generate_user_content(
    question: str,
    discharge_summaries: str = "",
    prompt_template: Optional[str] = None,
    is_source: bool = False,
) -> str:
    """Build user message content for a generation-mode (free-text) turn.

    If *discharge_summaries* is provided (first turn), uses *prompt_template*
    (the full template with ``[Discharge Summaries]``). Otherwise uses the
    compact follow-up template for the content or source question.
    """
    if prompt_template:
        tmpl = prompt_template
        if discharge_summaries:
            tmpl = tmpl.replace("{discharge summaries}", discharge_summaries)
        else:
            tmpl = strip_discharge_summaries_block(tmpl)
    else:
        tmpl = (
            FOLLOWUP_GENERATE_SOURCE_TEMPLATE
            if is_source
            else FOLLOWUP_GENERATE_CONTENT_TEMPLATE
        )
    tmpl = tmpl.replace("{question}", question)
    return tmpl


def format_multiturn_messages(
    tokenizer,
    messages: List[Dict[str, str]],
) -> str:
    """Format a multi-turn messages list into a prompt string.

    Falls back to plain concatenation when the tokenizer lacks
    ``apply_chat_template``.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    parts = []
    for msg in messages:
        role = msg["role"]
        if role == "system":
            parts.append(msg["content"])
        elif role == "user":
            parts.append(msg["content"])
        elif role == "assistant":
            parts.append(msg["content"])
    return "\n\n".join(parts)


def resolve_temperature(temperature: Optional[float], model_name: str) -> float:
    """Default temperature: 0.6 for DeepSeek-R1-Distill-*, else 0.0."""
    if temperature is not None:
        return temperature
    return 0.6 if "DeepSeek-R1-Distill-" in model_name else 0.0


def resolve_max_tokens(max_tokens: int, model_name: str) -> int:
    """Default max_tokens: 4096 for DeepSeek-R1-Distill-* (chain-of-thought), else as given."""
    if "DeepSeek-R1-Distill-" in model_name and max_tokens == 2048:
        return 4096
    return max_tokens


def strip_think_tags(text: str) -> str:
    """Strip ``<think>...</think>`` reasoning from model output, returning only the final answer."""
    idx = text.rfind("</think>")
    if idx != -1:
        return text[idx + len("</think>"):].strip()
    return text


# Chat-template control tokens that must never appear in a model's answer.
# Some models do not reliably halt at their own turn boundary, and keep
# generating the *next* user turn -- inventing questions and
# answering them. vLLM stops on the model's configured EOS only, which misses
# this, so we stop on the literal strings as well. None of these can occur in a
# legitimate answer, so this is safe to apply to every model.
CONTROL_TOKEN_STOP_STRINGS = [
    "</s>",              # Mistral / Ministral end-of-sequence
    "[INST]",            # Mistral / Ministral user-turn opener
    "[/INST]",
    "<|im_end|>",        # ChatML (Qwen and others)
    "<|im_start|>",
    "<|eot_id|>",        # Llama 3
    "<|start_header_id|>",
    "<|end_of_text|>",
    "<end_of_turn>",     # Gemma
    "<start_of_turn>",
    "<|end|>",           # Phi-3 / Phi-4
    "<|user|>",
    "<|assistant|>",
    "<|endoftext|>",
    "<|eot|>",           # Llama 4
    "<|header_start|>",
    "<|header_end|>",
    # DeepSeek-R1-Distill-*: note these use FULL-WIDTH characters
    # (U+FF5C vertical bar, U+2581 underscore), not ASCII pipes.
    "<｜end▁of▁sentence｜>",
    "<｜User｜>",
    "<｜Assistant｜>",
]


def _instruction_lines(template: str, min_len: int = 30) -> List[str]:
    """Fixed instruction lines of a prompt template, i.e. no ``{placeholder}``.

    Lines shorter than *min_len* are skipped: they are things like ``Answer:``
    that a model could plausibly write itself.
    """
    return [
        line.strip()
        for line in template.splitlines()
        if len(line.strip()) >= min_len and "{" not in line
    ]


# Instruction text lifted from our own prompt templates. Some models finish
# their answer and then re-type the *next* question in full -- without emitting
# any control token, so the tokens above never fire -- after which the answer
# parser reads a letter out of the fabricated turn instead of the real one.
# These lines are our wording, not the model's, so stopping on them cannot
# truncate a genuine answer.
PROMPT_STOP_STRINGS = sorted({
    line
    for template in (
        FOLLOWUP_MCQ_TEMPLATE,
        FOLLOWUP_GENERATE_CONTENT_TEMPLATE,
        FOLLOWUP_GENERATE_SOURCE_TEMPLATE,
    )
    for line in _instruction_lines(template)
})

STOP_STRINGS = CONTROL_TOKEN_STOP_STRINGS + PROMPT_STOP_STRINGS


def get_sampling_params(temperature: float, max_tokens: int) -> SamplingParams:
    """Build vLLM SamplingParams with the given temperature and max_tokens.

    ``stop`` guards against models that run past their turn boundary and
    hallucinate the rest of the conversation (see ``STOP_STRINGS``). The stop
    string itself is excluded from the output.
    """
    return SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=STOP_STRINGS,
    )


def get_model_name(model_arg: str) -> str:
    """Extract model name from model path (last path component)."""
    return model_arg.split("/")[-1] if "/" in model_arg else model_arg


# Sentinel that cannot occur in a chat template, used to locate where our own
# content starts in the rendered prompt.
_PROBE = "<<<USER_CONTENT>>>"
_PROBE_REPLY = "<<<ASSISTANT_REPLY>>>"
# Single-brace placeholders a Jinja template cannot fill (Jinja only expands
# ``{{ ... }}``). Mistral's default system message contains {today}/{yesterday},
# which mistral_common substitutes but the Jinja path leaves as literal text.
_UNRENDERED_PLACEHOLDER_RE = re.compile(r"(?<!\{)\{[a-z_][a-z0-9_]*\}(?!\})")


def log_chat_template_prefix(tokenizer, model: str = "") -> None:
    """Print what the model's chat template wraps around our content.

    Chat templates can silently inject a default system prompt when no system
    message is supplied (Mistral/Ministral prepend ~2.4k characters of vendor
    prompt this way), and can leave single-brace placeholders unrendered. Both
    change what the model actually sees with no other visible signal, so log the
    rendered wrapper once at startup.
    """
    if not hasattr(tokenizer, "apply_chat_template"):
        print("[Chat template] tokenizer has no apply_chat_template; prompts sent verbatim\n")
        return
    try:
        single = tokenizer.apply_chat_template(
            [{"role": "user", "content": _PROBE}],
            tokenize=False,
            add_generation_prompt=True,
        )
        multi = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": _PROBE},
                {"role": "assistant", "content": _PROBE_REPLY},
                {"role": "user", "content": _PROBE},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as exc:  # noqa: BLE001 - diagnostics must never break a run
        print(f"[Chat template] could not render probe prompt: {type(exc).__name__}: {exc}\n")
        return

    idx = single.find(_PROBE)
    prefix = single[:idx] if idx != -1 else single
    suffix = single[idx + len(_PROBE):] if idx != -1 else ""

    # What separates the assistant's reply from the next user turn -- i.e. the
    # string the model must emit to end its turn.
    i = multi.find(_PROBE_REPLY)
    terminator = ""
    if i != -1:
        rest = multi[i + len(_PROBE_REPLY):]
        j = rest.find(_PROBE)
        terminator = rest[:j] if j != -1 else rest

    def _show(s: str, limit: int = 300) -> str:
        s = s.replace("\n", "\\n")
        return s if len(s) <= limit else s[:limit] + f"... [+{len(s) - limit} chars]"

    print("[Chat template]")
    print(f"  model                  : {model or '(unknown)'}")
    print(f"  eos_token              : {getattr(tokenizer, 'eos_token', None)!r} "
          f"(id={getattr(tokenizer, 'eos_token_id', None)})")
    print(f"  prepended before our content ({len(prefix)} chars): {_show(prefix)}")
    print(f"  appended after our content   ({len(suffix)} chars): {_show(suffix, 120)}")
    print(f"  assistant turn terminator    : {_show(terminator, 120)!r}")

    if len(prefix) > 200:
        print(f"  !! WARNING: template injected a default system prompt "
              f"({len(prefix)} chars) even though none was passed.")
        print("     Pass --system-prompt explicitly if this is not intended.")
    placeholders = sorted(set(_UNRENDERED_PLACEHOLDER_RE.findall(prefix)))
    if placeholders:
        print(f"  !! WARNING: unrendered placeholders reach the model as literal text: "
              f"{', '.join(placeholders)}")
        print("     For Mistral-family models this is fixed by tokenizer_mode='mistral'.")
    print(f"  stop: control tokens    : {CONTROL_TOKEN_STOP_STRINGS}")
    for line in PROMPT_STOP_STRINGS:
        print(f"  stop: prompt line      : {line[:96]}{'...' if len(line) > 96 else ''}")
    print()


def resolve_local_snapshot(model: str, hf_cache_dir: Optional[str] = None) -> Optional[str]:
    """Resolve a HF repo id to its local snapshot directory, without network.

    ``mistral_common`` resolves tokenizers by listing files on the Hub, which
    fails when the Hub is unreachable or ``HF_HUB_OFFLINE=1`` is set. Passing
    the snapshot directory directly keeps everything on the local cache.
    Returns None if the model is not cached locally.
    """
    if os.path.isdir(model):
        return model
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(
            repo_id=model,
            local_files_only=True,
            cache_dir=hf_cache_dir,
        )
    except Exception as exc:  # noqa: BLE001 - fall back to the plain repo id
        print(f"[Model loading config] could not resolve local snapshot for "
              f"{model!r}: {type(exc).__name__}")
        return None


def has_native_mistral_format(model_path: Optional[str]) -> bool:
    """True if the checkpoint ships Mistral's native format alongside the HF one.

    Detected by file presence rather than by model name: ``tekken.json`` +
    ``params.json`` is exactly the condition under which
    ``tokenizer_mode/config_format/load_format="mistral"`` are usable. Older
    Mistral checkpoints (Mistral-7B-v0.3, Mixtral, ...) ship only the HF format
    and are correctly excluded.
    """
    if not model_path or not os.path.isdir(model_path):
        return False
    return all(
        os.path.isfile(os.path.join(model_path, name))
        for name in ("tekken.json", "params.json")
    )


def create_llm(
    model: str,
    hf_cache_dir: Optional[str] = None,
    tensor_parallel_size: int = 1,
    max_model_len: int = 15000,
    gpu_memory_utilization: float = 0.9,
):
    """
    Create vLLM LLM and tokenizer with shared defaults.
    Returns (llm, tokenizer).

    If *hf_cache_dir* is None, vLLM and HuggingFace use the standard
    resolution order ($HF_HUB_CACHE -> $HF_HOME -> ~/.cache/huggingface).

    Checkpoints shipping Mistral's native format are loaded with
    ``tokenizer_mode/config_format/load_format="mistral"``. This is not a
    performance choice: the HF Jinja template for these models injects a ~2.4k
    character default system prompt (with unrendered ``{today}`` placeholders)
    whenever no system message is passed, while ``mistral_common`` injects
    nothing -- matching every other model in the suite.
    """
    print(
        "[Model loading config]\n"
        f"  model id/path          : {model}\n"
        f"  HF_HOME                : {os.environ.get('HF_HOME')}\n"
        f"  HF_HUB_CACHE           : {os.environ.get('HF_HUB_CACHE')}\n"
        f"  HF_DATASETS_CACHE      : {os.environ.get('HF_DATASETS_CACHE')}\n"
        f"  TRANSFORMERS_CACHE     : {os.environ.get('TRANSFORMERS_CACHE')}\n"
        f"  vLLM download_dir      : {hf_cache_dir if hf_cache_dir else '(default)'}\n"
    )
    if os.path.isabs(model) and os.path.exists(model):
        model_path = model
        print(f"Loading model from path: {model_path}...")
    else:
        model_path = model
        print(f"Loading model '{model_path}' from HuggingFace cache...")
    llm_kwargs = dict(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        trust_remote_code=False,
        enforce_eager=True,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    if hf_cache_dir:
        llm_kwargs["download_dir"] = hf_cache_dir

    # Mistral-native checkpoints: use their own tokenizer/config/weight format.
    snapshot = resolve_local_snapshot(model_path, hf_cache_dir)
    if has_native_mistral_format(snapshot):
        # mistral_common resolves tokenizers via the Hub; pass the local
        # snapshot directory so this works offline as well.
        llm_kwargs["model"] = snapshot
        llm_kwargs.update(
            tokenizer_mode="mistral",
            config_format="mistral",
            load_format="mistral",
        )
        print(
            "[Mistral native format detected]\n"
            f"  snapshot dir           : {snapshot}\n"
            "  tokenizer_mode         : mistral\n"
            "  config_format          : mistral\n"
            "  load_format            : mistral\n"
            "  -> no default system prompt is injected (matches other models)\n"
        )

    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    log_chat_template_prefix(tokenizer, model)
    return llm, tokenizer


def add_common_vllm_args(parser: argparse.ArgumentParser) -> None:
    """Add common vLLM/model arguments to an ArgumentParser (same defaults as evaluate scripts)."""
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model id or absolute path to a local model directory",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=str,
        default=None,
        help=(
            "HuggingFace cache directory. If not provided, falls back to "
            "$HF_HUB_CACHE / $HF_HOME or vLLM's default."
        ),
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=15000,
        help="Maximum model length",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization ratio (0.0-1.0)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (default: 0.0, or 0.6 for DeepSeek-R1-Distill-*)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate (default: 2048, or 4096 for DeepSeek-R1-Distill-*)",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="System prompt for the model",
    )
