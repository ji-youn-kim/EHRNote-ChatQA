#!/usr/bin/env python
# coding: utf-8

"""
Evaluate multi-turn QA using locally-served models via vLLM.

Two answer formats are supported:

- multiple choice (default): each turn presents the question with its A-E
  choices and the model answers with a letter.
- free-text generation (``--generate``): each turn presents the question
  with no choices and the model answers in prose. Source-location turns
  ask for the notes/headers supporting the model's own previous answer.

In both cases the model is asked the next question with the conversation so
far (its own prior outputs as assistant messages) formatted via the
tokenizer's native chat template. Past multiple-choice turns are replayed
compactly -- the question without its A-E choices, plus the answer the model
chose as ``{letter}: {choice text}`` -- so earlier turns' distractors stay
out of later turns' context. The current turn always shows all five choices.

Input is the released two-table dataset (``notes.csv`` + ``questions.csv``);
see core/release_data.py. Output is one row per subject with the per-turn
model outputs as a JSON blob in ``model_answer``.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Make sibling subfolders importable when this script is launched directly.
import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _sub in ("core", "data_generation", "evaluation", "scoring", ""):
    _p = str(_PROJECT_ROOT / _sub) if _sub else str(_PROJECT_ROOT)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from multiturn_gpt import DATA_DIR
from release_data import load_release_samples, sort_turn_keys
from model_answer_parsing import extract_answer_letter
from multiturn_gpt_evaluate_common import (
    GENERATE_SOURCE_QUESTION,
    add_common_vllm_args,
    build_generate_user_content,
    build_history_user_content,
    build_mcq_user_content,
    create_llm,
    format_multiturn_messages,
    get_model_name,
    get_sampling_params,
    load_prompt_template,
    resolve_max_tokens,
    resolve_temperature,
    strip_think_tags,
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate multi-turn QA (multiple choice, or free-text with "
            "--generate) using locally-served models via vLLM."
        ),
    )
    parser.add_argument(
        "--notes-csv",
        type=str,
        default=None,
        help=(
            "notes.csv from the released dataset (one row per discharge "
            "summary). If not provided, defaults to <data-dir>/notes.csv."
        ),
    )
    parser.add_argument(
        "--questions-csv",
        type=str,
        default=None,
        help=(
            "questions.csv from the released dataset (one row per turn). "
            "If not provided, defaults to <data-dir>/questions.csv."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=(
            "Directory holding notes.csv / questions.csv, and where outputs are "
            "written. If not provided, defaults to DATA_DIR from multiturn_gpt."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory for the model-answer CSV. If not provided, defaults to "
            "--data-dir (keep it separate to avoid mixing answers with the "
            "dataset)."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help=(
            "Output CSV file with model predictions. "
            "If not provided, derived from the model name."
        ),
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help=(
            "Free-text generation mode: the model answers in prose instead of "
            "selecting from A-E. Content turns use "
            "prompt/multiturn_gpt_evaluate_generate_content.txt; source turns "
            "use prompt/multiturn_gpt_evaluate_generate_source.txt and ask for "
            "the sources supporting the model's own previous answer. Answers "
            "are stored verbatim for downstream scoring; no letter parsing is "
            "performed and no unparseable-answer stats file is written."
        ),
    )
    add_common_vllm_args(parser)
    parser.set_defaults(max_model_len=50000)
    args = parser.parse_args()

    if args.data_dir:
        data_dir = Path(args.data_dir)
        if not data_dir.is_absolute():
            data_dir = data_dir.expanduser().resolve()
    else:
        data_dir = DATA_DIR

    model_name = get_model_name(args.model)
    args.temperature = resolve_temperature(args.temperature, model_name)
    args.max_tokens = resolve_max_tokens(args.max_tokens, model_name)
    is_deepseek_model = "DeepSeek-R1-Distill-" in model_name

    if args.notes_csv is None:
        args.notes_csv = str(data_dir / "notes.csv")
    if args.questions_csv is None:
        args.questions_csv = str(data_dir / "questions.csv")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else data_dir

    mode_suffix = "_generate" if args.generate else ""

    if args.output_csv is None:
        args.output_csv = str(output_dir / f"answers_{model_name}{mode_suffix}.csv")

    output_csv_path = Path(args.output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Invalid-parse statistics. Generation-mode answers are free text, so
    # there is no letter to parse and no stats file to write.
    stats_enabled = not args.generate
    invalid_parsed_turn_count = 0
    invalid_parsed_row_count = 0
    invalid_parsed_details: List[Dict[str, str]] = []
    stats_path = output_dir / f"{model_name}{mode_suffix}.txt"

    def save_stats_file() -> None:
        if not stats_enabled:
            return
        with open(stats_path, "w", encoding="utf-8") as f:
            f.write(f"invalid_parsed_turn_count: {invalid_parsed_turn_count}\n")
            f.write(f"invalid_parsed_row_count: {invalid_parsed_row_count}\n")
            f.write("\nDetailed list of invalid parsed answers:\n")
            f.write("=" * 80 + "\n")
            for i, detail in enumerate(invalid_parsed_details, 1):
                f.write(f"\n[{i}] subject_id: {detail['subject_id']}\n")
                f.write(f"    question_key: {detail['question_key']}\n")
                f.write(f"    model_answer: {detail['model_answer']}\n")
                f.write("-" * 80 + "\n")

        # Mirror copy for the scoring pipeline
        unparseable_dir = output_csv_path.parent / "scoring_unparseable"
        unparseable_dir.mkdir(parents=True, exist_ok=True)
        unparseable_file = unparseable_dir / f"{model_name}{mode_suffix}.txt"
        with open(unparseable_file, "w", encoding="utf-8") as f:
            f.write(f"invalid_parsed_turn_count: {invalid_parsed_turn_count}\n")
            f.write(f"invalid_parsed_row_count: {invalid_parsed_row_count}\n")
            f.write("\nDetailed list of invalid parsed answers:\n")
            f.write("=" * 80 + "\n")
            for i, detail in enumerate(invalid_parsed_details, 1):
                f.write(f"\n[{i}] subject_id: {detail['subject_id']}\n")
                f.write(f"    question_key: {detail['question_key']}\n")
                f.write(f"    model_answer: {detail['model_answer']}\n")
                f.write("-" * 80 + "\n")

    # Load the first-turn prompt template(s) for the selected answer format.
    if args.generate:
        prompt_template = None
        prompt_template_generate_content = load_prompt_template(
            "multiturn_gpt_evaluate_generate_content.txt"
        )
        prompt_template_generate_source = load_prompt_template(
            "multiturn_gpt_evaluate_generate_source.txt"
        )
    else:
        prompt_template = load_prompt_template()
        prompt_template_generate_content = None
        prompt_template_generate_source = None

    llm, tokenizer = create_llm(
        model=args.model,
        hf_cache_dir=args.hf_cache_dir,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    if not hasattr(tokenizer, "apply_chat_template"):
        base, ext = os.path.splitext(args.output_csv)
        if not ext:
            ext = ".csv"
        args.output_csv = f"{base}_no_chat_template{ext}"
        output_csv_path = Path(args.output_csv)
        print(
            f"Tokenizer has no chat template; results will be saved to: "
            f"{args.output_csv}"
        )

    sampling_params = get_sampling_params(args.temperature, args.max_tokens)

    print(f"Loading {args.notes_csv} + {args.questions_csv}...")
    samples = load_release_samples(args.notes_csv, args.questions_csv)
    total_rows = len(samples)
    print(f"Total samples (subjects): {total_rows}")

    answer_col = "model_answer"
    result_rows: List[Dict[str, str]] = []

    processed_rows = 0

    for sample in samples:
        subject_id = sample["subject_id"]
        discharge_summaries = sample["discharge_summaries"]
        qa_dict = sample["qas"]
        if not qa_dict:
            continue

        # Turn order: Q1 -> Q1-1 -> Q2 -> Q2-1 -> ...
        sorted_keys = sort_turn_keys(list(qa_dict.keys()))

        # ---------------------- GENERATION MODE ----------------------
        if args.generate:
            model_answers: Dict[str, str] = {}

            # Persistent chat, same as the MCQ path below.
            mt_gen_messages: List[Dict[str, str]] = []
            if args.system_prompt:
                mt_gen_messages.append(
                    {"role": "system", "content": args.system_prompt}
                )
            is_first_gen_turn = True

            for key in sorted_keys:
                qa = qa_dict[key]

                # Source turns ask for the evidence behind the model's own
                # previous answer; content turns ask the stored question.
                if "-" in key:
                    question_text = GENERATE_SOURCE_QUESTION
                    prompt_tmpl = prompt_template_generate_source
                    is_source = True
                else:
                    question_text = qa["Question"]
                    prompt_tmpl = prompt_template_generate_content
                    is_source = False

                if is_first_gen_turn:
                    user_content = build_generate_user_content(
                        question=question_text,
                        discharge_summaries=discharge_summaries,
                        prompt_template=prompt_tmpl,
                        is_source=is_source,
                    )
                    is_first_gen_turn = False
                else:
                    user_content = build_generate_user_content(
                        question=question_text,
                        is_source=is_source,
                    )

                prompt_messages = mt_gen_messages + [
                    {"role": "user", "content": user_content}
                ]
                formatted_prompt = format_multiturn_messages(
                    tokenizer, prompt_messages
                )
                outputs = llm.generate([formatted_prompt], sampling_params)
                raw_model_answer_text = outputs[0].outputs[0].text.strip()


                # Strip <think>...</think> CoT so it neither re-enters the
                # prompt nor reaches the scorer. The stored answer is the
                # stripped text; only the chat history substitutes "N/A" for
                # an empty turn, which some tokenizers reject.
                processed_answer = strip_think_tags(raw_model_answer_text)
                mt_gen_messages.append(
                    {"role": "user", "content": user_content}
                )
                mt_gen_messages.append(
                    {"role": "assistant", "content": processed_answer or "N/A"}
                )

                model_answers[key] = processed_answer

            result_rows.append({
                "subject_id": subject_id,
                "qa_category": sample["qa_category"],
                answer_col: json.dumps(model_answers),
            })
            processed_rows += 1
            print(
                f"[{model_name}] Processed {processed_rows} / {total_rows} "
                f"samples (generation mode)..."
            )
            continue

        # ------------------- MULTIPLE-CHOICE MODE --------------------
        model_answers: Dict[str, str] = {}
        row_has_invalid_parsed_answer = False

        # Persistent chat: each turn's user message and assistant reply are
        # appended so subsequent turns see the model's own prior answers.
        mt_messages: List[Dict[str, str]] = []
        if args.system_prompt:
            mt_messages.append({"role": "system", "content": args.system_prompt})
        is_first_mt_turn = True

        for key in sorted_keys:
            qa = qa_dict[key]
            question_text = qa["Question"]

            turn_summaries = discharge_summaries if is_first_mt_turn else ""
            user_content = build_mcq_user_content(
                question=question_text,
                choice_a=qa["A"], choice_b=qa["B"], choice_c=qa["C"],
                choice_d=qa["D"], choice_e=qa["E"],
                discharge_summaries=turn_summaries,
                prompt_template=prompt_template if is_first_mt_turn else None,
            )
            # What this turn looks like once it is history: the question
            # without its choices, so earlier turns' distractors stay out of
            # later turns' context. The current turn is unchanged.
            history_content = build_history_user_content(
                question_text, turn_summaries
            )
            is_first_mt_turn = False

            prompt_messages = mt_messages + [
                {"role": "user", "content": user_content}
            ]
            formatted_prompt = format_multiturn_messages(tokenizer, prompt_messages)
            outputs = llm.generate([formatted_prompt], sampling_params)
            model_answer_text = outputs[0].outputs[0].text.strip()

            letter = extract_answer_letter(
                model_answer_text,
                is_deepseek=is_deepseek_model,
            )



            # Strip <think>...</think> CoT from the assistant turn to keep
            # subsequent prompts compact; fall back to raw output / "N/A"
            # so the tokenizer never receives empty assistant content.
            if letter in ("A", "B", "C", "D", "E"):
                # The history drops the choices, so replay the selected one in
                # full; a bare letter would be meaningless without them.
                assistant_content = f"{letter}: {qa.get(letter, '')}"
            else:
                assistant_content = (
                    strip_think_tags(model_answer_text) or model_answer_text or "N/A"
                )
            mt_messages.append({"role": "user", "content": history_content})
            mt_messages.append({"role": "assistant", "content": assistant_content})

            model_answers[key] = model_answer_text

            if letter not in ("A", "B", "C", "D", "E"):
                invalid_parsed_turn_count += 1
                row_has_invalid_parsed_answer = True
                invalid_parsed_details.append({
                    "subject_id": str(subject_id),
                    "question_key": key,
                    "model_answer": model_answer_text,
                })

        if row_has_invalid_parsed_answer:
            invalid_parsed_row_count += 1

        result_rows.append({
            "subject_id": subject_id,
            "qa_category": sample["qa_category"],
            answer_col: json.dumps(model_answers),
        })
        save_stats_file()
        processed_rows += 1
        print(
            f"[{model_name}] Processed {processed_rows} / {total_rows} "
            f"samples... "
            f"(invalid parsed turns: {invalid_parsed_turn_count}, "
            f"rows with invalid parsed: {invalid_parsed_row_count})"
        )

    print(f"\nSaving results to {args.output_csv}...")
    df = pd.DataFrame(result_rows, columns=["subject_id", "qa_category", answer_col])
    df.to_csv(args.output_csv, index=False)
    print(
        f"Done! Saved {len(df)} rows "
        f"(processed {processed_rows} of {total_rows} samples)."
    )
    save_stats_file()
    if stats_enabled:
        print(f"Final statistics saved to {stats_path}")


if __name__ == "__main__":
    main()
