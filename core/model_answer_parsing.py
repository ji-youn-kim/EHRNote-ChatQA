#!/usr/bin/env python
# coding: utf-8

"""
Model answer parsing utilities for extracting answer letters (A, B, C, D, or E) from model outputs.

This module provides consistent parsing logic across all evaluation scripts.
"""

import re
from typing import Optional


def _strip_pre_think(text: str) -> str:
    """
    Drop everything up to and including the last ``</think>`` so parsing only sees
    the model's final answer, not its reasoning. Mirrors
    multiturn_gpt_evaluate_common.strip_think_tags. Tolerates dangling
    ``</think>`` (no opening tag), which DeepSeek-R1-Distill models often emit.
    """
    if not text or not isinstance(text, str):
        return text
    idx = text.rfind("</think>")
    if idx != -1:
        return text[idx + len("</think>"):].strip()
    return text


def extract_answer_letter_default(model_answer_text: str) -> Optional[str]:
    """
    Default extraction of the answer letter (A, B, C, D, or E) from model answer text.
    Intended for models that output either:
      - Just the option letter (e.g., "A", "**B**", "A1")
      - The option letter followed by punctuation / rationale (e.g., "C: ...", "D) ...", "E. ...")
      - Phrases like "The correct answer is A", "correct answer is **B**", etc.
    
    Strategy:
      1. Look at the *start* of the string for a (possibly formatted) single letter A–E (including "A1").
      2. If that fails, look for phrases like "option C" / "Option C" etc.
      3. If that fails, look for "correct answer is X" patterns.
    """
    if not model_answer_text or not isinstance(model_answer_text, str):
        return None

    # Strip reasoning so we only see the model's final answer block.
    text = _strip_pre_think(model_answer_text).strip()
    if not text:
        return None

    # IMPORTANT: Do NOT immediately treat the *leading* capital letter as the
    # answer, because strings like "Answer: C" would be mis-parsed as "A".
    # Instead, we first look for explicit answer phrases and only if nothing
    # matches do we fall back to simpler heuristics.
    
    # Remove markdown formatting for matching
    text_clean = re.sub(r'\*+', '', text)

    # Pattern 1: "correct answer is X" / "Answer: X" / "Answer is X" / "Conclusion: X" / "\boxed{X}"
    correct_answer_patterns = [
        r'\\boxed\{([A-E])\}',  # LaTeX boxed: $\boxed{D}$
        r'(?:the\s+)?correct\s+answer\s+is\s*[:\s]*\(*([A-E])\)*\b',
        r'answer\s*[:\-]\s*[\(\["\'`]*([A-E])\b',
        r'answer\s+is\s+[\(\["\'`]*([A-E])\b',
        r'conclusion\s*[:\-]\s*[\(\["\'`]*([A-E])\b',
    ]
    for pattern in correct_answer_patterns:
        matches = list(re.finditer(pattern, text_clean, re.IGNORECASE))
        if matches:
            # Use the last match (most likely to be the final answer)
            return matches[-1].group(1).upper()

    # Pattern 2: Phrases like "option C", "Option B", etc.
    option_match = re.search(r'[Oo]ption\s+([A-E])\b', text_clean)
    if option_match:
        return option_match.group(1).upper()

    # Pattern 3: "correct answer is:" followed by newlines and then a letter with colon
    correct_answer_colon_match = re.search(
        r'(?:the\s+)?correct\s+answer\s+is\s*:.*?^([A-E])\s*:',
        text,
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    if correct_answer_colon_match:
        return correct_answer_colon_match.group(1).upper()

    # Pattern 4: Look for standalone letter followed by colon (not at start, but as
    # a line or after newline) – e.g. "...\n\nA: Note #3..."
    standalone_letter_colon = re.search(
        r'(?:^|\n)\s*([A-E])\s*:\s+[A-Z]',
        text,
        re.IGNORECASE | re.MULTILINE,
    )
    if standalone_letter_colon:
        return standalone_letter_colon.group(1).upper()

    # Pattern 5 (fallback): Leading single letter with optional markdown/punctuation.
    # Handles:
    #   "A"
    #   "A1" (extracts "A")
    #   "A: ...", "B) ...", "C. ..."
    #   "**E**", "**B**: rationale"
    #
    # ``(?![A-Za-z])`` requires the letter to stand alone, so a reply that merely
    # BEGINS with a word starting A-E is not mistaken for a choice. Without it,
    # "Answer: None of the options apply" scores as A, "Based on the notes..." as
    # B, and "Definitely A" as D -- silently, since a letter was returned and the
    # turn is never flagged as unparseable. A digit may still follow ("A1").
    leading_match = re.match(
        r'^[*\s]*([A-E])(?![A-Za-z])[*\s]*[:\.)]?\d*',
        text,
        re.IGNORECASE,
    )
    if leading_match:
        return leading_match.group(1).upper()

    return None


def extract_answer_letter_deepseek(model_answer_text: str) -> Optional[str]:
    """
    Extract the answer letter for DeepSeek-R1-Distill-* models.
    
    Heuristic (applied from the end of the text):
    1. Search entire text (prioritizing end) for "Answer: x", "Answer is x", "correct answer is x", etc.
    2. If a line consists of just a single letter A–E, use that.
    3. If not, fallback to the default extraction logic.
    """
    if not model_answer_text or not isinstance(model_answer_text, str):
        return None

    # Strip reasoning so we only see the model's final answer block.
    text = _strip_pre_think(model_answer_text).strip()
    if not text:
        return None

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None

    # Pattern 1: Look for "Answer: x", "Answer is x", "correct answer is x", etc.
    # Search from the end backwards (last 20 lines) for better accuracy
    for line in reversed(lines[-20:]):
        # Remove markdown formatting for matching
        line_clean = re.sub(r'\*+', '', line)
        
        # Various answer patterns (order matters - more specific first)
        patterns = [
            r'\\boxed\{([A-E])\}',  # LaTeX boxed: $\boxed{D}$
            r'(?:^|\s)(?:the\s+)?correct\s+answer\s+is\s*[:\s]*\(*([A-E])\)*\b',  # "correct answer is A", "correct answer is: A", "correct answer is (A)", "correct answer is **(D)**"
            r'(?:^|\s)answer\s*[:\-]\s*[\(\["\'`]*([A-E])\b',  # "Answer: A" or "Answer - A"
            r'(?:^|\s)answer\s+is\s+[\(\["\'`]*([A-E])\b',  # "Answer is A"
            r'(?:^|\s)so\s+the\s+correct\s+answer\s+is\s+[\(\["\'`]*([A-E])\b',  # "So the correct answer is A"
            r'(?:^|\s)therefore\s+the\s+correct\s+answer\s+is\s+[\(\["\'`]*([A-E])\b',  # "Therefore the correct answer is A"
            r'(?:^|\s)conclusion\s*[:\-]\s*[\(\["\'`]*([A-E])\b',  # "Conclusion: C"
        ]

        for pattern in patterns:
            m = re.search(pattern, line_clean, re.IGNORECASE)
            if m:
                return m.group(1).upper()

        # Pattern 2: Line is just a single letter A–E (possibly with markdown)
        line_clean_simple = re.sub(r'[*\s]', '', line)
        if re.fullmatch(r'[A-Ea-e]', line_clean_simple):
            return line_clean_simple.upper()

    # Also search the entire text (not just lines) for answer patterns
    text_clean = re.sub(r'\*+', '', text)
    patterns_full = [
        r'\\boxed\{([A-E])\}',  # LaTeX boxed: $\boxed{D}$
        r'(?:the\s+)?correct\s+answer\s+is\s*[:\s]*\(*([A-E])\)*\b',  # "correct answer is A", "correct answer is: A", "correct answer is (A)"
        r'answer\s*[:\-]\s*[\(\["\'`]*([A-E])\b',
        r'answer\s+is\s+[\(\["\'`]*([A-E])\b',
        r'conclusion\s*[:\-]\s*[\(\["\'`]*([A-E])\b',  # "Conclusion: C"
    ]
    for pattern in patterns_full:
        matches = list(re.finditer(pattern, text_clean, re.IGNORECASE))
        if matches:
            # Use the last match (most likely to be the final answer)
            return matches[-1].group(1).upper()
    
    # Pattern: Look for "correct answer is:" followed by newlines and then a letter with colon
    # This handles cases like "The correct answer is:\n\nB: Note #1..."
    correct_answer_colon_match = re.search(
        r'(?:the\s+)?correct\s+answer\s+is\s*:.*?^([A-E])\s*:', 
        text, 
        re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    if correct_answer_colon_match:
        return correct_answer_colon_match.group(1).upper()
    
    # Pattern: Look for standalone letter followed by colon (not at start, but as a line or after newline)
    # This handles cases like "...\n\nA: Note #3..."
    standalone_letter_colon = re.search(
        r'(?:^|\n)\s*([A-E])\s*:\s+[A-Z]', 
        text, 
        re.IGNORECASE | re.MULTILINE
    )
    if standalone_letter_colon:
        return standalone_letter_colon.group(1).upper()
    
    return extract_answer_letter_default(model_answer_text)


def extract_answer_letter_gemini(model_answer_text: str) -> Optional[str]:
    """
    Extract the answer letter for Gemini models (e.g., gemini-3-flash-preview).
    
    Gemini models often output:
    - Simple letters: "A", "B", etc.
    - "Answer: A" format
    - "The correct answer is B." format
    - "The correct answer is **C**." format (with markdown)
    - Long explanations that may contain the answer embedded in the text
    
    Strategy:
    1. First try the default parser (handles simple cases)
    2. If that fails, search the entire text for answer patterns (similar to DeepSeek)
    3. Look for patterns in both the full text and line-by-line (prioritizing end)
    """
    if not model_answer_text or not isinstance(model_answer_text, str):
        return None

    # Strip reasoning so we only see the model's final answer block.
    text = _strip_pre_think(model_answer_text).strip()
    if not text:
        return None

    # IMPORTANT: Do NOT call extract_answer_letter_default here first.
    # The default parser treats the *leading* capital letter as the answer,
    # which breaks common Gemini patterns like "Answer: C" (it would return "A").
    #
    # Instead, we search explicitly for answer patterns ("Answer: C",
    # "correct answer is B", etc.) and only as a last resort fall back to
    # very simple heuristics ("C" on its own line).
    #
    # Remove markdown formatting for pattern matching.
    text_clean = re.sub(r'\*+', '', text)
    
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    
    # Pattern 1: Search lines from the end backwards (last 10 lines) for answer patterns
    if lines:
        for line in reversed(lines[-10:]):
            line_clean = re.sub(r'\*+', '', line)
            
            # Various answer patterns (order matters - more specific first)
            patterns = [
                r'\\boxed\{([A-E])\}',  # LaTeX boxed: $\boxed{D}$
                r'(?:^|\s)(?:the\s+)?correct\s+answer\s+is\s*[:\s]*\(*([A-E])\)*\b',  # "correct answer is A", "correct answer is: A", "correct answer is (A)", "correct answer is **(D)**"
                r'(?:^|\s)answer\s*[:\-]\s*[\(\["\'`]*([A-E])\b',  # "Answer: A" or "Answer - A"
                r'(?:^|\s)answer\s+is\s+[\(\["\'`]*([A-E])\b',  # "Answer is A"
                r'(?:^|\s)so\s+the\s+correct\s+answer\s+is\s+[\(\["\'`]*([A-E])\b',  # "So the correct answer is A"
                r'(?:^|\s)therefore\s+the\s+correct\s+answer\s+is\s+[\(\["\'`]*([A-E])\b',  # "Therefore the correct answer is A"
                r'(?:^|\s)conclusion\s*[:\-]\s*[\(\["\'`]*([A-E])\b',  # "Conclusion: C"
            ]
            
            for pattern in patterns:
                m = re.search(pattern, line_clean, re.IGNORECASE)
                if m:
                    return m.group(1).upper()
            
            # Pattern 2: Line is just a single letter A–E (possibly with markdown)
            line_clean_simple = re.sub(r'[*\s]', '', line)
            if re.fullmatch(r'[A-Ea-e]', line_clean_simple):
                return line_clean_simple.upper()
    
    # Pattern 3: Search the entire text for answer patterns (not just lines)
    patterns_full = [
        r'\\boxed\{([A-E])\}',  # LaTeX boxed: $\boxed{D}$
        r'(?:the\s+)?correct\s+answer\s+is\s*[:\s]*\(*([A-E])\)*\b',
        r'answer\s*[:\-]\s*[\(\["\'`]*([A-E])\b',
        r'answer\s+is\s+[\(\["\'`]*([A-E])\b',
        r'conclusion\s*[:\-]\s*[\(\["\'`]*([A-E])\b',  # "Conclusion: C"
    ]
    for pattern in patterns_full:
        matches = list(re.finditer(pattern, text_clean, re.IGNORECASE))
        if matches:
            # Use the last match (most likely to be the final answer)
            return matches[-1].group(1).upper()
    
    # Pattern 4: Look for "correct answer is:" followed by newlines and then a letter with colon
    # This handles cases like "The correct answer is:\n\nB: Note #1..."
    correct_answer_colon_match = re.search(
        r'(?:the\s+)?correct\s+answer\s+is\s*:.*?^([A-E])\s*:', 
        text, 
        re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    if correct_answer_colon_match:
        return correct_answer_colon_match.group(1).upper()
    
    # Pattern 5: Look for standalone letter followed by colon (not at start, but as a line or after newline)
    # This handles cases like "...\n\nA: Note #3..."
    standalone_letter_colon = re.search(
        r'(?:^|\n)\s*([A-E])\s*:\s+[A-Z]', 
        text, 
        re.IGNORECASE | re.MULTILINE
    )
    if standalone_letter_colon:
        return standalone_letter_colon.group(1).upper()
    
    # Pattern 6: As a very last resort, if the entire cleaned text is just a
    # single letter A–E, treat that as the answer.
    text_simple = re.sub(r'[*\s]', '', text_clean)
    if re.fullmatch(r'[A-Ea-e]', text_simple):
        return text_simple.upper()
    
    return None


def extract_answer_letter(model_answer_text: str, is_deepseek: bool = False, is_gemini: bool = False) -> Optional[str]:
    """
    Wrapper that chooses the appropriate extraction logic based on model type.
    
    Args:
        model_answer_text: The raw model answer text
        is_deepseek: Whether this is a DeepSeek-R1-Distill-* model
        is_gemini: Whether this is a Gemini model
    
    Returns:
        The extracted answer letter (A, B, C, D, or E), or None if not found
    """
    if is_deepseek:
        return extract_answer_letter_deepseek(model_answer_text)
    if is_gemini:
        return extract_answer_letter_gemini(model_answer_text)
    return extract_answer_letter_default(model_answer_text)
