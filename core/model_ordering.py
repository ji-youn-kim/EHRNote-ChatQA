"""
Canonical model display order, shared across this codebase.

The paper evaluates the three API models 'gpt-5.4', 'gpt-5.4-mini', and
'gemini-3-flash-preview' plus the open-weights local models listed below.
Add new models here and they will appear in this order wherever
``MODEL_ORDER`` is imported (e.g., scoring summary tables).
"""

from typing import Dict, List

MODEL_ORDER: List[str] = [
    "gpt-5.4",
    "gpt-5.4-mini",
    "gemini-3-flash-preview",
    "Llama-4-Scout-17B-16E-Instruct",
    "Qwen3-Next-80B-A3B-Instruct",
    "Qwen3-30B-A3B-Instruct-2507",
    "Qwen3-4B-Instruct-2507",
    "DeepSeek-R1-Distill-Llama-70B",
    "DeepSeek-R1-Distill-Qwen-32B",
    "DeepSeek-R1-Distill-Qwen-14B",
    "DeepSeek-R1-Distill-Llama-8B",
    "DeepSeek-R1-Distill-Qwen-7B",
    "Ministral-3-14B-Instruct-2512",
    "Ministral-3-8B-Instruct-2512",
    "Ministral-3-3B-Instruct-2512",
    "Phi-4-mini-instruct",
    "Phi-3.5-mini-instruct",
    "medgemma-27b-it",
    "medgemma-4b-it",
    "MediPhi-Instruct",
    "MediPhi-Clinical",
    "MediPhi-PubMed",
]

# Position-lookup table; unknown models get len(MODEL_ORDER) so they sort last.
MODEL_ORDER_MAP: Dict[str, int] = {name: i for i, name in enumerate(MODEL_ORDER)}
