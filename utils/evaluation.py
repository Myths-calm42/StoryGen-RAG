"""
utils/evaluation.py
----------------------
Compares generation with retrieval (RAG) vs. without retrieval, on a
handful of test instructions, to evaluate the impact of retrieval on
narrative consistency and hallucination.
"""

import logging
from dataclasses import dataclass
from typing import List

from .generator import StoryGenerator
from .prompt_builder import build_prompt
from .retrieval import StoryVectorStore

logger = logging.getLogger(__name__)


@dataclass
class EvalCase:
    instruction: str


@dataclass
class EvalResult:
    instruction: str
    with_retrieval: str
    without_retrieval: str


def run_comparison(
    test_cases: List[EvalCase],
    vector_store: StoryVectorStore,
    generator: StoryGenerator,
    top_k: int = 5,
    max_new_tokens: int = 500,
) -> List[EvalResult]:
    results = []
    for case in test_cases:
        logger.info(f"Evaluating instruction: {case.instruction[:60]}...")

        retrieved = vector_store.search(case.instruction, top_k=top_k)
        prompt_with = build_prompt(case.instruction, retrieved)
        output_with = generator.generate(prompt_with, max_new_tokens=max_new_tokens)

        prompt_without = build_prompt(case.instruction, [])
        output_without = generator.generate(prompt_without, max_new_tokens=max_new_tokens)

        results.append(EvalResult(
            instruction=case.instruction,
            with_retrieval=output_with,
            without_retrieval=output_without,
        ))
    return results


def format_results_markdown(results: List[EvalResult]) -> str:
    lines = ["# Evaluation: With Retrieval vs. Without Retrieval\n"]
    lines.append(
        "Read each pair and note (by hand, or with an LLM judge) whether "
        "the retrieval-augmented version is more consistent with prior "
        "story context, less prone to contradicting established facts, "
        "and less repetitive.\n"
    )

    for i, r in enumerate(results, start=1):
        lines.append(f"## Case {i}\n")
        lines.append(f"**Instruction:** {r.instruction}\n")
        lines.append(f"**With retrieval:**\n> {r.with_retrieval}\n")
        lines.append(f"**Without retrieval:**\n> {r.without_retrieval}\n")
        lines.append("**Notes:** _(fill in manually: consistency, hallucination, relevance)_\n")
        lines.append("---\n")

    return "\n".join(lines)