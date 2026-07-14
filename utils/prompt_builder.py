"""
utils/prompt_builder.py
--------------------------
Constructs the final generation prompt from retrieved context, story
summary/memory, and the user's instruction for the next chapter.
"""

import logging
from typing import List, Optional, Tuple

from .chunking import Chunk

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a skilled novelist continuing an ongoing story. Write the next "
    "chapter so that it:\n"
    "- does not contradict any previously established events, characters, or facts\n"
    "- keeps the same characters, their traits, and their relationships consistent\n"
    "- preserves the story's established writing style and tone\n"
    "- introduces meaningful plot progression, without needless repetition\n"
    "- includes natural dialogue where appropriate\n"
    "\n"
    "Do NOT invent your own chapter number or title (e.g. do not write "
    "'Chapter 5: ...'). Just write the continuation's prose directly, "
    "starting from the first sentence of the new content.\n"
)


def build_prompt(
    instruction: str,
    retrieved_chunks: List[Tuple[Chunk, float]],
    story_summary: Optional[str] = None,
    memory_summary: Optional[str] = None,
    max_context_chars: int = 4000,
) -> str:
    """
    Build the full prompt for the LLM generator.

    Args:
        instruction: the user's instruction for the next chapter
            (e.g. "Continue the story where Alice enters the forbidden forest.")
        retrieved_chunks: (Chunk, score) pairs from the vector store, most
            relevant first.
        story_summary: optional running summary of the story so far.
        memory_summary: optional formatted string of tracked characters/
            locations/events/relationships (see utils/memory.py).
        max_context_chars: cap on how much retrieved context to include,
            to keep the prompt within the model's context window.

    Returns:
        The fully assembled prompt string.
    """
    context_parts = []
    running_len = 0
    for chunk, score in retrieved_chunks:
        if running_len + len(chunk.text) > max_context_chars:
            break
        context_parts.append(chunk.text)
        running_len += len(chunk.text)

    retrieved_context = "\n\n---\n\n".join(context_parts) if context_parts else "(no relevant context retrieved)"

    sections = [SYSTEM_PROMPT]

    if story_summary:
        sections.append(f"### Story Summary So Far:\n{story_summary}")

    if memory_summary:
        sections.append(f"### Known Characters, Locations, and Events:\n{memory_summary}")

    sections.append(f"### Retrieved Relevant Passages:\n{retrieved_context}")
    sections.append(f"### Instruction for Next Chapter:\n{instruction}")
    sections.append("### Next Chapter:\n")

    prompt = "\n\n".join(sections)
    logger.info(f"Built prompt with {len(context_parts)} retrieved chunks, ~{len(prompt)} chars")
    return prompt