"""
utils/memory.py
------------------
Lightweight persistent memory store for a story: characters, locations,
important objects, events, and relationships, tracked as JSON and updated
after each generated chapter using the same LLM via an extraction prompt.

This is a pragmatic, LLM-based extraction approach rather than a formal
NER/knowledge-graph pipeline -- described honestly in the README as a
"persistent memory module," not a full knowledge graph.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Union

logger = logging.getLogger(__name__)

EMPTY_MEMORY: Dict[str, Any] = {
    "summary": "",      # running one-paragraph summary of the story so far
    "characters": {},   # name -> {age, role, traits, relationships}
    "locations": [],
    "objects": [],
    "events": [],
}

EXTRACTION_PROMPT_TEMPLATE = """You are extracting structured story memory. \
Given the existing memory (JSON) and a new chapter of text, update the memory \
with any new or changed characters, locations, objects, and events, and update \
the running summary to reflect the story so far in 3-5 sentences.

Respond with ONLY valid JSON, no preamble, no markdown code fences, matching \
exactly this schema:
{{
  "summary": "a short running summary of the whole story so far",
  "characters": {{"Name": {{"age": "", "role": "", "traits": ["..."], "relationships": {{"OtherName": "relation"}}}}}},
  "locations": ["..."],
  "objects": ["..."],
  "events": ["..."]
}}

Existing memory:
{existing_memory}

New chapter text:
{chapter_text}

Updated memory (JSON only):"""


class StoryMemory:
    """Tracks structured story memory across chapters."""

    def __init__(self):
        self.memory: Dict[str, Any] = json.loads(json.dumps(EMPTY_MEMORY))  # deep copy

    def update_from_chapter(self, chapter_text: str, generator) -> None:
        """
        Use the provided generator (a StoryGenerator instance) to extract
        updated memory from the new chapter text, merging into self.memory.
        """
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(
            existing_memory=json.dumps(self.memory, ensure_ascii=False),
            chapter_text=chapter_text[:4000],  # cap to keep extraction prompt manageable
        )
        raw_output = generator.generate(prompt, max_new_tokens=800, temperature=0.2)
        parsed = self._safe_parse_json(raw_output)

        if parsed is not None:
            self.memory = parsed
            logger.info("Memory updated successfully from chapter extraction")
        else:
            logger.warning("Memory extraction failed to parse; keeping previous memory unchanged")

    @staticmethod
    def _safe_parse_json(raw_output: str):
        """Attempt to parse JSON from model output, tolerating minor formatting issues."""
        # Strip markdown code fences if the model added them despite instructions
        cleaned = re.sub(r"^```(json)?|```$", "", raw_output.strip(), flags=re.MULTILINE).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to extract the first {...} block as a fallback
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    return None
            return None

    def get_summary(self) -> str:
        """Return the running story summary, or a placeholder if none yet."""
        return self.memory.get("summary") or "(no summary yet — generate a chapter first)"

    def as_prompt_string(self) -> str:
        """Format the current memory as a readable string for prompt injection."""
        lines = []

        if self.memory.get("characters"):
            lines.append("Characters:")
            for name, info in self.memory["characters"].items():
                traits = ", ".join(info.get("traits", []))
                rel = ", ".join(f"{k}: {v}" for k, v in info.get("relationships", {}).items())
                line = f"- {name}"
                if info.get("role"):
                    line += f" ({info['role']})"
                if traits:
                    line += f", traits: {traits}"
                if rel:
                    line += f", relationships: {rel}"
                lines.append(line)

        if self.memory.get("locations"):
            lines.append("Locations: " + ", ".join(self.memory["locations"]))

        if self.memory.get("objects"):
            lines.append("Important objects: " + ", ".join(self.memory["objects"]))

        if self.memory.get("events"):
            lines.append("Key events so far: " + "; ".join(self.memory["events"]))

        return "\n".join(lines) if lines else "(no memory recorded yet)"

    def save(self, file_path: Union[str, Path]) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.memory, f, ensure_ascii=False, indent=2)

    def load(self, file_path: Union[str, Path]) -> None:
        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as f:
            self.memory = json.load(f)
