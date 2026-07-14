# StoryGen AI — Long-Form Story Continuation with Retrieval-Augmented Generation

A retrieval-augmented long-form story continuation system: upload a story
(TXT/PDF), and generate a coherent next chapter that stays consistent with
previously established characters, locations, events, and relationships —
using semantic retrieval over prior chapters plus a lightweight persistent
memory module, rather than relying on an LLM's raw context window alone.

Built as a second Gen AI portfolio project, following a QLoRA fine-tuning
project (see the companion repo: `QLoRA-Finetune-Story-continuation-llm`).
Where that project demonstrated parameter-efficient fine-tuning, this one
demonstrates retrieval-augmented generation, vector search, and
lightweight structured memory — the other core techniques named in
Gen AI Research job descriptions (alongside LLMs and fine-tuning).

## Architecture

```
StoryGenAI/
  app.py                  Streamlit interface
  requirements.txt
  utils/
    pdf_loader.py          TXT/PDF -> plain text
    chunking.py             paragraph-aware semantic chunking
    embedding.py             Sentence Transformers embeddings
    retrieval.py             FAISS vector store + top-K search
    prompt_builder.py       assembles system + context + memory + instruction
    generator.py             instruct LLM generation (Qwen2.5 / Llama-3.1 / Gemma)
    memory.py                 persistent character/location/event/relationship memory
    evaluation.py             with-retrieval vs. without-retrieval comparison
  models/                   (local model cache, if used)
  vector_db/                 saved FAISS indexes
  generated/                  saved generated chapters
  data/                        example story files
  assets/                      screenshots
```

## How it works

1. **Upload** a story (TXT or PDF) — parsed to plain text.
2. **Chunk** the story into overlapping, paragraph-respecting segments.
3. **Embed** each chunk with `BAAI/bge-small-en-v1.5` (Sentence Transformers).
4. **Index** the embeddings in a FAISS vector store.
5. Given an **instruction** for the next chapter (e.g. "Continue the story
   where Alice enters the forbidden forest"), **retrieve** the most
   relevant prior chunks.
6. **Build a prompt** combining: a system prompt, the story's tracked
   memory (characters/locations/events/relationships), the retrieved
   context, and the instruction.
7. **Generate** the next chapter with an instruct LLM (Qwen2.5-Instruct by
   default; Llama-3.1-Instruct or Gemma also supported).
8. **Update memory**: an LLM extraction pass pulls new/changed characters,
   locations, objects, and events from the generated chapter into the
   persistent JSON memory store.

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then in the browser UI:
1. Upload a story file and click "Build vector index" (Tab 1)
2. Enter an instruction and click "Generate Next Chapter" (Tab 2)
3. View tracked characters, relationships, locations, and objects as
   cards (Tab 3)
4. View the story's running summary and event timeline (Tab 4)
5. Optionally run the with/without-retrieval comparison (Tab 5)

## Evaluation

The Evaluation tab generates the same instruction both with retrieval and
without (empty context), for direct side-by-side comparison. This is a
**qualitative** comparison meant to be read and annotated by hand — it is
not an automated metric. Look for:
- Whether the with-retrieval version stays consistent with established
  characters/events, and the without-retrieval version doesn't
- Whether the without-retrieval version hallucinates details not present
  in the story so far
- General coherence and relevance to the instruction in both cases

## Scope and honesty notes

- The "memory" module is an **LLM-based extraction pipeline**, described
  honestly as a lightweight persistent memory store — not a formal
  knowledge graph or NER system. It's a pragmatic approach that works
  well for a portfolio-scale project but would need more robust
  extraction (validation, deduplication, conflict resolution) for
  production use.
- **Automatic story summarization** is folded into the memory extraction
  pass (not a separate module) — each chapter generation updates a
  running summary alongside characters/locations/events, in the same
  LLM call, and that summary feeds back into the next chapter's prompt.
- **Character Cards** (Tab 3) and **Story Timeline** (Tab 4) are UI views
  over the same memory data — no separate backend logic, since the
  memory module already tracks everything both views need.
- Retrieval quality depends on chunk size/overlap settings and the
  embedding model's fit to the story's genre/style — these are exposed
  as adjustable settings in the sidebar rather than hardcoded.
- This project prioritizes a working, well-scoped core pipeline over
  implementing every optional feature listed in the original project
  spec. Remaining ones are noted as future improvements below.

## Future improvements

- Relationship graph visualization (rendering `memory.json` as an actual
  node-edge graph, e.g. via `networkx` + `pyvis`, rather than the current
  text-based relationship list on Character Cards)
- Multi-chapter batch generation
- Story export as PDF
- Genre/style selection presets
- Swapping in the QLoRA-fine-tuned model from the companion project as
  the generator backend, combining both projects' techniques
