# StoryGen AI — Long-Form Story Continuation with Retrieval-Augmented Generation

A retrieval-augmented generation (RAG) system for long-form story continuation.
Upload a story (TXT/PDF), and generate a coherent next chapter that stays
consistent with previously established characters, locations, events, and
relationships — using semantic retrieval over prior chapters plus a
lightweight persistent memory module, rather than relying on an LLM's raw
context window alone.

Built as a second Gen AI portfolio project, alongside a companion QLoRA
fine-tuning project ([QLoRA-Finetune-Story-continuation-llm](https://github.com/Myths-calm42/QLoRA-Finetune-Story-continuation-llm)).
Where that project demonstrates parameter-efficient fine-tuning, this one
demonstrates retrieval-augmented generation, vector search, and lightweight
structured memory — together covering the core techniques named in Gen AI
Research job descriptions (LLMs, fine-tuning, knowledge graphs, and
long-form content generation).

## Why this project

Demonstrates, concretely and honestly:
- Retrieval-augmented generation: chunking, embedding, and semantic search
  over a story's own prior content
- A lightweight persistent memory system (characters, locations, objects,
  events, and a running summary), functioning as a knowledge-graph-style
  analog for narrative consistency
- Prompt engineering: structured system + memory + retrieved-context +
  instruction assembly
- Qualitative evaluation methodology (with-retrieval vs. without-retrieval
  comparison)
- A full working application (Streamlit), not just a script

## Architecture

```
StoryGenAI/
  app.py                   Streamlit interface (5 tabs)
  requirements.txt
  utils/
    pdf_loader.py            TXT/PDF -> plain text
    chunking.py                paragraph-aware semantic chunking
    embedding.py                 Sentence Transformers embeddings
    retrieval.py                  FAISS vector store + top-K search
    prompt_builder.py            assembles system + memory + context + instruction
    generator.py                   instruct LLM generation (Qwen2.5 / Mistral / Llama-3.1 / Gemma)
    memory.py                       persistent character/location/event/relationship memory + running summary
    evaluation.py                    with-retrieval vs. without-retrieval comparison
  data/                       sample story files
  models/                      (local model cache, if used)
  vector_db/                    saved FAISS indexes
  generated/                     saved generated chapters
  assets/                         screenshots
```

## How it works

1. **Upload** a story (TXT or PDF) — parsed to plain text.
2. **Chunk** the story into overlapping, paragraph-respecting segments.
3. **Embed** each chunk with `BAAI/bge-small-en-v1.5` (Sentence Transformers).
4. **Index** the embeddings in a FAISS vector store.
5. Given an **instruction** for the next chapter, **retrieve** the most
   relevant prior chunks via semantic similarity search.
6. **Build a prompt** combining: a system prompt, the story's running
   summary, tracked memory (characters/locations/events/relationships),
   the retrieved context, and the instruction.
7. **Generate** the next chapter with a selectable instruct LLM
   (Qwen2.5-7B-Instruct and Mistral-7B-Instruct  supported via a dropdown).
8. **Update memory**: a second LLM call extracts new/changed characters,
   locations, objects, events, and an updated running summary from the
   generated chapter, in a single pass.

## Features

- **Upload & chunk** any TXT/PDF story, with adjustable chunk size/overlap
- **Retrieval-grounded generation** with a visible debug view of exactly
  which passages were retrieved for a given instruction
- **Character Cards** — auto-generated cards for every tracked character
  (role, traits, relationships), plus tracked locations and objects
- **Story Timeline** — running story summary and a chronological event
  list, alongside the full text of every generated chapter
- **Configurable generation** — model selection (dropdown), temperature,
  top-p, repetition penalty, max tokens, all adjustable from the sidebar
- **Evaluation tab** — runs the same instruction with and without
  retrieval, side by side, for direct qualitative comparison

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then in the browser UI:
1. Upload a story file (a sample is included at `data/sample_story_full.txt`)
   and click "Build vector index" (Tab 1)
2. Enter an instruction and click "Generate Next Chapter" (Tab 2)
3. View tracked characters, relationships, locations, and objects as
   cards (Tab 3)
4. View the story's running summary and event timeline (Tab 4)
5. Run the with/without-retrieval comparison (Tab 5)

## Live demo

A live demo can be launched via Google Colab (free T4 GPU) using
`pyngrok` to expose the Streamlit app publicly. See `colab_launch.ipynb`
(or run the equivalent cells manually — clone this repo in Colab, install
dependencies, then launch Streamlit with an ngrok tunnel).

**Note on availability:** as with the companion QLoRA project, this link
is session-based, not permanently hosted — free hosting tiers that could
run a 7B model persistently (Hugging Face Spaces' ZeroGPU) require a PRO
subscription, and the free CPU-only tier cannot hold a 7B model in memory
with 4-bit quantization (which requires CUDA). A permanent free deployment
would require re-architecting the generator to use a smaller, GGUF-
quantized model via `llama-cpp-python` instead of `transformers` +
`bitsandbytes` — noted below as a future improvement rather than done
here, given project timeline constraints.

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
- **Character Cards** and **Story Timeline** are UI views over the same
  memory data — no separate backend logic, since the memory module
  already tracks everything both views need.
- No fine-tuning is used in this project — generation relies entirely on
  retrieval and prompt engineering with off-the-shelf instruct models.
  This is a deliberate contrast with the companion QLoRA project.
- Retrieval quality depends on chunk size/overlap settings and the
  embedding model's fit to the story's genre/style — these are exposed
  as adjustable settings in the sidebar rather than hardcoded.
- Evaluation (Tab 5) is a **qualitative** comparison meant to be read and
  annotated by hand, not an automated metric — appropriate for RAG
  consistency, which is harder to reduce to a single number than, say,
  perplexity.

## Future improvements

- Relationship graph visualization (rendering `memory.json` as an actual
  node-edge graph, e.g. via `networkx` + `pyvis`)
- Multi-chapter batch generation
- Story export as PDF
- Genre/style selection presets
- Permanent free deployment via GGUF-quantized smaller model +
  `llama-cpp-python`, deployable on Hugging Face Spaces' free CPU tier
- Swapping in the QLoRA-fine-tuned model from the companion project as
  an additional generator option, combining both projects' techniques

## Known gotchas (for anyone reproducing this)

- **Missing `bitsandbytes`:** required for 4-bit model loading but easy
  to omit from a fresh environment; without it (or without a CUDA-enabled
  torch build), generation silently falls back to CPU or fails.
- **CPU-only torch:** `pip install torch` alone installs a CPU-only build
  on Windows by default. Verify with
  `python -c "import torch; print(torch.cuda.is_available())"` before
  running the generator — reinstall via
  `pip install torch --index-url https://download.pytorch.org/whl/cu121`
  (matching your CUDA version) if it prints `False`.
- **Streamlit's file watcher and `transformers`:** Streamlit's dev-mode
  file watcher scans every submodule in installed packages, including
  unrelated vision models inside `transformers`, and prints harmless
  `ModuleNotFoundError: No module named 'torchvision'` warnings if
  `torchvision` isn't installed. These don't affect the app; run with
  `streamlit run app.py --server.fileWatcherType none` to suppress them.
- **Empty files after editor paste:** double-check that files actually
  contain content after creating them in an editor (`Length: 0` in a
  directory listing means nothing was saved) before debugging import
  errors that are actually just empty modules.
- **Custom model dropdown:** the sidebar's model selector requires a
  valid, existing Hugging Face model ID; gated models (Llama-3.1, Gemma-2)
  require accepting their license on huggingface.co and running
  `hf auth login` first.
