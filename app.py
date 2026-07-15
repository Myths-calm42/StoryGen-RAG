"""
app.py — StoryGen AI
-----------------------
Streamlit interface for retrieval-augmented long-form story continuation.

Run:
    streamlit run app.py
"""

import json
import logging
import tempfile
from pathlib import Path

import streamlit as st

from utils.chunking import chunk_story
from utils.embedding import EmbeddingModel
from utils.evaluation import EvalCase, format_results_markdown, run_comparison
from utils.generator import StoryGenerator
from utils.memory import StoryMemory
from utils.pdf_loader import load_story
from utils.prompt_builder import build_prompt
from utils.retrieval import StoryVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="StoryGen AI", layout="wide")

# ---------------------------------------------------------------------------
# Cached resource loading (models are expensive to load; cache across reruns)
# ---------------------------------------------------------------------------

@st.cache_resource
def get_embedding_model():
    return EmbeddingModel()


@st.cache_resource
def get_generator(model_name: str, load_in_4bit: bool):
    return StoryGenerator(model_name=model_name, load_in_4bit=load_in_4bit)


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "memory" not in st.session_state:
    st.session_state.memory = StoryMemory()
if "story_text" not in st.session_state:
    st.session_state.story_text = ""
if "generated_chapters" not in st.session_state:
    st.session_state.generated_chapters = []

# ---------------------------------------------------------------------------
# Sidebar: model + generation settings
# ---------------------------------------------------------------------------

st.sidebar.title("Settings")

MODEL_OPTIONS = {
    "Qwen2.5-7B-Instruct (default, ungated)": "Qwen/Qwen2.5-7B-Instruct",
    "Mistral-7B-Instruct-v0.3 (ungated)": "mistralai/Mistral-7B-Instruct-v0.3",
}

model_choice_label = st.sidebar.selectbox("Generator model", list(MODEL_OPTIONS.keys()), index=0)

if MODEL_OPTIONS[model_choice_label] is None:
    model_name = st.sidebar.text_input(
        "Custom Hugging Face model ID",
        value="",
        placeholder="e.g. your-username/your-model",
    )
    if not model_name:
        st.sidebar.warning("Enter a model ID above, or pick a preset option instead.")
else:
    model_name = MODEL_OPTIONS[model_choice_label]
    st.sidebar.caption(f"Using: `{model_name}`")

load_in_4bit = st.sidebar.checkbox("Load in 4-bit (recommended on limited VRAM)", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Chunking")
chunk_size = st.sidebar.slider("Chunk size (characters)", 300, 2000, 800, step=100)
chunk_overlap = st.sidebar.slider("Chunk overlap (characters)", 0, 400, 150, step=50)

st.sidebar.markdown("---")
st.sidebar.subheader("Retrieval")
top_k = st.sidebar.slider("Top-K chunks to retrieve", 1, 10, 5)

st.sidebar.markdown("---")
st.sidebar.subheader("Generation")
max_new_tokens = st.sidebar.slider("Max new tokens", 200, 2500, 1200, step=100)
temperature = st.sidebar.slider("Temperature", 0.1, 1.5, 0.8, step=0.1)

# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

st.title("📖 StoryGen AI — Long-Form Story Continuation with RAG")

tab_upload, tab_generate, tab_characters, tab_timeline, tab_eval = st.tabs(
    ["1. Upload Story", "2. Generate Next Chapter", "3. Character Cards", "4. Story Timeline", "5. Evaluation"]
)

# --- Tab 1: Upload ---
with tab_upload:
    st.subheader("Upload your story")
    uploaded_file = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])

    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        story_text = load_story(tmp_path)
        st.session_state.story_text = story_text

        st.success(f"Loaded {uploaded_file.name} ({len(story_text)} characters)")

        # Story statistics
        word_count = len(story_text.split())
        col1, col2 = st.columns(2)
        col1.metric("Characters", f"{len(story_text):,}")
        col2.metric("Words", f"{word_count:,}")

        with st.expander("Preview story text"):
            st.text(story_text[:2000] + ("..." if len(story_text) > 2000 else ""))

        if st.button("Build vector index"):
            with st.spinner("Chunking and embedding story..."):
                chunks = chunk_story(story_text, chunk_size=chunk_size, overlap=chunk_overlap)
                embedding_model = get_embedding_model()
                store = StoryVectorStore(embedding_model)
                store.build(chunks)
                st.session_state.vector_store = store
                st.session_state.last_chunks = chunks
            st.success(f"Built vector index with {len(chunks)} chunks")

        if st.session_state.get("last_chunks"):
            with st.expander(f"View all {len(st.session_state.last_chunks)} chunks"):
                for c in st.session_state.last_chunks:
                    st.markdown(f"**Chunk {c.chunk_id}** (chars {c.start_char}-{c.end_char}, length {len(c.text)})")
                    st.text(c.text)
                    st.markdown("---")

# --- Tab 2: Generate ---
with tab_generate:
    st.subheader("Generate the next chapter")

    if st.session_state.vector_store is None:
        st.info("Upload a story and build the vector index first (Tab 1).")
    else:
        instruction = st.text_area(
            "Instruction for the next chapter",
            placeholder="Continue the story where Alice enters the forbidden forest.",
        )

        if st.button("Generate Next Chapter", type="primary"):
            if not model_name:
                st.error("Enter a valid model ID in the sidebar (or pick a preset) before generating.")
                st.stop()

            with st.spinner("Loading generator model (first run may take a while)..."):
                generator = get_generator(model_name, load_in_4bit)

            with st.spinner("Retrieving relevant context..."):
                retrieved = st.session_state.vector_store.search(instruction, top_k=top_k)

            with st.expander("Retrieved context (debug view)"):
                for chunk, score in retrieved:
                    st.markdown(f"**Score: {score:.3f}**")
                    st.text(chunk.text[:500])
                    st.markdown("---")

            memory_summary = st.session_state.memory.as_prompt_string()
            story_summary = st.session_state.memory.get_summary()
            prompt = build_prompt(
                instruction, retrieved,
                story_summary=story_summary,
                memory_summary=memory_summary,
            )

            with st.spinner("Generating next chapter..."):
                chapter = generator.generate(
                    prompt, max_new_tokens=max_new_tokens, temperature=temperature
                )

            st.session_state.generated_chapters.append(chapter)

            with st.spinner("Updating story memory..."):
                st.session_state.memory.update_from_chapter(chapter, generator)

            st.subheader("Generated Chapter")
            st.write(chapter)

            st.download_button(
                "Download this chapter",
                data=chapter,
                file_name="generated_chapter.txt",
                mime="text/plain",
            )

        if st.session_state.generated_chapters:
            with st.expander("Story memory (characters, locations, events)"):
                st.code(json.dumps(st.session_state.memory.memory, indent=2, ensure_ascii=False), language="json")

# --- Tab 3: Character Cards ---
with tab_characters:
    st.subheader("Character Cards")

    characters = st.session_state.memory.memory.get("characters", {})

    if not characters:
        st.info("No characters tracked yet — generate at least one chapter first (Tab 2).")
    else:
        cols = st.columns(2)
        for i, (name, info) in enumerate(characters.items()):
            with cols[i % 2]:
                with st.container(border=True):
                    st.markdown(f"### {name}")
                    if info.get("role"):
                        st.caption(info["role"])
                    if info.get("age"):
                        st.write(f"**Age:** {info['age']}")
                    if info.get("traits"):
                        st.write("**Traits:** " + ", ".join(info["traits"]))
                    if info.get("relationships"):
                        st.write("**Relationships:**")
                        for other, relation in info["relationships"].items():
                            st.write(f"- {other}: {relation}")

        st.markdown("---")
        st.subheader("Locations & Objects")
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("**Locations**")
            for loc in st.session_state.memory.memory.get("locations", []):
                st.write(f"- {loc}")
        with col_b:
            st.write("**Important Objects**")
            for obj in st.session_state.memory.memory.get("objects", []):
                st.write(f"- {obj}")

# --- Tab 4: Story Timeline ---
with tab_timeline:
    st.subheader("Story Timeline")

    summary = st.session_state.memory.memory.get("summary", "")
    if summary:
        st.markdown("**Running summary so far:**")
        st.info(summary)

    events = st.session_state.memory.memory.get("events", [])

    if not events:
        st.info("No events tracked yet — generate at least one chapter first (Tab 2).")
    else:
        for i, event in enumerate(events, start=1):
            st.markdown(f"**Chapter beat {i}**")
            st.write(event)
            if i < len(events):
                st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;⬇", unsafe_allow_html=True)

    if st.session_state.generated_chapters:
        st.markdown("---")
        st.subheader("Generated Chapters (full text, in order)")
        for i, chapter in enumerate(st.session_state.generated_chapters, start=1):
            with st.expander(f"Chapter {i}"):
                st.write(chapter)

# --- Tab 5: Evaluation ---
with tab_eval:
    st.subheader("With-Retrieval vs. Without-Retrieval Comparison")

    if st.session_state.vector_store is None:
        st.info("Upload a story and build the vector index first (Tab 1).")
    else:
        st.markdown(
            "Enter 3-5 test instructions (one per line). Each will be generated "
            "both with retrieval and without, for side-by-side comparison."
        )
        test_instructions_raw = st.text_area(
            "Test instructions (one per line)",
            placeholder="Continue the story where Alice enters the forbidden forest.\n"
                        "Describe what happens when the dragon returns.",
            height=120,
        )

        if st.button("Run Evaluation"):
            test_cases = [
                EvalCase(instruction=line.strip())
                for line in test_instructions_raw.splitlines() if line.strip()
            ]
            if not test_cases:
                st.warning("Enter at least one test instruction.")
            elif not model_name:
                st.error("Enter a valid model ID in the sidebar (or pick a preset) before running evaluation.")
            else:
                with st.spinner("Loading generator model..."):
                    generator = get_generator(model_name, load_in_4bit)

                with st.spinner(f"Running {len(test_cases)} comparisons (this takes a while)..."):
                    results = run_comparison(
                        test_cases, st.session_state.vector_store, generator, top_k=top_k
                    )

                report = format_results_markdown(results)
                st.markdown(report)

                st.download_button(
                    "Download evaluation report",
                    data=report,
                    file_name="evaluation_report.md",
                    mime="text/markdown",
                )
