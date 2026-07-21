import faulthandler
faulthandler.enable()
import torch
import streamlit as st
import pipeline as P

st.set_page_config(
    page_title="SDSO Musical Analyst Chatbot",
    page_icon="🎶",
    layout="centered",
)

# Small, low-risk cosmetic trims. 
# The palette lives at ".streamlit/config.toml"
st.markdown(
    """
    <style>
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
      header {visibility: hidden;}
      .block-container {padding-top: 3rem; max-width: 820px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# Cache loaders (once per run, or once per session for the LLMs)
@st.cache_data(show_spinner="Loading corpus metadata...")
def get_csv():
    return P.load_music_csv()          # (csv_text, lookup, context_lookup)


@st.cache_resource(show_spinner="Loading Phi-4...")
def get_phi4():
    return P.load_phi4()               # (model, tok)


@st.cache_resource(show_spinner="Loading MERT...")
def get_mert():
    return P.load_mert()               # (mert, processor)


csv_text, lookup, context_lookup = get_csv()
phi_model, phi_tok = get_phi4()
mert, mert_proc = get_mert()


# Header and instructions
st.title("SDSO Cross-Cultural Musical Analyst")
st.caption("Please ask about the emotional or cultural significance of Lakota music from our SDSO recordings.")

# Conversation session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.info("Please ask a question:")

# Replay the visible transcript on every rerun.
for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar="🎶" if m["role"] == "assistant" else None):
        st.markdown(m["content"])

# Render a list of dicts as a Markdown table — bypasses PyArrow entirely,
# which segfaults on this box (pyarrow/numpy ABI mismatch). st.table and
# st.dataframe both go through Arrow, so neither can be used here.
def _md_table(rows: list) -> str:
    if not rows:
        return "_(none)_"
    cols = list(rows[0].keys())
    def cell(v):
        return str(v).replace("|", "\\|").replace("\n", " ")
    header = "| " + " | ".join(cell(c) for c in cols) + " |"
    sep    = "| " + " | ".join("---"    for _ in cols) + " |"
    body   = "\n".join(
        "| " + " | ".join(cell(r.get(c, "")) for c in cols) + " |"
        for r in rows
    )
    return "\n".join([header, sep, body])


# Run the entire pipeline for a given query.
def run_pipeline(query: str) -> str:
    try:
        with st.status("Working through the pipeline...", expanded=True) as status:
            st.write("Retrieving the 10 most relevant pieces...")
            ranked = P.similarity_llm(query, csv_text, phi_model, phi_tok)

            st.write("Extracting acoustic features + MERT embeddings...")
            results = P.query_mert(ranked, lookup, context_lookup, mert, mert_proc)

            if not results:
                status.update(label="No audio matched the retrieved titles.",
                            state="error", expanded=False)
                return "I couldn't match any audio files to the retrieved titles for that question."

            st.write("Generating the cross-cultural analysis...")
            analysis = P.query_output_llm(query, results, csv_text, phi_model, phi_tok)
            status.update(label="Done", state="complete", expanded=False)

        # Keep the UI tidy — Markdown tables, not st.table (see _md_table note above)
        with st.expander("Retrieved pieces (with accompanying similarity scores to your question)"):
            st.markdown(_md_table([{"rank": i + 1, "similarity score": s, "title": t}
                    for i, (t, s) in enumerate(ranked)]))
        with st.expander("Acoustic feature measurements"):
            st.markdown(_md_table([{"title": r["title"], **r["features"]} for r in results]))
        if len(results) > 1:
            with st.expander("MERT pairwise similarity"):
                st.text(P.mert_pairwise(results))

        return analysis

    except torch.cuda.OutOfMemoryError:
        return "OOM Error. Please try a shorter question or restart the application."


# Input box for user queries
if prompt := st.chat_input("Please ask a question regarding understanding, incorporating, or analyzing Lakota music."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant", avatar="🎶"):
        answer = run_pipeline(prompt)
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
