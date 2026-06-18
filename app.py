import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import streamlit as st
from core.loader import Loader
from core.vectorstore import VectorStore
from core.pipeline import Pipeline
from core.chain import RAGChain
import config

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

html, body, .stApp { background: #F7F6F2 !important; font-family: 'Inter', sans-serif; }
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stSidebar"] { display: none !important; }

.stButton > button {
    background: #3B5BDB !important; color: #FFFFFF !important;
    border: none !important; border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important; font-weight: 500 !important;
    font-size: 14px !important; padding: 10px 0 !important;
    width: 100% !important; transition: background 0.2s !important;
}
.stButton > button:hover { background: #2F4AC4 !important; }

[data-testid="stFileUploader"] {
    background: #FAFAFA !important; border: 1.5px dashed #D1D5DB !important;
    border-radius: 10px !important;
}
[data-testid="stChatInput"] {
    background: #FFFFFF !important; border: 1.5px solid #E8E6DF !important;
    border-radius: 14px !important; box-shadow: 0 2px 12px rgba(0,0,0,0.06) !important;
}
[data-testid="stChatInput"] textarea { color: #1A1A2E !important; font-size: 14px !important; }
[data-testid="stExpander"] {
    background: #FAFAFA !important; border: 1px solid #F3F2EE !important;
    border-radius: 10px !important; margin-top: 8px !important;
}
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-thumb { background: #D1D5DB; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


def init():
    if "vs" not in st.session_state:
        st.session_state.vs = VectorStore()
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = Pipeline(st.session_state.vs)
    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "messages" not in st.session_state:
        st.session_state.messages = []


def get_chain():
    if st.session_state.chain is None:
        retriever = st.session_state.vs.get_retriever()
        st.session_state.chain = RAGChain(retriever)
    return st.session_state.chain


init()
vs            = st.session_state.vs
indexed_files = vs.list_files()
chunk_count   = vs.count()


# ── Navbar ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="
    background:#FFFFFF; border-bottom:1px solid #E8E6DF;
    padding:0 40px; height:60px;
    display:flex; align-items:center; justify-content:space-between;
">
    <div style="display:flex; align-items:center; gap:10px;">
        <div style="
            width:32px; height:32px; background:#3B5BDB;
            border-radius:8px; display:flex; align-items:center;
            justify-content:center; font-size:16px;
        ">🧠</div>
        <span style="font-size:17px; font-weight:600; color:#1A1A2E; letter-spacing:-0.02em;">
            DocuMind AI
        </span>
    </div>
    <div style="display:flex; align-items:center; gap:24px; font-size:13px; color:#6B7280;">
        <span>Files: <strong style="color:#1A1A2E;">{len(indexed_files)}</strong></span>
        <span>Chunks: <strong style="color:#1A1A2E;">{chunk_count}</strong></span>
        <span style="
            background:#ECFDF5; color:#065F46;
            padding:4px 12px; border-radius:20px;
            font-size:12px; font-weight:500;
        ">✦ Fully free</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Two-column layout ─────────────────────────────────────────────────────────
left_col, right_col = st.columns([2.2, 1], gap="large")


# ══════════════════════════════════════════════
# RIGHT — Upload + stats panel
# ══════════════════════════════════════════════
with right_col:
    st.markdown("<div style='padding-top:24px;'>", unsafe_allow_html=True)

    # Panel wrapper
    st.markdown("""
    <div style="
        background:#FFFFFF; border:1px solid #E8E6DF;
        border-radius:16px; padding:24px 20px;
    ">
        <div style="font-size:11px; font-weight:600; color:#9CA3AF;
                    text-transform:uppercase; letter-spacing:0.08em; margin-bottom:14px;">
            Upload Documents
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploads = st.file_uploader(
        "PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Max 50MB each",
    )

    if uploads:
        if st.button("⚡  Index Documents"):
            prog = st.progress(0)
            result = {}
            for i, f in enumerate(uploads):
                prog.progress((i + 1) / len(uploads), text=f"Indexing {f.name}…")
                result = st.session_state.pipeline.run_uploaded(f)
            prog.empty()
            if result.get("success"):
                st.success(f"✓ {result['chunks']} chunks stored")
                st.session_state.chain = None
                st.rerun()
            else:
                st.error(result.get("error", "Indexing failed."))

    # Indexed files
    if indexed_files:
        st.markdown("""
        <div style="font-size:11px; font-weight:600; color:#9CA3AF;
                    text-transform:uppercase; letter-spacing:0.08em;
                    margin: 20px 0 10px;">
            Indexed Files
        </div>
        """, unsafe_allow_html=True)
        for f in indexed_files:
            st.markdown(f"""
            <div style="
                display:flex; align-items:center; gap:8px;
                background:#F0F4FF; border:1px solid #C7D2FE;
                border-radius:8px; padding:8px 12px; margin-bottom:6px;
                font-size:12px; color:#3B5BDB; font-weight:500;
                white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
            ">📄 {f}</div>
            """, unsafe_allow_html=True)

    # Stats
    st.markdown("""
    <div style="font-size:11px; font-weight:600; color:#9CA3AF;
                text-transform:uppercase; letter-spacing:0.08em;
                margin: 20px 0 10px;">
        Config
    </div>
    """, unsafe_allow_html=True)

    def stat_row(label, value):
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center;
                    padding:8px 0; border-bottom:1px solid #F3F2EE; font-size:13px;">
            <span style="color:#9CA3AF;">{label}</span>
            <span style="font-weight:600; color:#374151; font-size:12px;">{value}</span>
        </div>
        """, unsafe_allow_html=True)

    stat_row("LLM", config.LLM_MODEL)
    stat_row("Embeddings", "MiniLM-L6-v2")
    stat_row("Chunk size", f"{config.CHUNK_SIZE} chars")
    stat_row("Top-K", config.TOP_K)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Clear chat"):
            st.session_state.messages = []
            if st.session_state.chain:
                st.session_state.chain.clear_memory()
            st.rerun()
    with c2:
        if st.button("Clear DB"):
            vs.clear()
            st.session_state.chain = None
            st.session_state.messages = []
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# LEFT — Chat area
# ══════════════════════════════════════════════
with left_col:
    st.markdown("<div style='padding-top:24px;'>", unsafe_allow_html=True)

    # Empty state
    if not indexed_files:
        st.markdown("""
        <div style="
            background:#FFFFFF; border:1px solid #E8E6DF;
            border-radius:16px; padding:64px 40px; text-align:center;
        ">
            <div style="font-size:48px; margin-bottom:16px; opacity:0.4;">📂</div>
            <div style="
                font-family:'DM Serif Display',serif;
                font-size:22px; color:#1A1A2E; margin-bottom:10px;
            ">No documents indexed yet</div>
            <div style="font-size:14px; color:#9CA3AF; line-height:1.7;">
                Upload a PDF or TXT file on the right panel,<br>
                click <strong style="color:#6B7280;">Index Documents</strong>,<br>
                then start asking questions here.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Ready banner
        if not st.session_state.messages:
            st.markdown(f"""
            <div style="
                background:#EEF2FF; border:1px solid #C7D2FE;
                border-radius:14px; padding:18px 22px; margin-bottom:20px;
            ">
                <div style="font-size:15px; font-weight:600; color:#3730A3; margin-bottom:4px;">
                    Ready to answer
                </div>
                <div style="font-size:13px; color:#4338CA; line-height:1.6;">
                    {len(indexed_files)} file(s) indexed · {chunk_count} chunks searchable.
                    Ask anything — answers come with page citations.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Chat messages
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div style="display:flex; justify-content:flex-end; margin:8px 0;">
                    <div style="
                        background:#3B5BDB; color:#FFFFFF;
                        border-radius:18px 18px 4px 18px;
                        padding:12px 18px; max-width:74%;
                        font-size:14px; line-height:1.6;
                    ">{msg["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="display:flex; align-items:flex-start; gap:10px; margin:8px 0;">
                    <div style="
                        width:30px; height:30px; background:#EEF2FF;
                        border:1px solid #C7D2FE; border-radius:8px;
                        display:flex; align-items:center; justify-content:center;
                        font-size:15px; flex-shrink:0; margin-top:2px;
                    ">🧠</div>
                    <div style="
                        background:#FFFFFF; border:1px solid #E8E6DF;
                        border-radius:4px 18px 18px 18px;
                        padding:14px 18px; max-width:82%;
                        font-size:14px; color:#374151; line-height:1.7;
                    ">{msg["content"]}</div>
                </div>
                """, unsafe_allow_html=True)

                # Source chips
                if msg.get("sources"):
                    chips = "".join(
                        f'<span style="'
                        f'display:inline-flex; align-items:center; gap:5px;'
                        f'background:#F0F4FF; border:1px solid #C7D2FE;'
                        f'border-radius:20px; padding:4px 10px;'
                        f'font-size:11px; color:#3B5BDB; font-weight:500;'
                        f'margin:4px 4px 0 0; cursor:default;">'
                        f'📄 {s["file"]} · p.{s["page"]}</span>'
                        for s in msg["sources"]
                    )
                    st.markdown(
                        f'<div style="margin-left:40px; margin-top:4px;">{chips}</div>',
                        unsafe_allow_html=True
                    )

                    with st.expander("View source excerpts"):
                        for src in msg["sources"]:
                            st.markdown(f"""
                            <div style="
                                background:#FAFAFA;
                                border-left:3px solid #3B5BDB;
                                border-radius:0 8px 8px 0;
                                padding:10px 14px; margin-bottom:10px;
                                font-size:13px; line-height:1.6;
                            ">
                                <div style="font-weight:600; color:#374151; margin-bottom:4px;">
                                    {src['file']} — Page {src['page']}
                                </div>
                                <div style="color:#6B7280; font-style:italic;">
                                    "{src['snippet']}"
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

        # Chat input
        question = st.chat_input("Ask anything about your documents…")
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.spinner("Searching documents…"):
                result = get_chain().ask(question)
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
            })
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)