import os
import json
import hashlib
import streamlit as st
import pymupdf

from extract import (
    extract_pages,
    create_chunks,
    is_document_wide_question,
)

from embedding import create_embeddings
from similarity_search import search

from ollama_client import (
    build_context,
    build_document_context,
    create_chunk_batches,
    generate_answer,
    summarize_batch,
    combine_summaries,
    rewrite_followup_question,
)


# ============================================================
# CONFIG
# ============================================================

PDF_FOLDER = "pdfs"
CACHE_FOLDER = "cache"
HISTORY_FILE = "chat_history.json"

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(CACHE_FOLDER, exist_ok=True)

st.set_page_config(
    page_title="Local PDF Assistant",
    page_icon="📄",
    layout="wide",
)

st.markdown(
    """
    <style>

    /* Main app */

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
        max-width: 1500px;
    }


    /* Header */

    .app-header {
        margin-bottom: 1.5rem;
    }

    .app-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .app-subtitle {
        color: #6b7280;
        font-size: 0.95rem;
    }


    /* Workspace */

    .workspace-panel {
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1rem;
        background: white;
        min-height: 600px;
    }


    /* Section headers */

    .section-title {
        font-size: 1.05rem;
        font-weight: 650;
        margin-bottom: 0.75rem;
    }


    /* PDF page */

    .pdf-page-container {
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 0.5rem;
        background: #f8fafc;
    }


    /* Sources */

    .source-card {
        border: 1px solid #e5e7eb;
        border-radius: 9px;
        padding: 0.55rem 0.7rem;
        margin-bottom: 0.45rem;
        background: #fafafa;
    }

    .source-document {
        font-size: 0.82rem;
        font-weight: 600;
    }

    .source-page {
        color: #6b7280;
        font-size: 0.78rem;
    }


    /* Chat */

    [data-testid="stChatMessage"] {
        padding-top: 0.4rem;
        padding-bottom: 0.4rem;
    }


    /* Buttons */

    .stButton > button {
        border-radius: 8px;
    }


    /* Sidebar */

    section[data-testid="stSidebar"] {
        border-right: 1px solid #e5e7eb;
    }


    /* Divider */

    hr {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "chunks": [],
    "document_names": [],
    "summary": "",
    "chat_history": [],
    "chat_workspace_key": "",
    "pdf_preview_document": None,
    "pdf_preview_page": 1,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HELPERS
# ============================================================

def get_workspace_key(document_names):
    joined = "|".join(sorted(document_names))

    return hashlib.md5(
        joined.encode("utf-8")
    ).hexdigest()


def load_chat_histories():
    if not os.path.exists(HISTORY_FILE):
        return {}

    try:
        with open(
            HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as file:
            return json.load(file)

    except (json.JSONDecodeError, OSError):
        return {}


def save_chat_histories(histories):
    with open(
        HISTORY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            histories,
            file,
            indent=4,
            ensure_ascii=False
        )


def get_cache_path(pdf_path):
    filename = os.path.basename(pdf_path)
    modified_time = os.path.getmtime(pdf_path)

    cache_key = hashlib.md5(
        f"{filename}_{modified_time}".encode("utf-8")
    ).hexdigest()

    return os.path.join(
        CACHE_FOLDER,
        f"{cache_key}.pkl"
    )


def load_embeddings_cache(cache_path):
    import pickle

    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, "rb") as file:
            return pickle.load(file)

    except Exception:
        return None


def save_embeddings_cache(cache_path, chunks):
    import pickle

    with open(cache_path, "wb") as file:
        pickle.dump(chunks, file)


def process_pdf(pdf_path):
    cache_path = get_cache_path(pdf_path)

    cached_chunks = load_embeddings_cache(
        cache_path
    )

    if cached_chunks is not None:
        return cached_chunks

    pages = extract_pages(pdf_path)

    document_name = os.path.basename(pdf_path)

    chunks = create_chunks(
        pages,
        document_name
    )

    chunks = create_embeddings(chunks)

    save_embeddings_cache(
        cache_path,
        chunks
    )

    return chunks


def get_pdf_page_count(document_name):
    pdf_path = os.path.join(
        PDF_FOLDER,
        document_name
    )

    if not os.path.exists(pdf_path):
        return 0

    try:
        with pymupdf.open(pdf_path) as doc:
            return len(doc)

    except Exception:
        return 0


def render_pdf_page(document_name, page_number):
    pdf_path = os.path.join(
        PDF_FOLDER,
        document_name
    )

    if not os.path.exists(pdf_path):
        return None

    try:
        with pymupdf.open(pdf_path) as doc:

            if not doc:
                return None

            page_number = max(
                1,
                min(
                    page_number,
                    len(doc)
                )
            )

            page = doc[page_number - 1]

            matrix = pymupdf.Matrix(
                1.5,
                1.5
            )

            pixmap = page.get_pixmap(
                matrix=matrix,
                alpha=False
            )

            return pixmap.tobytes("png")

    except Exception:
        return None


def show_pdf_preview():
    """
    Displays the PDF viewer.
    """

    documents = st.session_state.document_names

    if not documents:
        return

    # Make sure a valid document is selected
    if (
        st.session_state.pdf_preview_document
        not in documents
    ):
        st.session_state.pdf_preview_document = (
            documents[0]
        )
        st.session_state.pdf_preview_page = 1

    document = st.session_state.pdf_preview_document

    page_count = get_pdf_page_count(
        document
    )

    if page_count == 0:
        st.error("Unable to open this PDF.")
        return

    current_page = st.session_state.pdf_preview_page

    current_page = max(
        1,
        min(
            current_page,
            page_count
        )
    )

    st.session_state.pdf_preview_page = current_page

    # Document selector
    selected_document = st.selectbox(
        "Document",
        documents,
        index=documents.index(document),
        key="preview_document_selector",
    )

    if selected_document != document:

        st.session_state.pdf_preview_document = (
            selected_document
        )

        st.session_state.pdf_preview_page = 1

        st.rerun()

    # Page controls
    previous_col, page_col, next_col = st.columns(
        [1, 2, 1]
    )

    with previous_col:

        if st.button(
            "← Previous",
            disabled=current_page <= 1,
            use_container_width=True,
            key="preview_previous",
        ):

            st.session_state.pdf_preview_page -= 1

            st.rerun()

    with page_col:

        st.markdown(
            f"""
            <div style="
                text-align:center;
                padding:8px;
                font-weight:600;
            ">
                Page {current_page} of {page_count}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with next_col:

        if st.button(
            "Next →",
            disabled=current_page >= page_count,
            use_container_width=True,
            key="preview_next",
        ):

            st.session_state.pdf_preview_page += 1

            st.rerun()

    # PDF page
    image = render_pdf_page(
        document,
        current_page
    )

    if image:

        st.image(
            image,
            use_container_width=True
        )

    else:

        st.error(
            "Could not render this PDF page."
        )


def show_source(source, key):
    """
    Displays a compact source card.
    """

    document = source["document"]
    page = source["page"]

    col1, col2 = st.columns(
        [4, 1],
        vertical_alignment="center"
    )

    with col1:

        st.markdown(
            f"""
            <div class="source-card">
                <div class="source-document">
                    📄 {document}
                </div>

                <div class="source-page">
                    Page {page}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        if st.button(
            "View",
            key=key,
            use_container_width=True,
        ):

            st.session_state.pdf_preview_document = (
                document
            )

            st.session_state.pdf_preview_page = (
                page
            )

            st.rerun()


# ============================================================
# LOAD CHAT HISTORY
# ============================================================

chat_histories = load_chat_histories()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 📄 PDF Assistant"
    )

    st.caption(
        "Local document Q&A and summarization"
    )

    st.divider()

    st.markdown(
        "### PDF Library"
    )

    existing_pdfs = sorted(
        [
            filename
            for filename in os.listdir(
                PDF_FOLDER
            )
            if filename.lower().endswith(".pdf")
        ]
    )

    selected_pdfs = st.multiselect(
        "Select documents",
        existing_pdfs,
    )

    uploaded_files = st.file_uploader(
        "Add PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:

        for uploaded_file in uploaded_files:

            save_path = os.path.join(
                PDF_FOLDER,
                uploaded_file.name
            )

            with open(
                save_path,
                "wb"
            ) as file:

                file.write(
                    uploaded_file.getbuffer()
                )

        st.success(
            "PDF uploaded."
        )

        st.rerun()

    st.divider()

    if selected_pdfs:

        if st.button(
            "Process documents",
            type="primary",
            use_container_width=True,
        ):

            all_chunks = []
            document_names = []

            progress = st.progress(0)

            total = len(selected_pdfs)

            for index, document_name in enumerate(
                selected_pdfs
            ):

                pdf_path = os.path.join(
                    PDF_FOLDER,
                    document_name
                )

                try:

                    chunks = process_pdf(
                        pdf_path
                    )

                    all_chunks.extend(
                        chunks
                    )

                    document_names.append(
                        document_name
                    )

                except Exception as error:

                    st.error(
                        f"Could not process "
                        f"{document_name}: {error}"
                    )

                progress.progress(
                    (index + 1) / total
                )

            st.session_state.chunks = (
                all_chunks
            )

            st.session_state.document_names = (
                document_names
            )

            if document_names:

                st.session_state.pdf_preview_document = (
                    document_names[0]
                )

                st.session_state.pdf_preview_page = 1

            workspace_key = get_workspace_key(
                document_names
            )

            st.session_state.chat_workspace_key = (
                workspace_key
            )

            st.session_state.chat_history = (
                chat_histories.get(
                    workspace_key,
                    []
                )
            )

            st.session_state.summary = ""

            st.success(
                f"{len(document_names)} "
                f"document(s) ready."
            )


# ============================================================
# HEADER
# ============================================================

st.title("📄 Local PDF Assistant")

st.caption(
    "Ask questions, explore your documents, and generate summaries with local AI."
)


# ============================================================
# NO DOCUMENTS
# ============================================================

if not st.session_state.chunks:

    st.info(
        "Select a PDF from the sidebar and "
        "click **Process documents**."
    )

    st.stop()


# ============================================================
# MAIN WORKSPACE
# ============================================================

pdf_column, chat_column = st.columns(
    [1.05, 0.95],
    gap="large"
)


# ============================================================
# PDF COLUMN
# ============================================================

with pdf_column:

    st.markdown(
        """
        <div class="section-title">
            📖 PDF Preview
        </div>
        """,
        unsafe_allow_html=True,
    )

    show_pdf_preview()


# ============================================================
# CHAT COLUMN
# ============================================================

with chat_column:

    st.markdown(
        """
        <div class="section-title">
            💬 Ask your documents
        </div>
        """,
        unsafe_allow_html=True,
    )

    qa_tab, summary_tab = st.tabs(
        [
            "Chat",
            "Summary"
        ]
    )


    # ========================================================
    # CHAT
    # ========================================================

    with qa_tab:

        # Existing messages
        for message_index, message in enumerate(
            st.session_state.chat_history
        ):

            with st.chat_message(
                message["role"]
            ):

                st.markdown(
                    message["content"]
                )

                if (
                    message["role"] == "assistant"
                    and message.get("sources")
                ):

                    st.markdown(
                        "**Sources**"
                    )

                    for source_index, source in enumerate(
                        message["sources"]
                    ):

                        show_source(
                            source,
                            (
                                f"history_"
                                f"{message_index}_"
                                f"{source_index}"
                            )
                        )


        question = st.chat_input(
            "Ask something about your PDF..."
        )


        if question:

            # Add user message
            st.session_state.chat_history.append(
                {
                    "role": "user",
                    "content": question
                }
            )


            with st.chat_message("user"):

                st.markdown(
                    question
                )


            # Follow-up rewriting
            search_question = question

            previous_messages = (
                st.session_state.chat_history[:-1]
            )

            if previous_messages:

                try:

                    search_question = (
                        rewrite_followup_question(
                            question,
                            previous_messages
                        )
                    )

                except Exception:

                    search_question = question


            # Search
            if is_document_wide_question(
                search_question
            ):

                context, sources = (
                    build_document_context(
                        st.session_state.chunks
                    )
                )

            else:

                results = search(
                    search_question,
                    st.session_state.chunks,
                    top_k=3
                )

                context, sources = build_context(
                    results
                )


            # Generate answer
            with st.chat_message(
                "assistant"
            ):

                with st.spinner(
                    "Thinking..."
                ):

                    try:

                        answer = generate_answer(
                            question,
                            context
                        )

                    except Exception as error:

                        answer = (
                            f"Error generating answer: "
                            f"{error}"
                        )

                st.markdown(
                    answer
                )


                # Sources
                if sources:

                    st.markdown(
                        "**Sources**"
                    )

                    for source_index, source in enumerate(
                        sources
                    ):

                        show_source(
                            source,
                            (
                                f"current_"
                                f"{source_index}_"
                                f"{hash(question)}"
                            )
                        )


            # Save assistant message
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                }
            )


            # Save persistent history
            workspace_key = (
                st.session_state.chat_workspace_key
            )

            if workspace_key:

                chat_histories[
                    workspace_key
                ] = st.session_state.chat_history

                save_chat_histories(
                    chat_histories
                )

            st.rerun()


    # ========================================================
    # SUMMARY
    # ========================================================

    with summary_tab:

        st.markdown(
            "Generate a summary of the selected documents."
        )

        if st.button(
            "Generate Summary",
            type="primary",
            use_container_width=True,
            key="generate_summary",
        ):

            all_summaries = []

            batches = create_chunk_batches(
                st.session_state.chunks,
                batch_size=5
            )

            progress = st.progress(0)

            total_batches = len(batches)

            for index, batch in enumerate(
                batches
            ):

                try:

                    batch_summary = summarize_batch(
                        batch
                    )

                    all_summaries.append(
                        batch_summary
                    )

                except Exception as error:

                    st.error(
                        f"Error summarizing batch "
                        f"{index + 1}: {error}"
                    )

                progress.progress(
                    (index + 1) / total_batches
                )


            if all_summaries:

                with st.spinner(
                    "Combining summaries..."
                ):

                    try:

                        st.session_state.summary = (
                            combine_summaries(
                                all_summaries
                            )
                        )

                    except Exception as error:

                        st.error(
                            f"Error combining summaries: "
                            f"{error}"
                        )


        if st.session_state.summary:

            st.markdown(
                st.session_state.summary
            )

            st.download_button(
                "⬇️ Download Summary",
                data=st.session_state.summary,
                file_name="pdf_summary.txt",
                mime="text/plain",
                use_container_width=True,
            )