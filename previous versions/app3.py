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


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_workspace_key(document_names):
    """
    Creates a unique key for the currently selected PDFs.
    """

    joined = "|".join(sorted(document_names))

    return hashlib.md5(
        joined.encode("utf-8")
    ).hexdigest()


def load_chat_histories():
    """
    Loads saved chat history from chat_history.json.
    """

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
    """
    Saves chat history.
    """

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
    """
    Creates a cache filename based on the PDF name and
    modification time.
    """

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
    """
    Loads embeddings from cache.
    """

    import pickle

    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, "rb") as file:
            return pickle.load(file)

    except Exception:
        return None


def save_embeddings_cache(cache_path, chunks):
    """
    Saves chunks and embeddings to cache.
    """

    import pickle

    with open(cache_path, "wb") as file:
        pickle.dump(chunks, file)


def process_pdf(pdf_path):
    """
    Extracts, chunks and embeds a PDF.
    Uses cached embeddings when available.
    """

    cache_path = get_cache_path(pdf_path)

    cached_chunks = load_embeddings_cache(cache_path)

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
    """
    Returns the number of pages in a PDF.
    """

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
    """
    Renders one PDF page as an image.
    """

    pdf_path = os.path.join(
        PDF_FOLDER,
        document_name
    )

    if not os.path.exists(pdf_path):
        return None

    try:
        with pymupdf.open(pdf_path) as doc:

            if page_number < 1:
                page_number = 1

            if page_number > len(doc):
                page_number = len(doc)

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


def set_preview_page(document_name, page_number):
    """
    Changes the PDF preview document and page.
    """

    st.session_state.pdf_preview_document = document_name
    st.session_state.pdf_preview_page = page_number


# ============================================================
# SESSION STATE
# ============================================================

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "document_names" not in st.session_state:
    st.session_state.document_names = []

if "summary" not in st.session_state:
    st.session_state.summary = ""

if "answer" not in st.session_state:
    st.session_state.answer = ""

if "sources" not in st.session_state:
    st.session_state.sources = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chat_workspace_key" not in st.session_state:
    st.session_state.chat_workspace_key = ""

if "pdf_preview_document" not in st.session_state:
    st.session_state.pdf_preview_document = None

if "pdf_preview_page" not in st.session_state:
    st.session_state.pdf_preview_page = 1


# ============================================================
# LOAD SAVED CHAT HISTORIES
# ============================================================

chat_histories = load_chat_histories()


# ============================================================
# HEADER
# ============================================================

st.title("📄 Local PDF Assistant")

st.caption(
    "Ask questions, summarize documents, and inspect source pages."
)


# ============================================================
# PDF LIBRARY
# ============================================================

st.sidebar.header("📚 PDF Library")

existing_pdfs = sorted(
    [
        filename
        for filename in os.listdir(PDF_FOLDER)
        if filename.lower().endswith(".pdf")
    ]
)


selected_pdfs = st.sidebar.multiselect(
    "Select PDFs",
    existing_pdfs,
)


uploaded_files = st.sidebar.file_uploader(
    "Add new PDFs",
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

    st.sidebar.success(
        "PDF uploaded successfully."
    )

    st.rerun()


# ============================================================
# PROCESS DOCUMENTS
# ============================================================

if selected_pdfs:

    if st.sidebar.button(
        "Process selected PDFs",
        type="primary"
    ):

        all_chunks = []
        document_names = []

        progress = st.sidebar.progress(0)

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

                all_chunks.extend(chunks)

                document_names.append(
                    document_name
                )

            except Exception as error:

                st.sidebar.error(
                    f"Could not process {document_name}: {error}"
                )

            progress.progress(
                (index + 1) / total
            )

        st.session_state.chunks = all_chunks

        st.session_state.document_names = document_names

        # Reset preview
        if document_names:

            st.session_state.pdf_preview_document = (
                document_names[0]
            )

            st.session_state.pdf_preview_page = 1

        # Load the correct saved chat history
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

        st.sidebar.success(
            f"{len(document_names)} PDF(s) ready."
        )


# ============================================================
# CHECK WHETHER DOCUMENTS ARE LOADED
# ============================================================

if not st.session_state.chunks:

    st.info(
        "Select one or more PDFs from the sidebar, "
        "then click **Process selected PDFs**."
    )

    st.stop()


# ============================================================
# PDF PREVIEW
# ============================================================

st.divider()

st.subheader("📖 PDF Preview")

preview_documents = st.session_state.document_names


# Make sure selected preview document still exists
if (
    st.session_state.pdf_preview_document
    not in preview_documents
):

    st.session_state.pdf_preview_document = (
        preview_documents[0]
    )

    st.session_state.pdf_preview_page = 1


preview_document = st.selectbox(
    "Document",
    preview_documents,
    index=preview_documents.index(
        st.session_state.pdf_preview_document
    ),
)


if preview_document != st.session_state.pdf_preview_document:

    st.session_state.pdf_preview_document = (
        preview_document
    )

    st.session_state.pdf_preview_page = 1


page_count = get_pdf_page_count(
    st.session_state.pdf_preview_document
)

current_page = st.session_state.pdf_preview_page


# Keep page within valid range
if page_count > 0:

    current_page = max(
        1,
        min(
            current_page,
            page_count
        )
    )

    st.session_state.pdf_preview_page = current_page


# Navigation
preview_col1, preview_col2, preview_col3 = st.columns(
    [1, 2, 1]
)


with preview_col1:

    if st.button(
        "← Previous",
        disabled=current_page <= 1,
        use_container_width=True
    ):

        st.session_state.pdf_preview_page -= 1

        st.rerun()


with preview_col2:

    st.markdown(
        f"<div style='text-align:center; padding-top:7px;'>"
        f"<b>Page {current_page} of {page_count}</b>"
        f"</div>",
        unsafe_allow_html=True
    )


with preview_col3:

    if st.button(
        "Next →",
        disabled=current_page >= page_count,
        use_container_width=True
    ):

        st.session_state.pdf_preview_page += 1

        st.rerun()


# Render current page
page_image = render_pdf_page(
    st.session_state.pdf_preview_document,
    current_page
)


if page_image:

    st.image(
        page_image,
        use_container_width=True
    )

else:

    st.error(
        "Could not display this PDF page."
    )


# ============================================================
# MAIN TABS
# ============================================================

st.divider()

qa_tab, summary_tab = st.tabs(
    [
        "💬 Q&A",
        "📝 Summarize"
    ]
)


# ============================================================
# Q&A
# ============================================================

with qa_tab:

    st.subheader("Ask your documents")

    # Display previous conversation
    for message in st.session_state.chat_history:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

            # Display sources for assistant messages
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

                    document = source["document"]
                    page = source["page"]

                    st.write(
                        f"📄 {document} — Page {page}"
                    )

                    if st.button(
                        f"View page {page}",
                        key=(
                            f"history_source_"
                            f"{len(st.session_state.chat_history)}_"
                            f"{source_index}"
                        )
                    ):

                        set_preview_page(
                            document,
                            page
                        )

                        st.rerun()


    question = st.chat_input(
        "Ask a question about your PDF..."
    )


    if question:

        # Add user question
        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question
            }
        )


        with st.chat_message("user"):

            st.markdown(question)


        # ----------------------------------------------------
        # Rewrite follow-up questions
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Search
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Generate answer
        # ----------------------------------------------------

        with st.chat_message("assistant"):

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
                        f"Error generating answer: {error}"
                    )


            st.markdown(answer)


            # Sources
            if sources:

                st.markdown(
                    "**Sources**"
                )

                for source_index, source in enumerate(
                    sources
                ):

                    document = source["document"]
                    page = source["page"]

                    source_col1, source_col2 = st.columns(
                        [4, 1]
                    )

                    with source_col1:

                        st.write(
                            f"📄 {document} — Page {page}"
                        )

                    with source_col2:

                        if st.button(
                            "View page",
                            key=(
                                f"current_source_"
                                f"{source_index}_"
                                f"{question}"
                            )
                        ):

                            set_preview_page(
                                document,
                                page
                            )

                            st.rerun()


        # ----------------------------------------------------
        # Save conversation
        # ----------------------------------------------------

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources
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


# ============================================================
# SUMMARIZE
# ============================================================

with summary_tab:

    st.subheader("Document Summary")

    st.write(
        "Generate a summary of the selected documents."
    )


    if st.button(
        "Generate Summary",
        type="primary"
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

                    final_summary = combine_summaries(
                        all_summaries
                    )

                    st.session_state.summary = (
                        final_summary
                    )

                except Exception as error:

                    st.error(
                        f"Error combining summaries: {error}"
                    )


    # Display saved summary
    if st.session_state.summary:

        st.markdown(
            st.session_state.summary
        )


        st.download_button(
            "⬇️ Download Summary",
            data=st.session_state.summary,
            file_name="pdf_summary.txt",
            mime="text/plain"
        )