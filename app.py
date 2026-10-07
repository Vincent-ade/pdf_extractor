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

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 1rem;
    max-width: 1500px;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    border-right: 1px solid #e5e7eb;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}


/* PDF viewer */

.pdf-viewer {
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    background: #f5f5f5;
    padding: 10px;
    max-height: 700px;
    overflow-y: auto;
}


/* Source cards */

.source-card {
    border: 1px solid #e5e7eb;
    border-radius: 9px;
    padding: 8px 10px;
    margin-bottom: 6px;
    background: #fafafa;
}

.source-document {
    font-size: 0.82rem;
    font-weight: 600;
    line-height: 1.3;
}

.source-page {
    color: #6b7280;
    font-size: 0.76rem;
    margin-top: 2px;
}


/* Chat */

[data-testid="stChatMessage"] {
    padding-top: 0.35rem;
    padding-bottom: 0.35rem;
}


/* Buttons */

.stButton > button {
    border-radius: 8px;
}


/* Tabs */

button[data-baseweb="tab"] {
    font-weight: 600;
}


/* Dividers */

hr {
    margin-top: 0.8rem;
    margin-bottom: 0.8rem;
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
    """
    Renders one PDF page and trims the blank top margin.
    """

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

            from PIL import Image
            import io

            image = Image.open(
                io.BytesIO(
                    pixmap.tobytes("png")
                )
            ).convert("RGB")

            width, height = image.size
            pixels = image.load()

            # Find where actual page content begins.
            # Use a high threshold so light-gray/near-white
            # backgrounds are treated as blank.
            top = height

            for y in range(height):

                content_found = False

                for x in range(
                    0,
                    width,
                    3
                ):

                    r, g, b = pixels[x, y]

                    # Anything noticeably different from white
                    # counts as content.
                    if (
                        r < 252
                        or g < 252
                        or b < 252
                    ):
                        content_found = True
                        break

                if content_found:

                    top = y
                    break

            # Only crop if a blank area was actually found.
            if top > 0 and top < height:

                # Keep a tiny margin above the content.
                top = max(
                    0,
                    top - 5
                )

                image = image.crop(
                    (
                        0,
                        top,
                        width,
                        height
                    )
                )

            output = io.BytesIO()

            image.save(
                output,
                format="PNG"
            )

            return output.getvalue()

    except Exception as error:

        print(
            f"PDF rendering error: {error}"
        )

        return None


def show_pdf_preview():
    """
    Displays the PDF viewer with page navigation.
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

    # --------------------------------------------------------
    # Document selector
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Page navigation
    # --------------------------------------------------------

    previous_col, page_col, next_col = st.columns(
        [1, 2, 1]
    )

    with previous_col:

        if st.button(
            "← Previous",
            disabled=current_page <= 1,
            width="stretch",
            key="preview_previous",
        ):

            st.session_state.pdf_preview_page -= 1

            st.rerun()


    with page_col:

        st.markdown(
            f"""
            <div style="
                text-align:center;
                padding:8px 0;
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
            width="stretch",
            key="preview_next",
        ):

            st.session_state.pdf_preview_page += 1

            st.rerun()


    # --------------------------------------------------------
    # PDF page
    # --------------------------------------------------------

    image = render_pdf_page(
        document,
        current_page
    )

    if image:

        st.markdown(
            '<div class="pdf-viewer">',
            unsafe_allow_html=True,
        )

        st.image(
            image,
            width="stretch"
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    else:

        st.error(
            "Could not render this PDF page."
        )


def show_source(source, key):
    """
    Displays a compact source item.
    """

    document = source.get("document")
    page = source.get("page")

    if not document or not page:
        return

    col1, col2 = st.columns(
        [4, 1],
        vertical_alignment="center"
    )

    with col1:

        st.markdown(
            f"""<div class="source-card">
<div class="source-document">📄 {document}</div>
<div class="source-page">Page {page}</div>
</div>""",
            unsafe_allow_html=True,
        )

    with col2:

        if st.button(
            "View",
            key=key,
            width="stretch",
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
        "Local document Q&A"
    )

    st.divider()


    # ========================================================
    # LIBRARY
    # ========================================================

    st.markdown(
        "### 📚 Library"
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


    if existing_pdfs:

        selected_pdfs = st.multiselect(
            "Documents",
            existing_pdfs,
            placeholder="Choose PDFs...",
        )

    else:

        selected_pdfs = []

        st.caption(
            "No PDFs in your library yet."
        )


    # ========================================================
    # UPLOAD
    # ========================================================

    st.markdown(
        "### ➕ Add documents"
    )

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
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
            f"{len(uploaded_files)} PDF(s) added."
        )

        st.rerun()


    # ========================================================
    # PROCESS
    # ========================================================

    st.markdown(
        "### ⚙️ Workspace"
    )

    if st.session_state.chat_history:

        if st.button(
            "＋ New chat",
            width="stretch",
        ):

            st.session_state.chat_history = []

            workspace_key = (
                st.session_state.chat_workspace_key
            )

            if workspace_key:

                chat_histories[
                    workspace_key
                ] = []

                save_chat_histories(
                    chat_histories
                )

            st.rerun()

    if selected_pdfs:

        st.caption(
            f"{len(selected_pdfs)} document(s) selected"
        )

        if st.button(
            "Process documents",
            type="primary",
            width="stretch",
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


            # Set first document as preview
            if document_names:

                st.session_state.pdf_preview_document = (
                    document_names[0]
                )

                st.session_state.pdf_preview_page = 1


            # Load workspace history
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

    else:

        st.caption(
            "Select at least one PDF to begin."
        )


    # ========================================================
    # CURRENT DOCUMENTS
    # ========================================================

    if st.session_state.document_names:

        st.divider()

        st.markdown(
            "### 📖 Current workspace"
        )

        for document in (
            st.session_state.document_names
        ):

            st.caption(
                f"📄 {document}"
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
    [1.1, 0.9],
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

        if not st.session_state.chat_history:

            st.markdown("""
                <div style="
                font-size: 1.1rem;
                font-weight: 600;
                margin-bottom: 0.4rem;
                ">
                    Enter your question
                </div>
                """, unsafe_allow_html=True)

        # Existing messages
        for message_index, message in enumerate(
            st.session_state.chat_history
        ):

            if (
                message_index > 0
                and message["role"] == "user"
                and st.session_state.chat_history[
                    message_index - 1
                ]["role"] == "assistant"
            ):
                st.markdown(
                    "<div style='height: 8px'></div>",
                    unsafe_allow_html=True,
                )

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

                    recent_messages = previous_messages[-6:]

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
            with st.chat_message("assistant"):

                with st.spinner("Thinking..."):

                    try:

                        answer = generate_answer(
                            question,
                            context
                        )

                        generation_failed = False

                    except Exception as error:

                        answer = (
                            "Sorry, I couldn't generate an answer "
                            "right now. Please make sure Ollama is "
                            "running and try again."
                        )

                        generation_failed = True

                st.markdown(answer)

                if sources and not generation_failed:

                    st.markdown("**Sources**")

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
            width="stretch",
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
                width="stretch",
            )