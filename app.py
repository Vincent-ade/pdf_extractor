import os
import json
import hashlib
import pymupdf
import streamlit as st

from extract import ( extract_pages, create_chunks, get_cache_path, load_embeddings, save_embeddings, is_document_wide_question,
)

from embedding import create_embeddings
from similarity_search import search

from ollama_client import ( build_context, build_document_context, create_chunk_batches, generate_answer, summarize_batch, combine_summaries, rewrite_followup_question,
)


PDF_FOLDER = "pdfs"
CACHE_FOLDER = "cache"

HISTORY_FILE = "chat_history.json"


def get_workspace_key(document_names):
    """Create a key for the selected PDFs."""

    documents = []

    for name in sorted(document_names):
        path = os.path.join(PDF_FOLDER, name)

        modified = (
            os.path.getmtime(path)
            if os.path.exists(path)
            else None
        )

        documents.append({
            "name": name,
            "modified": modified,
        })

    content = json.dumps(documents, sort_keys=True)

    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()


def load_all_histories():
    """Load saved conversations from disk."""

    if not os.path.exists(HISTORY_FILE):
        return {}

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as file:
            return json.load(file)

    except (json.JSONDecodeError, OSError):
        return {}


def save_all_histories(histories):
    """Save conversations to disk."""

    with open(HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump(
            histories,
            file,
            indent=2,
            ensure_ascii=False,
        )

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(CACHE_FOLDER, exist_ok=True)


st.set_page_config(
    page_title="Local PDF Assistant",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Local PDF Assistant")
st.write(
    "Upload a PDF to ask questions about its contents "
    "or generate a document summary."
)

def show_pdf_page(document_name, page_number):
    """
    Display a specific PDF page as an image.
    """

    pdf_path = os.path.join(
        PDF_FOLDER,
        document_name
    )

    if not os.path.exists(pdf_path):
        st.warning(
            f"Could not find {document_name}."
        )
        return

    try:

        with pymupdf.open(pdf_path) as doc:

            if page_number < 1 or page_number > len(doc):
                st.warning(
                    f"Page {page_number} does not exist."
                )
                return

            page = doc[page_number - 1]

            pix = page.get_pixmap(
                matrix=pymupdf.Matrix(1.5, 1.5)
            )

            image_bytes = pix.tobytes("png")

            st.image(
                image_bytes,
                caption=f"{document_name} — Page {page_number}",
                width="stretch"
            )

    except Exception as error:

        st.error(
            f"Could not display page: {error}"
        )

#
# --------------------------------------------------
# PDF LIBRARY AND PROCESSING
# --------------------------------------------------

saved_pdf_names = sorted(
    filename
    for filename in os.listdir(PDF_FOLDER)
    if filename.lower().endswith(".pdf")
)

st.subheader("Your PDF library")

selected_pdfs = st.multiselect(
    "Select PDFs already in your library",
    options=saved_pdf_names,
    help="You can select multiple PDFs to use together.",
)

uploaded_files = st.file_uploader(
    "Or upload new PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    key="pdf_library_upload",
)

if st.button("Process selected PDFs", type="primary"):

    # Map document names to their local file paths.
    files_to_process = {}

    for filename in selected_pdfs:
        files_to_process[filename] = os.path.join(
            PDF_FOLDER,
            filename,
        )

    # Save newly uploaded files without rewriting
    # files whose contents have not changed.
    for uploaded_file in uploaded_files or []:

        document_name = os.path.basename(
            uploaded_file.name
        )

        pdf_path = os.path.join(
            PDF_FOLDER,
            document_name,
        )

        new_bytes = uploaded_file.getvalue()

        existing_bytes = None

        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as file:
                existing_bytes = file.read()

        if existing_bytes != new_bytes:
            with open(pdf_path, "wb") as file:
                file.write(new_bytes)

        files_to_process[document_name] = pdf_path

    if not files_to_process:

        st.warning(
            "Select at least one saved PDF or upload a new PDF."
        )

    else:

        all_chunks = []
        processed_names = []

        progress = st.progress(0)
        status = st.empty()

        try:

            with st.spinner("Processing your PDFs..."):

                total_files = len(files_to_process)

                for index, (document_name, pdf_path) in enumerate(
                    files_to_process.items(),
                    start=1,
                ):

                    status.write(
                        f"Processing {document_name} "
                        f"({index}/{total_files})..."
                    )

                    cache_path = get_cache_path(
                        document_name
                    )

                    # Reuse cached embeddings when valid.
                    chunks = load_embeddings(
                        pdf_path,
                        cache_path,
                    )

                    if chunks is None:

                        pages = extract_pages(pdf_path)

                        if not pages:
                            st.warning(
                                f"No readable text found in "
                                f"{document_name}. Skipping it."
                            )
                            progress.progress(
                                index / total_files
                            )
                            continue

                        chunks = create_chunks(
                            pages,
                            document_name,
                        )

                        if not chunks:
                            st.warning(
                                f"No text chunks created for "
                                f"{document_name}. Skipping it."
                            )
                            progress.progress(
                                index / total_files
                            )
                            continue

                        chunks = create_embeddings(chunks)

                        save_embeddings(
                            chunks,
                            pdf_path,
                            cache_path,
                        )

                    all_chunks.extend(chunks)
                    processed_names.append(document_name)

                    progress.progress(
                        index / total_files
                    )

            progress.empty()
            status.empty()

            if not all_chunks:

                st.error(
                    "No readable text was found in the selected PDFs."
                )

            else:

                st.session_state["chunks"] = all_chunks
                st.session_state["document_names"] = processed_names
                st.session_state["summary"] = None
                st.session_state["answer"] = None
                st.session_state["sources"] = []
                # st.session_state["chat_history"] = []

                st.success(
                    f"Workspace ready: {len(processed_names)} PDF(s), "
                    f"{len(all_chunks)} chunks."
                )

        except Exception as error:

            progress.empty()
            status.empty()

            st.error(
                f"Could not process PDFs: {error}"
            )


# --------------------------------------------------
# DOCUMENT WORKSPACE
# --------------------------------------------------

if "chunks" in st.session_state:

    # chunks = st.session_state["chunks"]
    # document_name = st.session_state["document_name"]

    chunks = st.session_state["chunks"]
    document_names = st.session_state["document_names"]

    workspace_key = get_workspace_key(document_names)

    if st.session_state.get("chat_workspace_key") != workspace_key:

        histories = load_all_histories()

        st.session_state["chat_history"] = histories.get(
            workspace_key,
            [],
        )

        st.session_state["chat_workspace_key"] = workspace_key

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    st.divider()

    st.subheader("Document workspace")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("PDFs loaded", len(document_names))

    with col2:
        st.metric("Text chunks", len(chunks))

    st.write("Documents in this workspace:")

    for document_name in document_names:
        st.write(f"- {document_name}")

    question_tab, summary_tab = st.tabs(
        ["💬 Ask a question", "📝 Summarize"]
    )

    # --------------------------------------------------
    # QUESTION ANSWERING
    # --------------------------------------------------
    
    with question_tab:

        st.subheader("Ask your PDFs")

        # Display saved conversation history.
        for item in st.session_state["chat_history"]:

            with st.chat_message("user"):
                st.write(item["question"])

            with st.chat_message("assistant"):
                st.markdown(item["answer"])

                if item.get("sources"):
                    with st.expander("Sources"):
                        for source in item["sources"]:
                            st.write(
                                f"- {source['document']} — "
                                f"Page {source['page']}"
                            )

        # Clear history for this workspace only.
        if st.session_state["chat_history"]:

            if st.button("Clear Q&A history"):

                histories = load_all_histories()
                histories.pop(workspace_key, None)

                save_all_histories(histories)

                st.session_state["chat_history"] = []

                st.rerun()

        question = st.chat_input(
            "Ask a question about your PDFs..."
        )

        
        if question:

            # Save recent history before adding the new exchange.
            previous_history = st.session_state[
                "chat_history"
            ][-4:]

            try:

                with st.spinner("Finding an answer..."):

                    # Understand follow-up references.
                    search_question = rewrite_followup_question(
                        question,
                        previous_history,
                    )

                    results = search(
                        search_question,
                        chunks,
                        top_k=3,
                    )

                    if not results:

                        answer = (
                            "I could not find the answer "
                            "in the document."
                        )

                        sources = []

                    else:

                        if is_document_wide_question(
                            search_question
                        ):

                            context, sources = (
                                build_document_context(chunks)
                            )

                        else:

                            context, sources = build_context(
                                results
                            )

                        # Include recent exchanges as context.
                        conversation_context = ""

                        for item in previous_history:
                            conversation_context += (
                                f"\nPrevious question: "
                                f"{item['question']}\n"
                                f"Previous answer: "
                                f"{item['answer']}\n"
                            )

                        if conversation_context:
                            context += (
                                "\n\nRECENT CONVERSATION:\n"
                                + conversation_context
                            )

                        answer = generate_answer(
                            question,
                            context,
                        )

                st.session_state["chat_history"].append({
                    "question": question,
                    "answer": answer,
                    "sources": sources,
                })

                # Save history to disk.
                histories = load_all_histories()

                histories[workspace_key] = (
                    st.session_state["chat_history"]
                )

                save_all_histories(histories)

                st.rerun()

            except Exception as error:

                st.error(
                    f"Could not answer question: {error}"
                )

    # --------------------------------------------------
    # DOCUMENT SUMMARIZATION
    # --------------------------------------------------

    with summary_tab:

        st.subheader("Summarize your PDF")

        st.write(
            "Generate one combined summary from all "
            "the document's batches."
        )

        if st.button("Generate summary", type="primary"):

            try:

                batches = create_chunk_batches(
                    chunks,
                    batch_size=5,
                )

                summaries = []

                progress = st.progress(0)
                status = st.empty()

                with st.spinner("Summarizing document..."):

                    for index, batch in enumerate(
                        batches,
                        start=1,
                    ):

                        status.write(
                            f"Processing batch {index} "
                            f"of {len(batches)}..."
                        )

                        batch_summary = summarize_batch(batch)
                        summaries.append(batch_summary)

                        progress.progress(
                            index / len(batches)
                        )

                    status.write("Combining summaries...")

                    final_summary = combine_summaries(
                        summaries
                    )

                    st.session_state["summary"] = final_summary

                    status.empty()
                    progress.empty()

            except Exception as error:

                st.error(f"Could not summarize PDF: {error}")

        if st.session_state.get("summary"):

            st.markdown("### Document Summary")

            st.markdown(st.session_state["summary"])

            st.download_button(
                "Download summary as TXT",
                data=st.session_state["summary"],
                file_name="document_summary.txt",
                mime="text/plain",
            )

else:

    st.info(
        "Upload a PDF and click 'Process PDF' "
        "to get started."
    )