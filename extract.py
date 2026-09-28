import os
import re
import pickle
import pymupdf

from embedding import create_embeddings
from similarity_search import search

from ollama_client import (
    build_context,
    build_document_context,
    create_chunk_batches,
    generate_answer,
    summarize_batch,
    combine_summaries
)


def extract_pages(pdf_path):
    """Extract text from each PDF page while preserving page numbers."""

    pages = []

    with pymupdf.open(pdf_path) as doc:

        for page_number, page in enumerate(doc, start=1):

            text = page.get_text("text")

            if not text.strip():
                continue

            text = re.sub(r"\s+", " ", text).strip()

            pages.append({
                "page": page_number,
                "text": text
            })

    return pages


def chunk_text(text, chunk_size=1000, overlap=200):
    """
    Split text into overlapping chunks.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = min(start + chunk_size, text_length)

        if end < text_length:

            space_position = text.rfind(" ", start, end)

            if space_position > start:
                end = space_position

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        next_start = end - overlap

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def create_chunks(pages, document_name):
    """
    Create RAG-ready chunks while preserving metadata.
    """

    all_chunks = []

    for page in pages:

        page_number = page["page"]
        text = page["text"]

        page_chunks = chunk_text(
            text,
            chunk_size=1000,
            overlap=200
        )

        for index, chunk in enumerate(page_chunks):

            all_chunks.append({
                "id": f"{document_name}_page_{page_number}_chunk_{index}",
                "text": chunk,
                "metadata": {
                    "document": document_name,
                    "page": page_number,
                    "chunk": index
                }
            })

    return all_chunks


def is_document_wide_question(question):
    """
    Detect questions that require information
    from across the entire document.
    """

    keywords = [
        "all points",
        "all the points",
        "entire document",
        "whole document",
        "everything",
        "list all",
        "main points",
        "key points"
    ]

    question = question.lower()

    return any(
        keyword in question
        for keyword in keywords
    )


def ask_question(question, chunks):

    results = search(
        question,
        chunks,
        top_k=3
    )

    if not results:

        print(
            "\nI could not find the answer in the document."
        )

        return

    print("\nSimilarity scores:")

    for result in results:
        print(
            f"- {result['score']:.4f}"
        )

    if is_document_wide_question(question):

        print("\nMode: Document-wide Q&A")

        context, sources = build_document_context(
            chunks
        )

    else:

        print("\nMode: Focused Q&A")

        context, sources = build_context(
            results
        )

    answer = generate_answer(
        question,
        context
    )

    print("\nANSWER:")
    print(answer)

    print("\nSOURCES:")

    unique_sources = set()

    for source in sources:

        source_key = (
            source["document"],
            source["page"]
        )

        if source_key not in unique_sources:

            print(
                f"- {source['document']} — Page {source['page']}"
            )

            unique_sources.add(source_key)


def summarize_document(chunks):

    batches = create_chunk_batches(
        chunks,
        batch_size=5
    )

    print(
        f"\nDocument divided into {len(batches)} batches."
    )

    summaries = []

    for i, batch in enumerate(
        batches,
        start=1
    ):

        print(
            f"\nProcessing batch {i} "
            f"of {len(batches)}..."
        )

        summary = summarize_batch(batch)

        summaries.append(summary)

    print("\nCombining batch summaries...")

    final_summary = combine_summaries(
        summaries
    )

    print("\n" + "=" * 50)
    print("FINAL DOCUMENT SUMMARY")
    print("=" * 50)

    print(final_summary)

    print("=" * 50)


def get_pdf_files(pdf_folder):

    files = []

    for filename in os.listdir(pdf_folder):

        if filename.lower().endswith(".pdf"):

            files.append(filename)

    return sorted(files)


def choose_pdf(pdf_folder):

    pdf_files = get_pdf_files(
        pdf_folder
    )

    if not pdf_files:

        print(
            "\nNo PDF files found in the pdfs folder."
        )

        return None

    print("\nAvailable PDFs:")

    for index, filename in enumerate(
        pdf_files,
        start=1
    ):

        print(
            f"{index}. {filename}"
        )

    while True:

        choice = input(
            "\nChoose a PDF: "
        ).strip()

        if not choice.isdigit():

            print(
                "Please enter a number."
            )

            continue

        choice = int(choice)

        if 1 <= choice <= len(pdf_files):

            return pdf_files[choice - 1]

        print(
            "Invalid choice. Please select "
            "one of the numbers shown."
        )

def get_cache_path(document_name):
    """
    Return the cache file path for a document.
    """

    cache_name = os.path.splitext(document_name)[0] + ".pkl"

    return os.path.join(
        "cache",
        cache_name
    )


def save_embeddings(chunks, pdf_path, cache_path):
    """
    Save embedded chunks and the PDF modification time.
    """

    cache_data = {
        "pdf_modified": os.path.getmtime(pdf_path),
        "chunks": chunks
    }

    with open(cache_path, "wb") as file:
        pickle.dump(cache_data, file)


def load_embeddings(pdf_path, cache_path):
    """
    Load cached embeddings if the PDF has not changed.
    """

    if not os.path.exists(cache_path):
        return None

    try:

        with open(cache_path, "rb") as file:
            cache_data = pickle.load(file)

        current_modified = os.path.getmtime(
            pdf_path
        )

        if cache_data["pdf_modified"] != current_modified:
            return None

        return cache_data["chunks"]

    except Exception:
        return None


if __name__ == "__main__":

    pdf_folder = "pdfs"

    selected_pdf = choose_pdf(
        pdf_folder
    )

    if selected_pdf is None:
        exit()

    pdf_path = os.path.join(
        pdf_folder,
        selected_pdf
    )

    document_name = selected_pdf

    print(
        f"\nSelected PDF: {document_name}"
    )

    print("\nExtracting PDF...")

    pages = extract_pages(
        pdf_path
    )

    print(
        f"Extracted {len(pages)} pages."
    )

    if not pages:

        print(
            "No text found in the PDF."
        )

        exit()

    print("\nCreating chunks...")

    chunks = create_chunks(
        pages,
        document_name
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    cache_path = get_cache_path(
        document_name
    )

    cached_chunks = load_embeddings(
        pdf_path,
        cache_path
    )

    if cached_chunks is not None:

        print("\nLoading cached embeddings...")

        chunks = cached_chunks

        print(
            "Cached embeddings loaded successfully."
        )

    else:

        print("\nCreating embeddings...")

        chunks = create_embeddings(
            chunks
        )

        save_embeddings(
            chunks,
            pdf_path,
            cache_path
        )

        print(
            "Embeddings created and cached successfully."
        )

    while True:

        print("\n" + "=" * 40)
        print("What would you like to do?")
        print("1. Ask a question")
        print("2. Summarize the document")
        print("3. Exit")
        print("=" * 40)

        choice = input(
            "\nChoose an option: "
        ).strip()

        if choice == "1":

            question = input(
                "\nEnter your question: "
            ).strip()

            if not question:

                print(
                    "\nPlease enter a question."
                )

                continue

            ask_question(
                question,
                chunks
            )

        elif choice == "2":

            print(
                "\nSummarization selected."
            )

            summarize_document(
                chunks
            )

        elif choice == "3":

            print("\nGoodbye!")

            break

        else:

            print(
                "\nInvalid choice. "
                "Please select 1, 2, or 3."
            )