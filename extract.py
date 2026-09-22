import re
import pymupdf
from embedding import create_embeddings
from similarity_search import search

from similarity_search import search
from ollama_client import build_document_context, generate_answer


def extract_pages(pdf_path):
    """Extract text from each PDF page while preserving page numbers."""

    pages = []

    with pymupdf.open(pdf_path) as doc:

        for page_number, page in enumerate(doc, start=1):

            text = page.get_text("text")

            if not text.strip():
                continue

            # Clean excessive whitespace
            text = re.sub(r"\s+", " ", text).strip()

            pages.append({
                "page": page_number,
                "text": text
            })

    return pages


def chunk_text(text, chunk_size=1000, overlap=200):
    """
    Split text into overlapping chunks.

    chunk_size and overlap are measured in characters.
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

        # Proposed end of this chunk
        end = min(start + chunk_size, text_length)

        # If we're not at the end, try to break at a space
        if end < text_length:

            space_position = text.rfind(" ", start, end)

            if space_position > start:
                end = space_position

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # We've reached the end of the document
        if end >= text_length:
            break

        # Calculate next starting position
        next_start = end - overlap

        # Make absolutely sure we're moving forward
        if next_start <= start:
            next_start = end

        start = next_start

    return chunks

#testing


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


if __name__ == "__main__":

    pdf_path = "sample-doc.pdf"
    document_name = "sample-doc.pdf"

    print("Extracting PDF...")

    pages = extract_pages(pdf_path)

    print(f"Extracted {len(pages)} pages.")

    if not pages:
        print("No text found in the PDF.")
        exit()

    print("\nCreating chunks...")

    chunks = create_chunks(
        pages,
        document_name
    )

    print(f"Created {len(chunks)} chunks.")

    print("\nCreating embeddings...")

    chunks = create_embeddings(chunks)

    print("Embeddings created successfully.")

    question = input("\nAsk a question about the PDF: ")

    # Search the PDF (kept for future question-routing)
    results = search(
        question,
        chunks,
        top_k=10
    )

    # Build context from the entire document
    context, sources = build_document_context(chunks)

    # Generate answer using Ollama
    answer = generate_answer(question, context)

    # Display answer once
    print("\nAnswer:")
    print(answer)

    # Display sources once, with duplicates removed
    print("\nSources:")

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