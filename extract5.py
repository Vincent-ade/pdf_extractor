import re
import pymupdf
from embedding import create_embeddings
from similarity_search import search
from ollama_client import (
    build_context,
    build_document_context,
    create_chunk_batches,
    generate_answer,
    summarize_batch
)


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
    """Split text into overlapping chunks (character-based)."""
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
    """Create RAG-ready chunks while preserving metadata."""
    all_chunks = []
    for page in pages:
        page_number = page["page"]
        text = page["text"]

        page_chunks = chunk_text(text, chunk_size=1000, overlap=200)

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
    """Detect questions that require information from across the entire document."""
    keywords = [
        "all points", "all the points", "entire document", "whole document",
        "summarize", "summary", "everything", "list all", "main points", "key points"
    ]
    question = question.lower()
    return any(keyword in question for keyword in keywords)


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
    chunks = create_chunks(pages, document_name)
    print(f"Created {len(chunks)} chunks.")

    print("\nCreating embeddings...")
    chunks = create_embeddings(chunks)
    print("Embeddings created successfully.")

    while True:
        print("\nWhat would you like to do?")
        print("1. Ask a question")
        print("2. Summarize the document")
        print("3. Exit")

        choice = input("\nChoose an option: ")

        if choice == "1":
            question = input("\nEnter your question: ")

            # Select the appropriate retrieval strategy based on intent routing
            if is_document_wide_question(question):
                print("\nMode: Document-wide Q&A")
                context, sources = build_document_context(chunks)
            else:
                print("\nMode: Focused Q&A")
                results = search(question, chunks, top_k=3)
                context, sources = build_context(results)

            # Generate and print the answer
            answer = generate_answer(question, context)
            print("\nANSWER:")
            print(answer)

            # Display unique sources cleanly
            print("\nSOURCES:")
            unique_sources = set()
            for source in sources:
                # Handle dictionary formats or standard strings returned from ollama_client
                if isinstance(source, dict):
                    source_key = (source.get("document"), source.get("page"))
                    if source_key not in unique_sources:
                        print(f"- {source_key[0]} — Page {source_key[1]}")
                        unique_sources.add(source_key)
                else:
                    if source not in unique_sources:
                        print(source)
                        unique_sources.add(source)

        elif choice == "2":
            print("\nSummarization selected.")
            batches = create_chunk_batches(chunks, batch_size=5)
            print(f"Total batches to summarize: {len(batches)}")

            for i, batch in enumerate(batches, start=1):
                print(f"\nSummarizing batch {i} of {len(batches)}...")
                summary = summarize_batch(batch)
                print(f"\nBATCH {i} SUMMARY:")
                print(summary)

        elif choice == "3":
            print("\nGoodbye!")
            break

        else:
            print("\nInvalid choice. Please select 1, 2, or 3.")
