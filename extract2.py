# VERSION 2

import pymupdf


def extract_pages(pdf_path):
    """
    Extract text from a PDF while preserving page numbers.
    Returns a list of dictionaries containing page number and text.
    """
    try:
        doc = pymupdf.open(pdf_path)
        pages = []

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text().strip()

            if text:
                pages.append({
                    "page": page_num,
                    "text": text
                })

        doc.close()
        return pages

    except FileNotFoundError:
        print(f"Error: The file '{pdf_path}' could not be found.")
        return None


def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """
    Splits text into overlapping chunks while preserving word boundaries.
    """

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:

        end = min(start + chunk_size, text_length)

        # Try to end at a word boundary
        if end < text_length:
            last_space = text.rfind(" ", start, end)

            if last_space > start:
                end = last_space

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # Stop when we've reached the end
        if end >= text_length:
            break

        # Move forward while maintaining overlap
        start = end - chunk_overlap

        # Safety check: start MUST move forward
        if start <= 0:
            start = end

    return chunks


def create_document_chunks(pages, chunk_size=1000, chunk_overlap=200):
    """
    Creates chunks from every page while preserving
    page number and chunk metadata.
    """

    document_chunks = []

    for page in pages:

        page_number = page["page"]
        page_text = page["text"]

        chunks = chunk_text(
            page_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        for chunk_number, chunk in enumerate(chunks, start=1):

            document_chunks.append({
                "text": chunk,
                "page": page_number,
                "chunk_id": f"page_{page_number}_chunk_{chunk_number}"
            })

    return document_chunks


if __name__ == "__main__":

    pdf_filename = "sample.pdf"

    print("--- Extracting PDF ---")

    pages = extract_pages(pdf_filename)

    if pages:

        print(f"Extracted {len(pages)} pages.")

        print("\n--- Creating Chunks ---")

        document_chunks = create_document_chunks(
            pages,
            chunk_size=1000,
            chunk_overlap=200
        )

        print(f"Created {len(document_chunks)} chunks.")

        print("\n--- Sample Chunks ---")

        for chunk in document_chunks[:3]:

            print(f"\nChunk ID: {chunk['chunk_id']}")
            print(f"Page: {chunk['page']}")
            print("-" * 50)
            print(chunk["text"])
            print("-" * 50)