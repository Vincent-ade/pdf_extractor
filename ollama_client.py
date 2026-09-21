import ollama

def build_document_context(chunks):
    context_parts = []
    sources = []

    # Arrange chunks in their original document order
    sorted_chunks = sorted(
        chunks,
        key=lambda chunk: (
            chunk["metadata"]["page"],
            chunk["metadata"]["chunk"]
        )
    )

    for chunk in sorted_chunks:
        text = chunk["text"]
        metadata = chunk["metadata"]

        document = metadata["document"]
        page = metadata["page"]

        context_parts.append(
            f"[Document: {document} | Page: {page}]\n{text}"
        )

        sources.append({
            "document": document,
            "page": page
        })

    context = "\n\n".join(context_parts)

    return context, sources

def generate_answer(question, context):
    prompt = f"""
You are answering a question using a PDF as your source.

Follow these rules:
1. Answer the question directly.
2. Use only information from the provided context.
3. Do not list random keywords or phrases from the context.
4. Explain the answer clearly and naturally.
5. Keep the answer concise unless the question requires more detail.
6. If the context does not contain enough information to answer the question, say:
   "I couldn't find enough information in the document to answer that."
7. When using information from the context, cite its page using [Page X].

Context from the PDF:
{context}

Question:
{question}

Direct answer:
"""

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]