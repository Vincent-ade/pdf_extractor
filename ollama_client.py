import ollama

def build_context(results):
    """
    Build context from the most relevant retrieved chunks.
    Used for focused questions.
    """

    context_parts = []
    sources = []

    for result in results:

        chunk = result["chunk"]
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
You are a helpful assistant answering questions about a PDF document.

Your task is to answer the user's question directly, clearly, and accurately
using only the provided context.

Instructions:
1. Answer the specific question first.
2. Do not simply list every point in the context.
3. Include additional points only when they directly help answer the question.
4. Explain the main idea in simple, understandable language.
5. If the question asks for all points, provide all relevant points.
6. Do not invent information that is not in the context.
7. Do not create page numbers or citations. Sources are handled separately
   by the Python application.
8. If the answer cannot be found in the context, say:
   "I couldn't find enough information in the document to answer that."

Context:
{context}

Question:
{question}

Answer:
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

def create_chunk_batches(chunks, batch_size=5):
    """
    Group document chunks into smaller batches.

    Each batch contains up to batch_size chunks.
    """

    batches = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batches.append(batch)

    return batches

def summarize_batch(batch):
    """
    Summarize a batch of document chunks.
    """

    batch_text = ""

    for chunk in batch:
        page = chunk["metadata"]["page"]
        text = chunk["text"]

        batch_text += f"\n[Page {page}]\n{text}\n"

    prompt = f"""
You are summarizing part of a PDF document.

Summarize the following text by identifying:
- The main ideas
- Important facts
- Key explanations
- Important examples

Stay faithful to the text.
Do not invent information.
Do not answer questions outside the provided text.

TEXT:
{batch_text}
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