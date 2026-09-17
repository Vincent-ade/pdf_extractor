import ollama


def build_context(results):
    context = ""

    for result in results:
        chunk = result["chunk"]
        context += chunk["text"] + "\n\n"

    return context


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