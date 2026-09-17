import ollama


def build_context(results):
    context = ""

    for result in results:
        chunk = result["chunk"]

        context += chunk["text"] + "\n\n"

    return context


def generate_answer(question, context):
    prompt = f"""
Use the following context from a PDF to answer the question.

Context:
{context}

Question:
{question}

Answer using only the information provided in the context.
If the answer cannot be found in the context, say you don't know.
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