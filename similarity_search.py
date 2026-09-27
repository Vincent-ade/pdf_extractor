from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")


def search(question, chunks, top_k=3, threshold=0.35):
    """
    Find the most relevant chunks for a question.

    If the best similarity score is below the threshold,
    the question is considered unrelated to the document.
    """

    question_embedding = model.encode(question)

    chunk_embeddings = [
        chunk["embedding"]
        for chunk in chunks
    ]

    similarities = util.cos_sim(
        question_embedding,
        chunk_embeddings
    )[0]

    results = []

    for i, score in enumerate(similarities):

        results.append({
            "chunk": chunks[i],
            "score": float(score)
        })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    results = results[:top_k]

    if not results:
        return []

    # Check the best matching result
    best_score = results[0]["score"]

    if best_score < threshold:
        return []

    return results