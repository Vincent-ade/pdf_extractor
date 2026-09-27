from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")


def search(question, chunks, top_k=3):
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

    return results[:top_k]