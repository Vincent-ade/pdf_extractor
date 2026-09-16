from sentence_transformers import SentenceTransformer, util


model = SentenceTransformer("all-MiniLM-L6-v2")


def search(question, chunks, top_k=3):
    # Convert the question into an embedding
    question_embedding = model.encode(question)

    # Get the embeddings from our PDF chunks
    chunk_embeddings = [chunk["embedding"] for chunk in chunks]

    # Compare the question with every chunk
    similarities = util.cos_sim(
        question_embedding,
        chunk_embeddings
    )[0]

    # Attach the similarity score to each chunk
    results = []

    for chunk, score in zip(chunks, similarities):
        results.append({
            "chunk": chunk,
            "score": float(score)
        })

    # Highest similarity first
    results.sort(
        key=lambda result: result["score"],
        reverse=True
    )

    # Return only the best results
    return results[:top_k]