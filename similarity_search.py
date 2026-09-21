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

def inspect_chunk(chunk):
    print("\n--- CHUNK STRUCTURE ---")

    print("Top-level keys:")
    print(list(chunk.keys()))

    for key, value in chunk.items():
        if key == "embedding":
            print(f"{key}: <embedding hidden>")
        elif isinstance(value, dict):
            print(f"{key}:")
            print(f"  Nested keys: {list(value.keys())}")
            print(f"  Values: {value}")
        else:
            print(f"{key}: {value}")

    print("-----------------------\n")