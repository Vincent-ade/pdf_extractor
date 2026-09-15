from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2")


sentences = [
    "The company made 4.2 billion naira in revenue.",
    "The business generated 4.2 billion naira.",
    "The weather is very hot today."
]


embeddings = model.encode(sentences)


for sentence, embedding in zip(sentences, embeddings):
    print("\nSentence:")
    print(sentence)

    print("\nEmbedding:")
    print(embedding)

    print("\nNumber of values:", len(embedding))