from sentence_transformers import SentenceTransformer, util


model = SentenceTransformer("all-MiniLM-L6-v2")


sentences = [
    "Supervised learning uses labeled training data.",
    "The weather is very hot today.",
    "A supervised learning algorithm learns from examples."
]

question = "What is supervised learning?"


sentence_embeddings = model.encode(sentences)
question_embedding = model.encode(question)


similarities = util.cos_sim(
    question_embedding,
    sentence_embeddings
)


for sentence, score in zip(sentences, similarities[0]):
    print(f"{score:.4f} → {sentence}")