from InstructorEmbedding import INSTRUCTOR

model = INSTRUCTOR('hkunlp/instructor-xl')

texts = [
    ["Represent a sentence for retrieval", "Hugging Face provides state-of-the-art NLP models."],
    ["Represent a query for retrieval", "Best embedding models for NLP?"]
]

embeddings = model.encode(texts)

# Print shape and first embedding
print(f"Embedding shape: {embeddings.shape}")
print("First Embedding:", embeddings[0])
