from sentence_transformers import SentenceTransformer
import torch

# Load the instructor-xl model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("hkunlp/instructor-xl",device=device)

texts = [
    ["Represent a sentence for retrieval", "Hugging Face provides state-of-the-art NLP models."],
    ["Represent a query for retrieval", "Best embedding models for NLP?"]
]

embeddings = model.encode(texts)

# Print shape and first embedding
print(f"Embedding shape: {embeddings.shape}")
print("First Embedding:", embeddings[0])
