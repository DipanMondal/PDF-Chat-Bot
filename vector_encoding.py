from sentence_transformers import SentenceTransformer
import torch


class Embedding:
    def __init__(self,model_name="hkunlp/instructor-xl"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

    def get_embeddings(self, text_chunks):
        embeddings = self.model.encode(text_chunks)
        return embeddings


if __name__ == '__main__':
    texts = [
        ["Represent a sentence for retrieval", "Hugging Face provides state-of-the-art NLP models."],
        ["Represent a query for retrieval", "Best embedding models for NLP?"]
    ]
    ob = Embedding()
    embeddings = ob.get_embeddings(texts)
    print(embeddings)