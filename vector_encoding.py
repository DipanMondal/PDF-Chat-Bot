from InstructorEmbedding import INSTRUCTOR


class Embedding:
    def __init__(self,model_name="hkunlp/instructor-xl"):
        self.model = INSTRUCTOR(model_name)

    def get_embeddings(self, text_chunks):
        embeddings = self.model.encode(text_chunks)
        return embeddings


ob = Embedding()

if __name__ == '__main__':
    texts = [
        ["Represent a sentence for retrieval", "Hugging Face provides state-of-the-art NLP models."],
        ["Represent a query for retrieval", "Best embedding models for NLP?"]
    ]
    embeddings = ob.get_embeddings(texts)
    print(embeddings)
