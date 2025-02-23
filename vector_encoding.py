from InstructorEmbedding import INSTRUCTOR
from dotenv import load_dotenv
import os
import requests


load_dotenv()


class Embedding:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model_id = model_name
        self.__hf_token = os.getenv('HF_TOKEN_READ')
        self.__api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model_id}"
        self.__headers = {"Authorization": f"Bearer {self.__hf_token}"}

    def get_embeddings(self, texts):
        response = requests.post(self.__api_url, headers=self.__headers, json={"inputs": texts, "options": {"wait_for_model": True}})
        return response.json()


ob = Embedding()

if __name__ == '__main__':
    texts = [
        "Hugging Face provides state-of-the-art NLP models.",
        "Best embedding models for NLP?"
    ]
    embeddings = ob.get_embeddings(texts)
    print(embeddings)
