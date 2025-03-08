from dotenv import load_dotenv
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import io
import torch
import os


load_dotenv()


class Embedding:
    def __init__(self):
        self.text_model_id = "sentence-transformers/all-MiniLM-L6-v2"
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.__hf_token = os.getenv('HF_TOKEN_READ')
        self.__api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.text_model_id}"
        self.__headers = {"Authorization": f"Bearer {self.__hf_token}"}

    def get_text_embeddings(self, text_data: list):
        texts = []
        for each in text_data:
            texts.append(each['content'])
        inputs = self.clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            text_embeddings = self.clip_model.get_text_features(**inputs)
        return text_embeddings.tolist()

    def get_image_embeddings(self, images:list):
        image_bytes = [img['image_data'] for img in images]
        # print(image_bytes[0])
        embeddings = []
        for image_byte in image_bytes:
            image = Image.open(io.BytesIO(image_byte)).convert("RGB")
            inputs = self.clip_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                image_embedding = self.clip_model.get_image_features(**inputs)
            embeddings.append(image_embedding.squeeze().tolist())
        return embeddings

    def get_query_embedding(self, query_text:str):
        if isinstance(query_text, str):  # If a single string is provided, convert it to a list
            text = [query_text]
        inputs = self.clip_processor(text=text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            text_embeddings = self.clip_model.get_text_features(**inputs)
        return text_embeddings.tolist()


if __name__ == '__main__':
    ob = Embedding()
    texts = [
        {'page': 1, 'content': "Hugging Face provides state-of-the-art NLP models."},
        {'page': 2, 'content': "Best embedding models for NLP?"}
    ]
    embeddings = ob.get_text_embeddings(texts)
    print(len(embeddings),len(embeddings[0]))
