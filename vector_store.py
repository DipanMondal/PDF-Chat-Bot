import chromadb
import base64
from PIL import Image
import io

db_path = r"./chroma_db"
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="data_embeddings")


def encode(image_byte):
    image_base64 = base64.b64encode(image_byte).decode("utf-8")
    return image_base64


def decode(image_base64):
    image_bytes = base64.b64decode(image_base64)  # Decode Base64 to bytes
    return image_bytes


def get_image_from_bytes(img_bytes):
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    # Convert image to byte stream (PNG format)
    img_io = io.BytesIO()
    image.save(img_io, format="png")
    img_io.seek(0)

    return img_io.getvalue()


def store_text_data(chunks,embeddings,name):
    for i, text in enumerate(chunks):
        collection.add(
            ids=[name+'text'+str(i)],  # Unique ID for each entry
            embeddings=[embeddings[i]],  # Embedding vector
            metadatas=[{"type": 'text', "page": text['page'], "content": text['content']}]  # Store text as metadata
        )


def store_image_data(image_data,embeddings,name):
    for i, img in enumerate(image_data):
        collection.add(
            ids=[name+'img'+str(i)],
            embeddings=[embeddings[i]],
            metadatas=[{"type": "image", "page": img['page'], "content": encode(img['image_data'])}]
        )


def get_query_match(query_embedding,n_results=6):
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results  # Retrieve top 5 matches(default)
    )
    return results['metadatas'][0]


def get_relavant_data(query_emb:str):
    results = get_query_match(query_emb)
    texts = []
    images = []
    for data in results:
        if data['type'] == 'text':
            texts.append(data['content'])
        else:
            images.append(decode(data['content']))
    return texts, images


def delete_data(name):
    data = collection.get()  # Retrieve all data (includes IDs by default)
    if "ids" in data:
        ids_to_delete = [id_ for id_ in data["ids"] if id_.startswith(name)]
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)

    data = collection.get()  # Retrieve all data (includes IDs by default)
    if "ids" in data:
        ids_to_delete = [id_ for id_ in data["ids"] if id_.startswith(name)]
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
