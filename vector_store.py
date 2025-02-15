import chromadb

db_path = r"./chroma_db"
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="text_embeddings")


def store_data(chunks,embeddings):
    for i, text in enumerate(chunks):
        collection.add(
            ids=[str(i)],  # Unique ID for each entry
            embeddings=[embeddings[i]],  # Embedding vector
            metadatas=[{"text": text}]  # Store text as metadata
        )


def get_query_match(query_embedding,n_results=5):
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results  # Retrieve top 5 matches(default)
    )
