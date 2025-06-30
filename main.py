from typing import Union

import json
from fastapi import FastAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import json
import chromadb
from llama_index.core.schema import Document

embed_model = HuggingFaceEmbedding(model_name="BAAI/llm-embedder", device="cuda")

app = FastAPI()


@app.put("/api/upload")
def upload_chunk(chunk_json):


    with open('D:\GenAILabs_Task\Sample_chunks.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    embed_model = HuggingFaceEmbedding(model_name="BAAI/llm-embedder", device="cuda")
    # Load your documents

    with open('D:\GenAILabs_Task\Sample_chunks.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = [Document(text=elem["text"]) for elem in data]

    db = chromadb.PersistentClient(path="./storage/chroma")
    collection_name = "articles_chunk_database"
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    if chroma_collection.count() == 0:
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        chroma_index = VectorStoreIndex.from_documents(documents, storage_context=storage_context,
                                                       embed_model=embed_model)
    else:
        chroma_index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
        chroma_index.insert_nodes(documents)

    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}