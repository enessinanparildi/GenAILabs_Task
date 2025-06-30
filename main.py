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
from llama_index.core.retrievers import VectorIndexRetriever


embed_model = HuggingFaceEmbedding(model_name="BAAI/llm-embedder", device="cuda")

app = FastAPI()

from pydantic import BaseModel

class SearchPayload(BaseModel):
    query: str
    k: int
    min_score: float

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


@app.post("/api/similarity_search")
def search(query_dict: SearchPayload):

    embed_model = HuggingFaceEmbedding(model_name="BAAI/llm-embedder", device="cuda")


    db = chromadb.PersistentClient(path="./storage/chroma")
    collection_name = "articles_chunk_database"
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    chroma_index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)

    retriever = VectorIndexRetriever(
        index=chroma_index,
        similarity_top_k=query_dict.k,
        alpha=None,
        doc_ids=None,
    )

    example_query = "Given the adaptability of velvet bean to various environmental conditions, including its tolerance for long dry spells and poor soils, how might its agronomic traits contribute to food security and soil fertility in semi-arid regions such as natural regions IV and V of Zimbabwe?"

    results = retriever.retrieve(query_dict.query)
    filtered_results = [r for r in results if r.score >= query_dict.min_score]
    result_list = [result.text for result in filtered_results]

    return result_list