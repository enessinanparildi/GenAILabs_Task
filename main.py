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
from pydantic import BaseModel, HttpUrl
from pydantic import ValidationError
from typing import List


embed_model = HuggingFaceEmbedding(model_name="BAAI/llm-embedder", device="cuda")

app = FastAPI()


class SearchPayload(BaseModel):
    query: str
    k: int
    min_score: float

class InputChunk(BaseModel):
    id: str
    source_doc_id: str
    chunk_index: int
    section_heading: str
    journal: str
    publish_year: int
    usage_count: int
    attributes: List[str]
    link: HttpUrl
    text: str


@app.put("/api/upload")
def upload_chunk(json_file_url: str):


    with open('D:\GenAILabs_Task\Sample_chunks.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(json_file_url, 'r', encoding='utf-8') as f:
        data = json.load(f)


    embed_model = HuggingFaceEmbedding(model_name="BAAI/llm-embedder", device="cuda")
    # Load your documents

    with open('D:\GenAILabs_Task\Sample_chunks.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = []
    for item in data:
        try:
            chunk = InputChunk(**item)
        except ValidationError as e:
            print(e)

        doc = Document(
            text=item["text"],
            metadata={
                "id": item["id"],
                "chunk_index": item["chunk_index"],
                "section_heading": item["section_heading"],
                "journal": item["journal"],
                "publish_year": item["publish_year"],
                "usage_count": item["usage_count"],
                "attributes": ", ".join(item["attributes"]),
                "source_doc_id": item["source_doc_id"]
            }
        )
        documents.append(doc)

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

@app.get("/api/{journal_id}")
async def get_journal(journal_id: int):

    return {"journal_id": journal_id, "status": "Fetched successfully"}