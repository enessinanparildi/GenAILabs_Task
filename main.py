from typing import Union

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.core.schema import Document
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.chroma import ChromaVectorStore

import json
import chromadb

from pydantic import BaseModel, HttpUrl
from pydantic import ValidationError

from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse
import requests
import os

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


def fetch_json_from_url(file_url: str):
    try:
        response = requests.get(file_url)
        response.raise_for_status()  # Raises HTTPError for bad status codes
        data = response.json()       # Parses JSON response
        return data
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch JSON from URL: {e}")

@app.put("/api/upload")
def upload_chunk(schema_version: str = Form(...), file_url: Optional[str] = Form(None),
                 file: Optional[str] = Form(...)):


    if (file_url is None and file is None) or (file_url is not None and file is not None):
        raise HTTPException(status_code=400, detail="Provide exactly one of file_url or file")

    if file_url:
        json_data = fetch_json_from_url(file_url)

    if file:
        if not os.path.exists(file):
            raise HTTPException(status_code=400, detail="File not found on server")

        try:
            with open(file, "r", encoding="utf-8") as f:
                json_data = json.load(f)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON file")



    documents = []
    for item in json_data:
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

    embed_model = HuggingFaceEmbedding(model_name="BAAI/llm-embedder", device="cuda")

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

    print("upload_done")
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={"message": f"Processing file from URL {file_url} with schema {schema_version}"}
    )


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

    #example_query = "Given the adaptability of velvet bean to various environmental conditions, including its tolerance for long dry spells and poor soils, how might its agronomic traits contribute to food security and soil fertility in semi-arid regions such as natural regions IV and V of Zimbabwe?"

    results = retriever.retrieve(query_dict.query)
    filtered_results = [r for r in results if r.score >= query_dict.min_score]
    result_list = [result.text for result in filtered_results]

    return result_list, filtered_results


@app.get("/api/{journal_id}")
def get_journal(journal_name: str):

    db = chromadb.PersistentClient(path="./storage/chroma")
    collection_name = "articles_chunk_database"
    chroma_collection = db.get_or_create_collection(collection_name)
    results = chroma_collection.get(where={"journal": journal_name})

    return results