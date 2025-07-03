"""
FastAPI application for document indexing, retrieval, and analysis using ChromaDB and LlamaIndex.

This module provides a RAG (Retrieval-Augmented Generation) system that can:
- Upload and index document chunks from JSON files or URLs
- Perform semantic similarity search on indexed documents
- Summarize individual journals/documents
- Compare multiple research papers
- Retrieve documents by journal ID

The system uses ChromaDB for vector storage and HuggingFace embeddings for semantic search.
"""

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.ingestion import IngestionPipeline
from fastapi import Depends
from utils import get_llm_gemini

import json
import chromadb
import requests
import os

from pydantic import BaseModel, HttpUrl
from pydantic import ValidationError

from typing import Optional, List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse

# Initialize FastAPI application
app = FastAPI()

# ChromaDB collection name for storing document chunks
COLLECTION_NAME = "articles_chunk_database_new"


class SearchPayload(BaseModel):
    """
    Request payload for similarity search endpoint.

    Attributes:
        query (str): The search query text
        k (int): Number of top results to return
        min_score (float): Minimum similarity score threshold for filtering results
    """
    query: str
    k: int
    min_score: float


class InputChunk(BaseModel):
    """
    Pydantic model for validating input document chunks.

    Attributes:
        id (str): Unique identifier for the chunk
        source_doc_id (str): Identifier of the source document
        chunk_index (int): Index position of chunk within the document
        section_heading (str): Heading of the section this chunk belongs to
        journal (str): Name of the journal/publication
        publish_year (int): Year of publication
        usage_count (int): Number of times this chunk has been accessed
        attributes (List[str]): List of attributes/tags associated with the chunk
        link (HttpUrl): URL link to the original document
        text (str): The actual text content of the chunk
    """
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


def fetch_json_from_url(file_url: str) -> Dict[str, Any]:
    """
    Fetch JSON data from a given URL.

    Args:
        file_url (str): URL to fetch JSON data from

    Returns:
        Dict[str, Any]: Parsed JSON data as dictionary

    Raises:
        HTTPException: If the URL request fails or returns invalid JSON
    """
    try:
        response = requests.get(file_url)
        response.raise_for_status()  # Raises HTTPError for bad status codes
        data = response.json()  # Parses JSON response
        return data
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch JSON from URL: {e}")


@app.put("/api/upload")
def upload_chunk(
        schema_version: str = Form(...),
        file_url: Optional[str] = Form(None),
        file: Optional[str] = Form(...)
) -> JSONResponse:
    """
    Upload and index document chunks from either a URL or local file.

    This endpoint accepts JSON data containing document chunks and processes them
    for vector indexing. It supports both URL-based and local file uploads.

    Args:
        schema_version (str): Version of the data schema being used
        file_url (Optional[str]): URL to fetch JSON data from
        file (Optional[str]): Local file path containing JSON data

    Returns:
        JSONResponse: Status message indicating processing completion

    Raises:
        HTTPException: If neither or both file sources are provided, or if processing fails
    """
    # Validate that exactly one file source is provided
    if (not file_url and not file) or (file_url and file):
        raise HTTPException(status_code=400, detail="Provide exactly one of file_url or file")

    # Fetch data from URL if provided
    if file_url:
        json_data = fetch_json_from_url(file_url)

    # Load data from local file if provided
    if file:
        if not os.path.exists(file):
            raise HTTPException(status_code=400, detail="File not found on server")

        try:
            with open(file, "r", encoding="utf-8") as f:
                json_data = json.load(f)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON file")

    def create_unique_doc_id(item: Dict[str, Any]) -> str:
        """
        Create a unique document ID based on source document ID and chunk index.

        Args:
            item (Dict[str, Any]): Document chunk data

        Returns:
            str: Unique document identifier
        """
        return f"{item['source_doc_id']}_{item['chunk_index']}"

    # Process JSON data into Document objects
    documents = []
    for item in json_data:
        try:
            # Validate chunk data against schema
            chunk = InputChunk(**item)
        except ValidationError as e:
            print(e)  # Log validation errors but continue processing

        # Create Document object for indexing
        doc = Document(
            text=item["text"],
            id_=create_unique_doc_id(item),
            metadata={
                "id": item["id"],
                "chunk_index": item["chunk_index"],
                "section_heading": item["section_heading"],
                "journal": item["journal"],
                "publish_year": item["publish_year"],
                "usage_count": item["usage_count"],
                "attributes": ", ".join(item["attributes"]),
                "source_doc_id": item["source_doc_id"],
                "url_link": item['link']
            }
        )
        documents.append(doc)

    # Initialize embedding model
    embed_model = HuggingFaceEmbedding(model_name="BAAI/llm-embedder", device="cuda")

    # Upload documents to vector database
    chroma_index = upload_db_with_deduplication(documents, embed_model)

    print("upload_done")

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={"message": f"Processing file from URL {file_url} with schema {schema_version}"}
    )


def upload_db_with_deduplication(documents: List[Document], embed_model: HuggingFaceEmbedding) -> VectorStoreIndex:
    """
    Upload documents to ChromaDB with deduplication support.

    This function creates an ingestion pipeline that processes documents,
    generates embeddings, and stores them in a ChromaDB collection.
    It handles both new collections and existing ones.

    Args:
        documents (List[Document]): List of Document objects to index
        embed_model (HuggingFaceEmbedding): Embedding model for generating vectors

    Returns:
        VectorStoreIndex: Index object for querying the vector store
    """
    # Initialize ChromaDB client and collection
    db = chromadb.PersistentClient(path="./storage/chroma")
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Create ingestion pipeline with deduplication
    pipeline = IngestionPipeline(
        transformations=[
            SimpleNodeParser(),  # Parse documents into nodes
            embed_model,  # Generate embeddings
        ],
        vector_store=vector_store,
        docstore=SimpleDocumentStore()
    )

    print(chroma_collection.count())

    # Handle new vs existing collections
    if chroma_collection.count() == 0:
        # New collection: run pipeline directly
        pipeline.run(documents=documents, show_progress=True)
        pipeline.persist(persist_dir="./data")
    else:
        # Existing collection: load existing data first
        pipeline.load(persist_dir="./data")
        pipeline.run(documents=documents, show_progress=True)
        pipeline.persist(persist_dir="./data")

    # Create and return vector index
    chroma_index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
    return chroma_index


@app.post("/api/similarity_search")
def search(query_dict: SearchPayload) -> Dict[str, Any]:
    """
    Perform semantic similarity search on indexed documents.

    This endpoint searches for documents similar to the provided query
    using vector similarity and returns filtered results above a threshold.

    Args:
        query_dict (SearchPayload): Search parameters including query, k, and min_score

    Returns:
        Dict[str, Any]: Dictionary containing result texts and filtered results
    """
    # Initialize embedding model
    embed_model = HuggingFaceEmbedding(model_name="BAAI/llm-embedder", device="cuda")

    # Connect to ChromaDB
    db = chromadb.PersistentClient(path="./storage/chroma")
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    chroma_index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)

    # Create retriever with specified parameters
    retriever = VectorIndexRetriever(
        index=chroma_index,
        similarity_top_k=query_dict.k,
        alpha=None,
        doc_ids=None,
    )

    # Perform search and filter results
    results = retriever.retrieve(query_dict.query)
    filtered_results = [r for r in results if r.score >= query_dict.min_score]
    result_test_list = [result.node.text for result in filtered_results]

    return {"result_test_list": result_test_list, "filtered_results": filtered_results}


@app.get("/api/{journal_id}")
def get_journal(journal_name: str) -> Dict[str, Any]:
    """
    Retrieve all chunks for a specific journal by its ID.

    Args:
        journal_name (str): Name/ID of the journal to retrieve

    Returns:
        Dict[str, Any]: Dictionary containing all chunks for the specified journal
    """
    # Connect to ChromaDB and query by source document ID
    db = chromadb.PersistentClient(path="./storage/chroma")
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    results = chroma_collection.get(where={"source_doc_id": journal_name})

    return results


class SummaryRequest(BaseModel):
    """
    Request payload for journal summarization endpoint.

    Attributes:
        journal (str): Name/ID of the journal to summarize
    """
    journal: str


def get_llm():
    """
    Dependency injection function for LLM instance.

    Returns:
        LLM instance from utils module
    """
    from utils import get_llm_gemini
    return get_llm_gemini()


@app.post("/api/summary")
def summarize_journal(request: SummaryRequest, llm=Depends(get_llm)) -> Dict[str, str]:
    """
    Generate a summary of a journal using an LLM.

    This endpoint retrieves all chunks for a specified journal,
    concatenates them, and generates a summary using an LLM.

    Args:
        request (SummaryRequest): Contains the journal name to summarize
        llm: LLM instance injected via dependency injection

    Returns:
        Dict[str, str]: Dictionary containing the summary and status message
    """
    # Retrieve journal chunks
    chunks = get_journal(request.journal)
    full_text = " ".join(chunks['documents'])

    # Handle empty content
    if not full_text.strip():
        return {
            "summary": "No content to summarize.",
            "message": f"No content found for journal: {request.journal}"
        }

    # Create summarization prompt
    prompt = f"""
    You are a scientific research assistant. Summarize the following journal content:

    {full_text}

    Summary:
    """

    # Generate summary using LLM
    response = llm.complete(prompt)
    return {"summary": response.text, "message": "Success"}


class CompareRequest(BaseModel):
    """
    Request payload for paper comparison endpoint.

    Attributes:
        doc_id_1 (str): ID of the first document to compare
        doc_id_2 (str): ID of the second document to compare
    """
    doc_id_1: str
    doc_id_2: str


@app.post("/api/compare_papers")
def compare_papers(request: CompareRequest, llm=Depends(get_llm)) -> Dict[str, str]:
    """
    Compare two research papers using an LLM.

    This endpoint retrieves content from two specified documents,
    concatenates their chunks, and generates a structured comparison.

    Args:
        request (CompareRequest): Contains the IDs of documents to compare
        llm: LLM instance injected via dependency injection

    Returns:
        Dict[str, str]: Dictionary containing the comparison analysis and status message
    """
    # Retrieve document chunks for both papers
    chunks = get_journal(request.doc_id_1)
    full_text_1 = " ".join(chunks['documents'])

    chunks = get_journal(request.doc_id_2)
    full_text_2 = " ".join(chunks['documents'])

    # Handle missing content
    if not full_text_1.strip() or not full_text_2.strip():
        return {
            "comparison": "No content to compare.",
            "message": f"No content found for journals"
        }

    # Create comparison prompt
    prompt = f"""
    You are a scientific reviewer. Compare the following two research documents.

    === Paper 1 ===
    {full_text_1}

    === Paper 2 ===
    {full_text_2}

    Provide a structured comparison including:
    - Objectives
    - Methodology
    - Findings
    - Differences
    - Key strengths/limitations
    """

    # Generate comparison using LLM
    response = llm.complete(prompt)

    return {"comparison": response.text, "message": "Success"}