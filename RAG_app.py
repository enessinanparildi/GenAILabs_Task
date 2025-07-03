"""
RAG Application Runner and Testing Suite

This module serves as the main application runner and testing suite for the RAG system.
It provides functionality to:
- Start the FastAPI server
- Test document upload functionality
- Perform similarity searches
- Run chatbot queries with citation tracking
- Test summarization and comparison endpoints
- Manage usage count tracking for retrieved documents

The module demonstrates end-to-end usage of the RAG system with sample queries
and includes utilities for monitoring document usage statistics.
"""

import uvicorn
import time
import requests
import chromadb
from threading import Thread
from typing import Dict, Any, List, Optional

from llama_index.core.indices.list.base import ListIndex
from llama_index.core import Settings
from llama_index.core.schema import NodeWithScore

import utils

# ChromaDB collection name - must match main.py
COLLECTION_NAME = "articles_chunk_database_new"


def start_server() -> None:
    """
    Start the FastAPI server on localhost:8000.

    This function runs the FastAPI application defined in main.py
    using uvicorn as the ASGI server.
    """
    uvicorn.run("main:app", host="127.0.0.1", port=8000)


def run_upload() -> None:
    """
    Test the document upload endpoint.

    This function sends a PUT request to the upload endpoint
    with a local JSON file containing document chunks.
    """
    url = "http://localhost:8000/api/upload"

    # Request payload for uploading local file
    data = {
        "schema_version": "1.0",
        "file": "./Sample_chunks.json"  # This file must exist on the **server**
    }

    response = requests.put(url, data=data)
    print(response.status_code)
    print(response.json())


def run_similarity_search(query: str) -> requests.Response:
    """
    Perform a similarity search using the API endpoint.

    Args:
        query (str): The search query text

    Returns:
        requests.Response: HTTP response containing search results
    """
    url = "http://localhost:8000/api/similarity_search"

    # Search parameters
    payload = {
        "query": query,
        "k": 5,  # Return top 5 results
        "min_score": 0.4  # Minimum similarity score threshold
    }

    response = requests.post(url, json=payload)
    return response


def run_chatbot(example_query: str, filtered_results: List[Dict[str, Any]]) -> Any:
    """
    Run the chatbot with retrieved documents and track usage.

    This function creates a ListIndex from filtered search results,
    runs a query through the chatbot, and tracks document usage statistics.

    Args:
        example_query (str): The query to ask the chatbot
        filtered_results (List[Dict[str, Any]]): Search results from similarity search

    Returns:
        Query response from the chatbot with source citations
    """
    # Convert dictionary results to NodeWithScore objects
    node_list = [utils.dict_to_node_with_score(d) for d in filtered_results]

    print("Usage count before update")
    check_usage_count(node_list)

    # Update usage statistics for retrieved documents
    update_usage_count(node_list)

    print("Usage count after update")
    check_usage_count(node_list)

    # Create index from retrieved nodes
    index = ListIndex(nodes=[elem.node for elem in node_list])

    # Configure LLM settings
    Settings.llm = utils.get_llm_gemini()

    # Create query engine and run query
    query_engine = index.as_query_engine()
    response = query_engine.query(example_query)

    print("-------------------------")
    print("CHATBOT RESPONSE")
    print(response)

    # Display citations
    print("Citations")
    for node in response.source_nodes:
        print(node.node.metadata["source_doc_id"])
        print(node.node.metadata["id"])
        print("-------------------------")

    return response


def update_usage_count(filtered_results: List[NodeWithScore]) -> None:
    """
    Update usage count for retrieved documents in ChromaDB.

    This function increments the usage_count metadata field
    for each document that was retrieved in a search.

    Args:
        filtered_results (List[NodeWithScore]): List of retrieved nodes with scores
    """
    # Connect to ChromaDB
    db = chromadb.PersistentClient(path="./storage/chroma")
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)

    # Extract document IDs
    id_list = [result.id_ for result in filtered_results]

    def update_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Increment usage count in metadata dictionary.

        Args:
            metadata (Dict[str, Any]): Document metadata

        Returns:
            Dict[str, Any]: Updated metadata with incremented usage count
        """
        for key, value in metadata.items():
            if key == "usage_count":
                metadata[key] = metadata[key] + 1
        return metadata

    # Batch update usage counts
    batch = chroma_collection.get(ids=id_list, include=["metadatas"])
    chroma_collection.update(
        ids=batch["ids"],
        metadatas=[update_metadata(metadata) for metadata in batch["metadatas"]]
    )


def check_usage_count(filtered_results: List[NodeWithScore]) -> None:
    """
    Check and display usage count statistics for retrieved documents.

    This function prints usage statistics for both the specific
    retrieved documents and all documents in the collection.

    Args:
        filtered_results (List[NodeWithScore]): List of retrieved nodes with scores
    """
    # Connect to ChromaDB
    db = chromadb.PersistentClient(path="./storage/chroma")
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)

    # Get usage counts for search results
    id_list = [result.id_ for result in filtered_results]
    batch = chroma_collection.get(ids=id_list, include=["metadatas"])

    usage_counts = []
    for metadata in batch.get("metadatas", []):
        usage = metadata.get("usage_count")  # default to 0 if missing
        usage_counts.append(usage)

    print("Usage counts search results:", usage_counts)

    # Get usage counts for all documents
    batch = chroma_collection.get(include=["metadatas", "documents"])

    usage_counts = []
    for metadata in batch.get("metadatas", []):
        usage = metadata.get("usage_count")
        usage_counts.append(usage)

    print("Total documents:", len(batch["ids"]))
    print("Usage counts all instances:", usage_counts)


def run_summary_endpoint() -> Dict[str, Any]:
    """
    Test the journal summarization endpoint.

    Returns:
        Dict[str, Any]: Response from the summary endpoint
    """
    summary_request = {
        "journal": "extension_brief_mucuna.pdf"
    }

    response = requests.post("http://localhost:8000/api/summary", json=summary_request)

    print("Status code:", response.status_code)
    return response.json()


def run_compare_endpoint() -> Dict[str, Any]:
    """
    Test the paper comparison endpoint.

    Returns:
        Dict[str, Any]: Response from the comparison endpoint
    """
    url = "http://localhost:8000/api/compare_papers"

    payload = {
        "doc_id_1": 'extension_brief_mucuna.pdf',
        "doc_id_2": "1706.03762v7.pdf"
    }

    response = requests.post(url, json=payload)

    print("Status code:", response.status_code)
    return response.json()


if __name__ == "__main__":
    """
    Main execution block for running the RAG application demo.

    This block demonstrates the complete workflow:
    1. Start the FastAPI server in a background thread
    2. Upload sample documents
    3. Perform similarity searches with different queries
    4. Run chatbot queries with citation tracking
    5. Test summarization and comparison endpoints
    """
    # Start server in background thread
    server_thread = Thread(target=start_server, daemon=True)
    server_thread.start()
    time.sleep(10)  # Wait for server to start

    # Define sample queries for testing
    example_query_1 = (
        "Given the adaptability of velvet bean to various environmental conditions, including its "
        "tolerance for long dry spells and poor soils, how might its agronomic traits contribute to "
        "food security and soil fertility in semi-arid regions such as natural regions IV and V of "
        "Zimbabwe?"
    )
    example_query_2 = "What is the optimal planting depth for mucuna?"
    example_query_3 = "How does the Transformer model improve over RNNs and CNNs in machine translation?"

    # Run the complete workflow
    run_upload()

    # Perform similarity search
    response_text = run_similarity_search(example_query_3)

    # Run chatbot with search results
    run_chatbot(example_query_3, response_text.json()["filtered_results"])

    # Test summarization endpoint
    summary_text = run_summary_endpoint()
    print("Summary:", summary_text['summary'])

    # Test comparison endpoint
    compare_text = run_compare_endpoint()
    print(compare_text['comparison'])