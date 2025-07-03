import uvicorn
import time
import requests
import chromadb
from threading import Thread

from llama_index.core.indices.list.base import ListIndex
from llama_index.core import Settings
import utils

COLLECTION_NAME = "articles_chunk_database_new"

def start_server():
    uvicorn.run("main:app", host="127.0.0.1", port=8000)

def run_upload():
    url = "http://localhost:8000/api/upload"

    data = {
        "schema_version": "1.0",
        "file": "./Sample_chunks.json"  # This file must exist on the **server**
    }
    response = requests.put(url, data= data)
    print(response.status_code)
    print(response.json())

def run_similarity_search(query):
    url = "http://localhost:8000/api/similarity_search"


    payload = {
        "query": query,
        "k": 5,
        "min_score": 0.4
    }

    response = requests.post(url, json=payload)
    return response


def run_chatbot(example_query, filtered_results):

    node_list = [utils.dict_to_node_with_score(d) for d in filtered_results]

    print("Usage count before update")
    check_usage_count(node_list)

    update_usage_count(node_list)

    print("Usage count after update")
    check_usage_count(node_list)

    index = ListIndex(nodes=[elem.node for elem in node_list])

    Settings.llm = utils.get_llm_gemini()

    query_engine = index.as_query_engine()

    response = query_engine.query(example_query)
    print("-------------------------")

    print("CHATBOT RESPONSE")
    print(response)

    print("Citations")
    for node in response.source_nodes:
        print(node.node.metadata["source_doc_id"])
        print(node.node.metadata["id"])
        print("-------------------------")

    return response

def update_usage_count(filtered_results):

    db = chromadb.PersistentClient(path="./storage/chroma")
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)

    id_list = [result.id_ for result in filtered_results]

    def update_metadata(metadata: dict):
        for key, value in metadata.items():
            if key == "usage_count":
                metadata[key] = metadata[key] + 1
        return metadata

    batch = chroma_collection.get(ids=id_list, include=["metadatas"])
    chroma_collection.update(ids=batch["ids"], metadatas=[update_metadata(metadata) for metadata in batch["metadatas"]])

def check_usage_count(filtered_results):

    db = chromadb.PersistentClient(path="./storage/chroma")
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)

    id_list = [result.id_ for result in filtered_results]
    batch = chroma_collection.get(ids=id_list, include=["metadatas"])

    usage_counts = []
    for metadata in batch.get("metadatas", []):
        usage = metadata.get("usage_count")  # default to 0 if missing
        usage_counts.append(usage)

    print("Usage counts search results:", usage_counts)

    batch = chroma_collection.get(include=["metadatas", "documents"])

    usage_counts = []
    for metadata in batch.get("metadatas", []):
        usage = metadata.get("usage_count")
        usage_counts.append(usage)

    print("Total documents:", len(batch["ids"]))
    print("Usage counts all instances:", usage_counts)


def run_summary_endpoint():

    summary_request = {
        "journal": "extension_brief_mucuna.pdf"
    }

    response = requests.post("http://localhost:8000/api/summary", json=summary_request)

    print("Status code:", response.status_code)
    return response.json()

def run_compare_endpoint():

    url = "http://localhost:8000/api/compare_papers"

    payload = {
        "doc_id_1": 'extension_brief_mucuna.pdf',
        "doc_id_2": "1706.03762v7.pdf"
    }

    response = requests.post(url, json=payload)

    print("Status code:", response.status_code)
    return response.json()


if __name__ == "__main__":
    server_thread = Thread(target=start_server, daemon=True)
    server_thread.start()
    time.sleep(10)

    example_query_1 = ("Given the adaptability of velvet bean to various environmental conditions, including its "
                       "tolerance for long dry spells and poor soils, how might its agronomic traits contribute to "
                       "food security and soil fertility in semi-arid regions such as natural regions IV and V of "
                       "Zimbabwe?")
    example_query_2 = "What is the optimal planting depth for mucuna?"

    example_query_3 = "How does the Transformer model improve over RNNs and CNNs in machine translation?"

    run_upload()

    response_text = run_similarity_search(example_query_3)

    run_chatbot(example_query_3, response_text.json()["filtered_results"])
    
    summary_text = run_summary_endpoint()
    print("Summary:", summary_text['summary'])

    compare_text = run_compare_endpoint()
    print(compare_text['comparison'])

