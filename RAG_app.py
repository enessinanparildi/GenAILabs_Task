import uvicorn
import time
import requests
import chromadb
from threading import Thread

from llama_index.core.indices.list.base import ListIndex
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import Settings
from llama_index.core.schema import NodeWithScore, TextNode, NodeRelationship, RelatedNodeInfo


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

    node_list = [dict_to_node_with_score(d) for d in filtered_results]

    print("Usage count before update")
    check_usage_count(node_list)

    update_usage_count(node_list)

    print("Usage count after update")
    check_usage_count(node_list)

    index = ListIndex(nodes=[elem.node for elem in node_list])

    SAFE = [
        {
            "category": "HARM_CATEGORY_DANGEROUS",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        },
    ]

    llm_gemini = GoogleGenAI(model_name="models/gemini-2.0-flash", api_key=gemini_api_key_2,
                             temperature=0.01, safety_settings=SAFE)

    Settings.llm = llm_gemini

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

def dict_to_node_with_score(node_dict):
    """Convert dictionary representation to NodeWithScore object"""
    # Extract node data
    node_data = node_dict['node']
    # Create TextNode
    text_node = TextNode(
        id_=node_data['id_'],
        text=node_data['text'],
        metadata=node_data['extra_info'],
        excluded_embed_metadata_keys=node_data.get('excluded_embed_metadata_keys', []),
        excluded_llm_metadata_keys=node_data.get('excluded_llm_metadata_keys', []),
        metadata_template=node_data.get('metadata_template', '{key}: {value}'),
        metadata_seperator=node_data.get('metadata_seperator', '\n'),
        text_template=node_data.get('text_template', '{metadata_str}\n\n{content}'),
        start_char_idx=node_data.get('start_char_idx'),
        end_char_idx=node_data.get('end_char_idx'),
        mimetype=node_data.get('mimetype', 'text/plain')
    )
    # Add relationships if they exist
    if 'relationships' in node_data:
        for rel_type, rel_data in node_data['relationships'].items():
            # Convert string relationship type to NodeRelationship enum
            rel_type_enum = NodeRelationship(rel_type)
            # Create RelatedNodeInfo
            related_node = RelatedNodeInfo(
                node_id=rel_data['node_id'],
                node_type=rel_data.get('node_type'),
                metadata=rel_data.get('metadata', {}),
                hash=rel_data.get('hash')
            )
            text_node.relationships[rel_type_enum] = related_node
    # Create NodeWithScore
    node_with_score = NodeWithScore(
        node=text_node,
        score=node_dict['score']
    )
    return node_with_score

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



def get_llm_gemini():

    SAFE = [
        {
            "category": "HARM_CATEGORY_DANGEROUS",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        },
    ]


    llm_gemini = GoogleGenAI(model_name="models/gemini-2.5-flash", api_key=gemini_api_key_2,
                             temperature=0.01, safety_settings=SAFE)
    return llm_gemini

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

    example_query_1 = "Given the adaptability of velvet bean to various environmental conditions, including its tolerance for long dry spells and poor soils, how might its agronomic traits contribute to food security and soil fertility in semi-arid regions such as natural regions IV and V of Zimbabwe?"
    example_query_2 = "What is the optimal planting depth for mucuna?"

    run_upload()

    response_text = run_similarity_search(example_query_2)

    run_chatbot(example_query_2, response_text.json()["filtered_results"])
    summary_text = run_summary_endpoint()
    print("Summary:", summary_text['summary'])

    compare_text = run_compare_endpoint()
    print(compare_text['comparison'])

