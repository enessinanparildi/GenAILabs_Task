# Project Overview: RAG Pipeline for Academic Journals

## Detecting Newly Uploaded Journal Files

I didn't fully understand the first point. Are we aiming to prevent duplicate entries from being added to the vector database? I assume that each journal file corresponds to a single entry in the example JSON.

If our goal is to detect duplicate journal PDFs, we could use the MD5 hash of the PDF content to identify duplicates and ensure that only unique uploads are processed.

Otherwise, if we use the IngestionPipeline class from LlamaIndex with a Docstore, the IngestionPipeline class from LlamaIndex automatically implements a deduplication strategy to prevent duplicate uploads to the vector database.

## Vector Database Choice

I use ChromaDB. ChromaDB is excellent for POCs due to its simplicity and zero-infrastructure requirements, but lacks hybrid search capabilities. It can be run locally and does not require setting up a docker service.

For production systems handling academic journals, Pinecone or Weaviate would be better choices as they support hybrid search (combining semantic similarity with keyword matching), which is crucial for supporting both conceptual and exact-term searches. Pinecone or Weaviate can be hosted on cloud making them more suitable for production environments.

## Embedding Generation

I utilized an open source encoder model from Huggingface called BAAI/llm-embedder. For details refer to https://huggingface.co/BAAI/llm-embedder. Alternatively, if a smaller model is needed, a model from the sentence transformers library can be utilized.

We can improve chunk embedding quality by prepending section headings to main chunk bodies.

Metadata attachment is straightforward, as this is supported by ChromaDB. Generating unique chunk IDs is a good idea. Llamaindex automatically handles this by generating uuid for each row.

## Document Chunking

The one important constraint is that chunk size has to be smaller than embedding model’s context length. Embedding models typically have much smaller context lengths than LLMs.

Since we are tackling academic documents, we can reasonably expect that the language is dense and technical. For this type of language, making chunks larger (e.g., 200-300 tokens) might be more optimal, because the meaning often spans multiple sentences.

We should also use a certain level of overlap between consecutive chunks for ensuring continuity and flow. Finally, considering the logical sections (Abstract, Introduction, Methods, Result) of articles when running chunking is always beneficial. We can use these sections as hard boundaries during chunking.

## Optional Features

### Summary Endpoint

This FastAPI endpoint receives a journal name, retrieves all related text chunks from a vector database, combines them into a single document, and sends the merged content to a language model (LLM) to generate a concise summary. It acts as an automated summarization service for full journal content using semantic retrieval and LLM reasoning.

For this I am not completely sure by which json key we should perform the chunk merging.

### Compare Endpoint

The /api/compare_papers endpoint takes two document IDs, retrieves their associated text chunks from the vector database, merges the content of each, and uses a language model to generate a structured comparison highlighting similarities, differences, methodologies, and key findings between the two documents.

### Persistent Usage Tracking

I added a utility function that updates the usage_count for the rows in the database that are returned by a similarity query.

### A Unit Test Suite

I added some test functions with the help of Claude 4 to validate the upload/ingestion logic and the summary endpoint. I could have made the tests more comprehensive, but I didn't have much time. The main RAG_app script also implicitly runs tests.

### Frontend UI

I implemented a simple Flask app as the frontend with the help of Claude 4. While it’s far from perfect, I didn’t have much time to refine it. Frontend development isn’t my primary area of expertise, but I’m confident I can quickly get up to speed if needed.
