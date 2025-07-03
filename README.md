# RAG Document Indexing and Retrieval System

A FastAPI-based Retrieval-Augmented Generation (RAG) system that provides document indexing, semantic search, summarization, and comparison capabilities using ChromaDB and LlamaIndex.

## 🚀 Features

- **Document Indexing**: Upload and index document chunks from JSON files or URLs
- **Semantic Search**: Perform similarity search on indexed documents using HuggingFace embeddings
- **Document Summarization**: Generate AI-powered summaries of individual journals
- **Paper Comparison**: Compare multiple research papers side-by-side
- **Usage Tracking**: Monitor document access patterns and usage statistics
- **Citation Support**: Automatic citation generation for retrieved content

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    FastAPI      │    │   ChromaDB      │    │   LlamaIndex    │
│   Web Server    │◄──►│ Vector Storage  │◄──►│   Processing    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  HuggingFace    │    │    Gemini AI    │    │   Usage Stats   │
│   Embeddings    │    │      LLM        │    │   Tracking      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for embeddings)
- Google Gemini API key
- Required Python packages (see requirements below)

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd rag-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:
   ```bash
   pip install fastapi uvicorn chromadb llama-index requests pydantic
   pip install llama-index-embeddings-huggingface
   pip install llama-index-vector-stores-chroma
   pip install llama-index-llms-google-genai
   ```

3. **Set up environment variables**
   ```bash
   # Create a .env file or set environment variable
   export GEMINI_API_KEY="your_gemini_api_key_here"
   ```

4. **Create required directories**
   ```bash
   mkdir -p storage/chroma data
   ```

## 🚀 Quick Start

### 1. Start the Server

```python
python RAG_app.py
```

The server will start on `http://localhost:8000`

### 2. Upload Documents

**Via API:**
```bash
curl -X PUT "http://localhost:8000/api/upload" \
  -F "schema_version=1.0" \
  -F "file=./sample_chunks.json"
```

**Via URL:**
```bash
curl -X PUT "http://localhost:8000/api/upload" \
  -F "schema_version=1.0" \
  -F "file_url=https://example.com/data.json"
```

### 3. Search Documents

```bash
curl -X POST "http://localhost:8000/api/similarity_search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning transformers",
    "k": 5,
    "min_score": 0.4
  }'
```

## 📊 Data Format

### Input JSON Structure

Your `sample_chunks.json` should follow this schema:

```json
[
  {
    "id": "unique_chunk_id",
    "source_doc_id": "document_identifier",
    "chunk_index": 0,
    "section_heading": "Introduction",
    "journal": "Nature Machine Intelligence",
    "publish_year": 2023,
    "usage_count": 0,
    "attributes": ["AI", "NLP", "Transformers"],
    "link": "https://example.com/paper.pdf",
    "text": "The actual text content of the document chunk..."
  }
]
```

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier for the chunk |
| `source_doc_id` | string | Identifier of the source document |
| `chunk_index` | integer | Index position within the document |
| `section_heading` | string | Section heading for the chunk |
| `journal` | string | Journal/publication name |
| `publish_year` | integer | Publication year |
| `usage_count` | integer | Access count (starts at 0) |
| `attributes` | array | List of tags/categories |
| `link` | string (URL) | Link to original document |
| `text` | string | The actual text content |

## 🔌 API Endpoints

### Document Management

#### Upload Documents
```http
PUT /api/upload
Content-Type: multipart/form-data

schema_version: "1.0"
file: "path/to/chunks.json"
# OR
file_url: "https://example.com/data.json"
```

#### Get Journal Documents
```http
GET /api/{journal_id}
```

### Search & Retrieval

#### Similarity Search
```http
POST /api/similarity_search
Content-Type: application/json

{
  "query": "search query text",
  "k": 5,
  "min_score": 0.4
}
```

### AI Services

#### Summarize Journal
```http
POST /api/summary
Content-Type: application/json

{
  "journal": "journal_name"
}
```

#### Compare Papers
```http
POST /api/compare_papers
Content-Type: application/json

{
  "doc_id_1": "first_paper_id",
  "doc_id_2": "second_paper_id"
}
```

## 📝 Usage Examples

### Python Client Examples

```python
import requests

# Upload documents
response = requests.put("http://localhost:8000/api/upload", 
                       data={"schema_version": "1.0", "file": "./data.json"})

# Search documents
search_payload = {
    "query": "transformer architecture attention mechanism",
    "k": 3,
    "min_score": 0.5
}
response = requests.post("http://localhost:8000/api/similarity_search", 
                        json=search_payload)

# Summarize a journal
summary_request = {"journal": "nature_ai_2023.pdf"}
response = requests.post("http://localhost:8000/api/summary", 
                        json=summary_request)

# Compare papers
compare_request = {
    "doc_id_1": "transformer_paper.pdf",
    "doc_id_2": "bert_paper.pdf"
}
response = requests.post("http://localhost:8000/api/compare_papers", 
                        json=compare_request)
```



## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | Required |
| `CHROMA_PERSIST_PATH` | ChromaDB storage path | `./storage/chroma` |
| `EMBEDDING_MODEL` | HuggingFace model name | `BAAI/llm-embedder` |
| `COLLECTION_NAME` | ChromaDB collection name | `articles_chunk_database_new` |

### Model Configuration

The system uses:
- **Embeddings**: BAAI/llm-embedder (HuggingFace)
- **LLM**: Google Gemini 2.5 Flash
- **Vector Store**: ChromaDB with persistent storage

## 🏃‍♂️ Running Tests

Run the complete test suite:

```python
python RAG_app.py
```

This will:
1. Start the FastAPI server
2. Upload sample documents
3. Perform similarity searches
4. Test summarization and comparison
5. Display usage statistics

## 📁 File Structure

```
rag-system/
├── main.py              # FastAPI application and endpoints
├── RAG_app.py           # Application runner and test suite
├── utils.py             # Utility functions for LLM integration
├── sample_chunks.json   # Example input data
├── requirements.txt     # Python dependencies
├── storage/             # ChromaDB persistent storage
│   └── chroma/
├── data/                # Pipeline persistence directory
└── README.md           # This documentation
```

## 🔍 Core Components

### main.py
- FastAPI application setup
- Document upload and indexing
- Similarity search implementation
- Summarization and comparison endpoints
- ChromaDB integration

### RAG_app.py
- Application runner and testing suite
- Usage tracking and statistics
- End-to-end workflow demonstration
- Performance monitoring

### utils.py
- Google Gemini LLM integration
- Node conversion utilities
- Safety settings configuration

## 🎯 Performance Considerations

- **GPU Acceleration**: Uses CUDA for embedding generation
- **Batch Processing**: Efficient batch operations for large document sets
- **Deduplication**: Automatic handling of duplicate documents
- **Persistent Storage**: ChromaDB provides efficient vector storage
- **Usage Tracking**: Monitors document access patterns for optimization

## 🔒 Security Notes

- The Gemini LLM is configured with safety filters disabled for research purposes
- Ensure proper API key management and environment variable security
- Consider implementing authentication for production deployments

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🐛 Troubleshooting

### Common Issues

**ChromaDB Connection Error**
```bash
# Ensure storage directory exists
mkdir -p storage/chroma
```

**CUDA Out of Memory**
```python
# Reduce batch size or use CPU
embed_model = HuggingFaceEmbedding(model_name="BAAI/llm-embedder", device="cpu")
```

**API Key Issues**
```bash
# Verify environment variable
echo $GEMINI_API_KEY
```

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📞 Support

For issues and questions:
- Create an issue in the GitHub repository
- Check the troubleshooting section
- Review the API documentation

## 🎉 Acknowledgments

- [LlamaIndex](https://github.com/run-llama/llama_index) for RAG framework
- [ChromaDB](https://github.com/chroma-core/chroma) for vector storage
- [HuggingFace](https://huggingface.co/) for embedding models
- [Google Gemini](https://ai.google.dev/) for language model capabilities
