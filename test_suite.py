import pytest
import json
from unittest.mock import Mock, patch, MagicMock, mock_open
from fastapi.testclient import TestClient
from fastapi import HTTPException
import requests

# Import your app and models
from main import app, InputChunk, SearchPayload, SummaryRequest
from RAG_app import get_llm_gemini
from pydantic import ValidationError

client = TestClient(app)


class TestUploadEndpoint:
    """Test cases for the /api/upload endpoint"""

    @pytest.fixture
    def sample_json_data(self):
        return [
            {
                "id": "test_id_1",
                "source_doc_id": "doc_1",
                "chunk_index": 0,
                "section_heading": "Introduction",
                "journal": "Test Journal",
                "publish_year": 2023,
                "usage_count": 5,
                "attributes": ["attribute1", "attribute2"],
                "link": "https://example.com/doc1",
                "text": "This is test text 1"
            },
            {
                "id": "test_id_2",
                "source_doc_id": "doc_1",
                "chunk_index": 1,
                "section_heading": "Methods",
                "journal": "Test Journal",
                "publish_year": 2023,
                "usage_count": 3,
                "attributes": ["attribute3"],
                "link": "https://example.com/doc1",
                "text": "This is test text 2"
            }
        ]

    @patch('main.requests.get')
    @patch('main.HuggingFaceEmbedding')
    @patch('main.upload_db_with_deduplication')
    def test_upload_with_file_url(self, mock_upload_db, mock_embedding, mock_requests_get, sample_json_data):
        """Test uploading chunks from a URL"""
        # Mock the requests.get response
        mock_response = Mock()
        mock_response.json.return_value = sample_json_data
        mock_response.raise_for_status = Mock()
        mock_requests_get.return_value = mock_response

        # Mock embedding model
        mock_embed_instance = Mock()
        mock_embedding.return_value = mock_embed_instance

        # Mock upload_db function
        mock_upload_db.return_value = Mock()

        # Use proper form data encoding
        response = client.put(
            "/api/upload",
            data={
                "schema_version": "v1",
                "file_url": "https://example.com/data.json",
                "file": ""  # Empty string for optional parameter
            }
        )

        assert response.status_code == 202
        assert "Processing file from URL" in response.json()["message"]
        mock_requests_get.assert_called_once_with("https://example.com/data.json")
        mock_upload_db.assert_called_once()

    @patch('main.os.path.exists')
    @patch('main.HuggingFaceEmbedding')
    @patch('main.upload_db_with_deduplication')
    def test_upload_with_file_path(self, mock_upload_db, mock_embedding, mock_exists, sample_json_data):
        """Test uploading chunks from a local file"""
        mock_exists.return_value = True

        # Mock file reading
        with patch('builtins.open', mock_open(read_data=json.dumps(sample_json_data))):
            mock_embed_instance = Mock()
            mock_embedding.return_value = mock_embed_instance
            mock_upload_db.return_value = Mock()

            response = client.put(
                "/api/upload",
                data={
                    "schema_version": "v1",
                    "file": "/path/to/file.json",
                    "file_url": ""  # Empty string for optional parameter
                }
            )

        assert response.status_code == 202
        mock_upload_db.assert_called_once()

    def test_upload_missing_both_params(self):
        """Test upload fails when neither file_url nor file is provided"""
        response = client.put(
            "/api/upload",
            data={
                "schema_version": "v1",
                "file_url": "",
                "file": ""
            }
        )

        assert response.status_code == 400
        assert "Provide exactly one of file_url or file" in response.json()["detail"]

    def test_upload_both_params_provided(self):
        """Test upload fails when both file_url and file are provided"""
        response = client.put(
            "/api/upload",
            data={
                "schema_version": "v1",
                "file_url": "https://example.com/data.json",
                "file": "/path/to/file.json"
            }
        )

        assert response.status_code == 400
        assert "Provide exactly one of file_url or file" in response.json()["detail"]

    @patch('main.requests.get')
    def test_upload_url_fetch_failure(self, mock_requests_get):
        """Test handling of URL fetch failure"""
        mock_requests_get.side_effect = requests.exceptions.RequestException("Network error")

        response = client.put(
            "/api/upload",
            data={
                "schema_version": "v1",
                "file_url": "https://example.com/data.json",
                "file": ""
            }
        )

        assert response.status_code == 400
        assert "Failed to fetch JSON from URL" in response.json()["detail"]

    @patch('main.os.path.exists')
    def test_upload_file_not_found(self, mock_exists):
        """Test handling of non-existent file"""
        mock_exists.return_value = False

        response = client.put(
            "/api/upload",
            data={
                "schema_version": "v1",
                "file": "/nonexistent/file.json",
                "file_url": ""
            }
        )

        assert response.status_code == 400
        assert "File not found on server" in response.json()["detail"]


class TestSummaryEndpoint:
    """Test cases for the /api/summary endpoint"""

    @patch('main.get_journal')
    @patch('RAG_app.get_llm_gemini')
    def test_summary_success(self, mock_get_llm, mock_get_journal):
        """Test successful journal summarization"""
        # Mock journal data
        mock_get_journal.return_value = {
            'documents': ["Document chunk 1", "Document chunk 2", "Document chunk 3"]
        }

        # Mock LLM
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.text = "This is a summary of the journal content."
        mock_llm.complete.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        response = client.post(
            "/api/summary",
            json={"journal": "Test Journal"}
        )

        assert response.status_code == 200
        assert response.json() == {"summary": "This is a summary of the journal content."}
        mock_get_journal.assert_called_once_with("Test Journal")
        mock_llm.complete.assert_called_once()

    @patch('main.get_journal')
    @patch('main.get_llm_gemini')
    def test_summary_empty_journal(self, mock_get_llm, mock_get_journal):
        """Test summarization of empty journal"""
        # Mock empty journal data
        mock_get_journal.return_value = {
            'documents': []
        }

        # Mock LLM
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.text = "No content to summarize."
        mock_llm.complete.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        response = client.post(
            "/api/summary",
            json={"journal": "Empty Journal"}
        )

        assert response.status_code == 200

    def test_summary_invalid_request(self):
        """Test summary with invalid request body"""
        response = client.post(
            "/api/summary",
            json={"invalid_field": "Test Journal"}
        )

        assert response.status_code == 422  # Validation error


class TestInputChunkModel:
    """Test cases for the InputChunk Pydantic model"""

    def test_valid_input_chunk(self):
        """Test creation of valid InputChunk"""
        chunk_data = {
            "id": "test_id",
            "source_doc_id": "doc_1",
            "chunk_index": 0,
            "section_heading": "Introduction",
            "journal": "Test Journal",
            "publish_year": 2023,
            "usage_count": 5,
            "attributes": ["attr1", "attr2"],
            "link": "https://example.com/doc",
            "text": "Sample text"
        }

        chunk = InputChunk(**chunk_data)
        assert chunk.id == "test_id"
        assert chunk.source_doc_id == "doc_1"
        assert chunk.chunk_index == 0
        assert chunk.journal == "Test Journal"
        assert chunk.publish_year == 2023
        assert len(chunk.attributes) == 2
        assert str(chunk.link) == "https://example.com/doc"

    def test_invalid_url_in_chunk(self):
        """Test InputChunk validation with invalid URL"""
        chunk_data = {
            "id": "test_id",
            "source_doc_id": "doc_1",
            "chunk_index": 0,
            "section_heading": "Introduction",
            "journal": "Test Journal",
            "publish_year": 2023,
            "usage_count": 5,
            "attributes": ["attr1"],
            "link": "not_a_valid_url",  # Invalid URL
            "text": "Sample text"
        }

        with pytest.raises(ValidationError):
            InputChunk(**chunk_data)

    def test_missing_required_fields(self):
        """Test InputChunk validation with missing required fields"""
        chunk_data = {
            "id": "test_id",
            "source_doc_id": "doc_1",
            # Missing other required fields
        }

        with pytest.raises(ValidationError):
            InputChunk(**chunk_data)


class TestUploadDbWithDeduplication:
    """Test cases for the upload_db_with_deduplication function"""

    @patch('main.chromadb.PersistentClient')
    @patch('main.SimpleNodeParser')
    @patch('main.IngestionPipeline')
    @patch('main.SimpleDocumentStore')
    @patch('main.VectorStoreIndex')
    def test_upload_db_empty_collection(self, mock_index, mock_docstore, mock_pipeline_class,
                                        mock_parser, mock_chromadb):
        """Test uploading to empty collection"""
        # Mock chromadb
        mock_db_instance = Mock()
        mock_collection = Mock()
        mock_collection.count.return_value = 0  # Empty collection
        mock_db_instance.get_or_create_collection.return_value = mock_collection
        mock_chromadb.return_value = mock_db_instance

        # Mock pipeline
        mock_pipeline_instance = Mock()
        mock_pipeline_class.return_value = mock_pipeline_instance

        # Mock index
        mock_index_instance = Mock()
        mock_index.from_vector_store.return_value = mock_index_instance

        # Create test documents
        from main import Document
        documents = [
            Document(
                text="Test text",
                id_="doc_1_0",
                metadata={"source_doc_id": "doc_1", "chunk_index": 0}
            )
        ]

        # Call function
        from main import upload_db_with_deduplication
        result = upload_db_with_deduplication(documents, Mock())

        # Assertions
        mock_pipeline_instance.run.assert_called_once_with(documents=documents, show_progress=True)
        mock_pipeline_instance.persist.assert_called_once_with(persist_dir="./data")
        assert result == mock_index_instance

    @patch('main.chromadb.PersistentClient')
    @patch('main.SimpleNodeParser')
    @patch('main.IngestionPipeline')
    @patch('main.SimpleDocumentStore')
    @patch('main.VectorStoreIndex')
    def test_upload_db_existing_collection(self, mock_index, mock_docstore, mock_pipeline_class,
                                           mock_parser, mock_chromadb):
        """Test uploading to existing collection"""
        # Mock chromadb
        mock_db_instance = Mock()
        mock_collection = Mock()
        mock_collection.count.return_value = 10  # Non-empty collection
        mock_db_instance.get_or_create_collection.return_value = mock_collection
        mock_chromadb.return_value = mock_db_instance

        # Mock pipeline
        mock_pipeline_instance = Mock()
        mock_pipeline_class.return_value = mock_pipeline_instance

        # Mock index
        mock_index_instance = Mock()
        mock_index.from_vector_store.return_value = mock_index_instance

        # Create test documents
        from main import Document
        documents = [
            Document(
                text="Test text",
                id_="doc_1_0",
                metadata={"source_doc_id": "doc_1", "chunk_index": 0}
            )
        ]

        # Call function
        from main import upload_db_with_deduplication
        result = upload_db_with_deduplication(documents, Mock())

        # Assertions
        mock_pipeline_instance.load.assert_called_once_with(persist_dir="./data")
        mock_pipeline_instance.run.assert_called_once_with(documents=documents, show_progress=True)
        assert result == mock_index_instance


class TestHelperFunctions:
    """Test cases for helper functions"""

    @patch('main.requests.get')
    def test_fetch_json_from_url_success(self, mock_requests_get):
        """Test successful JSON fetch from URL"""
        from main import fetch_json_from_url

        mock_response = Mock()
        mock_response.json.return_value = {"key": "value"}
        mock_response.raise_for_status = Mock()
        mock_requests_get.return_value = mock_response

        result = fetch_json_from_url("https://example.com/data.json")

        assert result == {"key": "value"}
        mock_requests_get.assert_called_once_with("https://example.com/data.json")

    @patch('main.requests.get')
    def test_fetch_json_from_url_failure(self, mock_requests_get):
        """Test JSON fetch failure from URL"""
        from main import fetch_json_from_url

        mock_requests_get.side_effect = requests.exceptions.RequestException("Network error")

        with pytest.raises(HTTPException) as exc_info:
            fetch_json_from_url("https://example.com/data.json")

        assert exc_info.value.status_code == 400
        assert "Failed to fetch JSON from URL" in str(exc_info.value.detail)


class TestIntegration:
    """Integration tests for multiple components"""

    @patch('main.chromadb.PersistentClient')
    @patch('main.HuggingFaceEmbedding')
    @patch('main.VectorStoreIndex')
    @patch('main.SimpleNodeParser')
    @patch('main.IngestionPipeline')
    @patch('main.SimpleDocumentStore')
    @patch('main.requests.get')
    def test_end_to_end_upload_and_search(self, mock_requests_get, mock_docstore,
                                          mock_pipeline_class, mock_parser, mock_index,
                                          mock_embedding, mock_chromadb):
        """Test complete flow of upload and search"""
        # Setup mock data
        json_data = [{
            "id": "test_id",
            "source_doc_id": "doc_1",
            "chunk_index": 0,
            "section_heading": "Introduction",
            "journal": "Test Journal",
            "publish_year": 2023,
            "usage_count": 5,
            "attributes": ["ML", "AI"],
            "link": "https://example.com/doc",
            "text": "Machine learning is transforming AI"
        }]

        # Mock URL fetch
        mock_response = Mock()
        mock_response.json.return_value = json_data
        mock_response.raise_for_status = Mock()
        mock_requests_get.return_value = mock_response

        # Mock chromadb
        mock_db_instance = Mock()
        mock_collection = Mock()
        mock_collection.count.return_value = 0
        mock_db_instance.get_or_create_collection.return_value = mock_collection
        mock_chromadb.return_value = mock_db_instance

        # Mock embedding
        mock_embed_instance = Mock()
        mock_embedding.return_value = mock_embed_instance

        # Mock pipeline
        mock_pipeline_instance = Mock()
        mock_pipeline_class.return_value = mock_pipeline_instance

        # Mock index
        mock_index_instance = Mock()
        mock_index.from_vector_store.return_value = mock_index_instance

        # Test upload
        upload_response = client.put(
            "/api/upload",
            data={
                "schema_version": "v1",
                "file_url": "https://example.com/data.json",
                "file": "",
            }
        )

        assert upload_response.status_code == 202

        # Verify document creation
        assert mock_pipeline_instance.run.called
        call_args = mock_pipeline_instance.run.call_args
        documents = call_args[1]['documents']
        assert len(documents) == 1
        assert documents[0].text == "Machine learning is transforming AI"
        assert documents[0].id_ == "doc_1_0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])