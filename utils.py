"""
Utility functions for LLM integration and node processing.

This module provides helper functions for working with Google's Gemini AI model
and converting dictionary representations to LlamaIndex node objects.
"""

from typing import Dict, List, Any, Optional
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.schema import NodeWithScore, TextNode, NodeRelationship, RelatedNodeInfo


def get_llm_gemini() -> GoogleGenAI:
    """
    Initialize and return a Google Gemini AI model instance with safety settings disabled.

    This function creates a GoogleGenAI instance configured with:
    - Model: gemini-2.5-flash
    - Temperature: 0.01 (low randomness for consistent outputs)
    - All safety filters disabled to allow unrestricted content generation

    Returns:
        GoogleGenAI: Configured Gemini AI model instance

    Note:
        Requires 'gemini_api_key_2' to be defined in the global scope
    """
    # Safety settings configuration - all categories set to BLOCK_NONE
    # This disables all content filtering for maximum flexibility
    SAFE: List[Dict[str, str]] = [
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

    # Initialize the Gemini LLM with specified configuration
    llm_gemini: GoogleGenAI = GoogleGenAI(
        model_name="models/gemini-2.5-flash",
        api_key=gemini_api_key_2,
        temperature=0.01,
        safety_settings=SAFE
    )
    return llm_gemini


def dict_to_node_with_score(node_dict: Dict[str, Any]) -> NodeWithScore:
    """
    Convert dictionary representation to NodeWithScore object.

    This function reconstructs a NodeWithScore object from its dictionary representation,
    including all node metadata, relationships, and scoring information.

    Args:
        node_dict (Dict[str, Any]): Dictionary containing node data with the following structure:
            - 'node': Dict containing TextNode data (id_, text, extra_info, etc.)
            - 'score': Float representing the node's relevance score
            - 'node']['relationships']: Optional dict of node relationships

    Returns:
        NodeWithScore: Reconstructed NodeWithScore object with all original properties

    Example:
        >>> node_dict = {
        ...     'node': {
        ...         'id_': 'node_123',
        ...         'text': 'Sample text content',
        ...         'extra_info': {'source': 'document.pdf'}
        ...     },
        ...     'score': 0.85
        ... }
        >>> node_with_score = dict_to_node_with_score(node_dict)
    """
    # Extract node data from the dictionary
    node_data: Dict[str, Any] = node_dict['node']

    # Create TextNode with all available properties
    # Using .get() with defaults for optional fields
    text_node: TextNode = TextNode(
        id_=node_data['id_'],  # Required: unique identifier for the node
        text=node_data['text'],  # Required: main text content
        metadata=node_data['extra_info'],  # Node metadata (renamed from extra_info)
        excluded_embed_metadata_keys=node_data.get('excluded_embed_metadata_keys', []),
        excluded_llm_metadata_keys=node_data.get('excluded_llm_metadata_keys', []),
        metadata_template=node_data.get('metadata_template', '{key}: {value}'),
        metadata_seperator=node_data.get('metadata_seperator', '\n'),
        text_template=node_data.get('text_template', '{metadata_str}\n\n{content}'),
        start_char_idx=node_data.get('start_char_idx'),  # Optional: start position in source
        end_char_idx=node_data.get('end_char_idx'),  # Optional: end position in source
        mimetype=node_data.get('mimetype', 'text/plain')  # Default to plain text
    )

    # Add relationships if they exist in the node data
    if 'relationships' in node_data:
        for rel_type, rel_data in node_data['relationships'].items():
            # Convert string relationship type to NodeRelationship enum
            rel_type_enum: NodeRelationship = NodeRelationship(rel_type)

            # Create RelatedNodeInfo object for the relationship
            related_node: RelatedNodeInfo = RelatedNodeInfo(
                node_id=rel_data['node_id'],  # Required: ID of the related node
                node_type=rel_data.get('node_type'),  # Optional: type of related node
                metadata=rel_data.get('metadata', {}),  # Optional: relationship metadata
                hash=rel_data.get('hash')  # Optional: content hash for verification
            )

            # Add the relationship to the text node
            text_node.relationships[rel_type_enum] = related_node

    # Create and return NodeWithScore object
    node_with_score: NodeWithScore = NodeWithScore(
        node=text_node,  # The reconstructed TextNode
        score=node_dict['score']  # The relevance/similarity score
    )

    return node_with_score