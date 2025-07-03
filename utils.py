from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.schema import NodeWithScore, TextNode, NodeRelationship, RelatedNodeInfo

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
