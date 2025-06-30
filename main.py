from typing import Union


from fastapi import FastAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


embed_model = HuggingFaceEmbedding(model_name="BAAI/llm-embedder", device="cuda")

app = FastAPI()


@app.put("/")
def read_root():
    example_txt = ["WARNING: Failed to find MSVC"]
    embed_model._get_text_embedding()
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}