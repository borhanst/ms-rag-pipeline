# main.py
import os

import uvicorn
from fastapi import FastAPI

from config import settings
from src.document_processing import (
    load_and_split_documents,
    load_and_split_txt_documents,
)
from src.embedding_process import EmbeddingHandler
from src.llm_chain import create_rag_chain_lcel
from src.vector_storage import VectorStoreManager

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello, World!"}


@app.post("/create_index")
async def create_index():
    """Create an index from the uploaded PDF or TXT file."""

    chunks = load_and_split_txt_documents(
        settings.TXT_PATH,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        # max_chunks=5
    )
    manager = VectorStoreManager(
        EmbeddingHandler(model_name=settings.EMBEDDING_MODEL).embeddings_client,
        index_path=settings.FAISS_INDEX_PATH,
    )
    if not manager.store_exists():
        manager.create_store_from_documents(chunks)
    manager.load_store()
    return {
        "message": "Index deleted. Please run main() again to create a new one."
    }


@app.post("/chat")
async def chat(query: str):
    emb_handler = EmbeddingHandler(model_name=settings.EMBEDDING_MODEL)

    manager = VectorStoreManager(
        emb_handler.embeddings_client, settings.FAISS_INDEX_PATH
    )

    manager.load_store()
    retriever = manager.get_retriever(search_kwargs={"k": 5})

    # --- Create RAG Chain ---
    qa_chain = create_rag_chain_lcel(
        retriever, chat_model_name=settings.CHAT_MODEL, temperature=0.0
    )
    result = qa_chain.invoke({"question": query})

    return {
        "result": result,
    }



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
