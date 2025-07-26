# main.py

from typing import List

import uvicorn
from fastapi import FastAPI, Request

from config import settings
from src.document_processing import (
    bangla_process_pdf,
    load_and_split_documents,
)
from src.embedding_process import EmbeddingHandler
from src.llm_chain import create_rag_chain_lcel
from src.vector_storage import VectorStoreManager

app = FastAPI()

# In-memory chat history
chat_history: List[str] = []


@app.get("/")
async def root():
    return {"message": "Hello, World!"}


@app.post("/create_pdf_to_txt")
async def create_pdf_to_txt():
    """Convert uploaded PDF file to TXT with enhanced Bangla text handling."""
    try:
        bangla_process_pdf(
            pdf_path=settings.PDF_PATH,
            output_file=settings.TXT_PATH,
        )
        return {
            "status": "success",
            "message": "PDF converted to TXT successfully.",
            "output_file": settings.TXT_PATH,
        }
    except Exception as e:
        return {"status": "error", "message": f"Conversion failed: {str(e)}"}


@app.post("/create_index")
async def create_index():
    """Create an optimized vector index from documents."""
    try:
        # Use only load_and_split_documents as it's more comprehensive
        chunks = load_and_split_documents(
            settings.TXT_PATH,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        manager = VectorStoreManager(
            EmbeddingHandler(
                model_name=settings.EMBEDDING_MODEL
            ).embeddings_client,
            index_path=settings.FAISS_INDEX_PATH,
        )
        if not manager.store_exists():
            manager.create_store_from_documents(chunks)
        manager.load_store()
        return {
            "status": "success",
            "message": "Vector index created successfully",
            "chunk_count": len(chunks),
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Index creation failed: {str(e)}",
        }


@app.post("/chat")
async def chat(query: str, request: Request):
    # --- Short-Term Memory: Maintain chat history ---
    chat_history.append(query)
    # Optionally, limit history length
    if len(chat_history) > 10:
        chat_history.pop(0)

    emb_handler = EmbeddingHandler(model_name=settings.EMBEDDING_MODEL)
    manager = VectorStoreManager(
        emb_handler.embeddings_client, settings.FAISS_INDEX_PATH
    )
    manager.load_store()
    retriever = manager.get_retriever(search_kwargs={"k": 5})

    # --- Create RAG Chain ---
    # Pass recent chat history as context (short-term memory)
    qa_chain = create_rag_chain_lcel(
        retriever,
        chat_model_name=settings.CHAT_MODEL,
        temperature=0.0,
        history=chat_history,  # Assumes your chain supports history/context
    )
    result = qa_chain.invoke({"question": query})

    return {
        "result": result,
        "short_term_memory": chat_history,
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
