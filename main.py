# main.py
import os

import uvicorn
from fastapi import FastAPI

from config import settings
from src.document_processing import load_and_split_documents, load_and_split_txt_documents
from src.embedding_process import EmbeddingHandler
from src.llm_chain import create_rag_chain_lcel
from src.vector_storage import VectorStoreManager

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello, World!"}


@app.post("/recreate_index")
async def recreate_index():
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
    # Re-run main to create new index (or call relevant parts)
    # For simplicity, we'll just print a message here.
    # You could call main() or the creation logic directly.
    print("Index deleted. Please run main() again to create a new one.")
    return {"message": "Index deleted. Please run main() again to create a new one."}


@app.post("/chat")
async def chat(query: str):
    emb_handler = EmbeddingHandler(model_name=settings.EMBEDDING_MODEL)

    manager = VectorStoreManager(
        emb_handler.embeddings_client, settings.FAISS_INDEX_PATH
    )

    manager.load_store()
    retriever = manager.get_retriever(search_kwargs={"k": 5})
    data = retriever.invoke(query)
    with open("retriver.txt", "w") as f:
        for d in data:
            f.write(str(d.page_content) + "\n")
    # --- Create RAG Chain ---
    qa_chain = create_rag_chain_lcel(
        retriever, chat_model_name=settings.CHAT_MODEL, temperature=0.0
    )
    result = qa_chain.invoke({"question": query})
    # result = invoke_rag_chain(qa_chain, query)
    # result_text = result["result"]
    # source_docs = result["source_documents"]

    return {
        "result": result,
        # "source_documents": source_docs
    }


def main():
    """Main function to orchestrate the RAG process using FAISS."""

    # Ensure API key is set
    if not settings.GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY not found in environment variables. Please set it in your .env file."
        )
    os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY

    # --- Document Processing ---
    chunks = load_and_split_documents(
        settings.PDF_PATH,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        # max_chunks=5
    )

    print(f"Loaded {len(chunks)} chunks from {settings.PDF_PATH}")

    # --- Embedding Handler ---
    emb_handler = EmbeddingHandler(model_name=settings.EMBEDDING_MODEL)

    manager = VectorStoreManager(
        emb_handler.embeddings_client, settings.FAISS_INDEX_PATH
    )
    if not manager.store_exists():
        manager.create_store_from_documents(chunks)
    manager.load_store()
    retriever = manager.get_retriever(search_kwargs={"k": 5})

    # --- Create RAG Chain ---
    # qa_chain = create_rag_chain_lcel(
    #     retriever,
    #     chat_model_name=settings.CHAT_MODEL,
    #     temperature=0.0
    # )
    # r = qa_chain.invoke({"question": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"})
    # print(r)
    # --- Query ---
    # query = "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?" # Replace with your actual query
    # print(f"\n🔍 Asking: {query}")
    # result = invoke_rag_chain(qa_chain, query) # Use wrapper
    # result_text = result["result"]
    # source_docs = result["source_documents"]

    # # --- Output ---
    # print("\n🔍 Answer:\n", result_text)
    # print("\n📄 Retrieved sources:", source_docs)
    # if "source_documents" in result:
    #     for i, doc in enumerate(result["source_documents"]):
    #         source = doc.metadata.get("source", "unknown")
    #         page = doc.metadata.get("page", "N/A")
    #         content_preview = doc.page_content[:200].replace("\n", " ")
    #         print(f"{i+1}. {source} (Page {page}) … {content_preview}")
    # else:
    #     print("No source documents found.")



    





if __name__ == "__main__":
    uvicorn.run('main:app', host="0.0.0.0", port=8001, reload=True)

