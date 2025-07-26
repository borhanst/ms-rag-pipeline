import marimo

__generated_with = "0.14.12"
app = marimo.App(width="medium")


@app.cell
def _():
    from config import settings
    from src.document_processing import load_and_split_txt_documents
    from src.embedding_process import EmbeddingHandler
    from src.vector_storage import VectorStoreManager

    return (
        EmbeddingHandler,
        VectorStoreManager,
        load_and_split_txt_documents,
        settings,
    )


@app.cell
def _(load_and_split_txt_documents, settings):
    load_and_split_txt_documents(
        settings.TXT_PATH, chunk_overlap=100, chunk_size=2000
    )
    return


@app.cell
def _(EmbeddingHandler, settings):
    emb_handler = EmbeddingHandler(model_name=settings.EMBEDDING_MODEL)
    return (emb_handler,)


@app.cell
def _(VectorStoreManager, chunks, emb_handler, settings):
    manager = VectorStoreManager(
        emb_handler.embeddings_client, settings.FAISS_INDEX_PATH
    )
    if not manager.store_exists():
        manager.create_store_from_documents(chunks)
    manager.load_store()
    retriever = manager.get_retriever(search_kwargs={"k": 5})
    return (retriever,)


@app.cell
def _():
    question = "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
    return (question,)


@app.cell
def _(question, retriever):
    data = retriever.invoke(question)
    return (data,)


@app.cell
def _(data):
    for d in data:
        print(d.page_content)
        print("\n\n\n")
    return


if __name__ == "__main__":
    app.run()
