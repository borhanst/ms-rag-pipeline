# modules/vector_store.py
from __future__ import annotations

import os
import shutil
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever


class VectorStoreManager:
    """
    Manages the life-cycle of a local FAISS vector store:
    creation, saving, loading, retrieval, and deletion.
    """

    def __init__(
        self,
        embeddings_client: Embeddings,
        index_path: str = "faiss_index",
    ) -> None:
        self.embeddings_client = embeddings_client
        self.index_path = index_path
        self.vectorstore: FAISS | None = None

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------
    def create_store_from_documents(self, documents: List[Document]) -> None:
        """
        Build a new FAISS index from a list of documents and persist it.

        Parameters
        ----------
        documents : list[Document]
            Texts to index. An empty list will raise ValueError.
        """
        if not documents:
            raise ValueError("Cannot create store from empty document list.")

        print("Creating FAISS vector store from documents...")
        self.vectorstore = FAISS.from_documents(documents, self.embeddings_client)
        self.save_store()
        print(f"FAISS vector store saved to '{self.index_path}'.")

    def save_store(self) -> None:
        """Persist the current vector store to disk."""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_store_from_documents first.")

        os.makedirs(self.index_path, exist_ok=True)
        self.vectorstore.save_local(self.index_path)

    def load_store(self) -> None:
        """Load the vector store from disk."""
        if not self.store_exists():
            raise FileNotFoundError(
                f"FAISS index not found at '{self.index_path}'. "
                "Create it with create_store_from_documents."
            )

        print(f"Loading FAISS vector store from '{self.index_path}'...")
        self.vectorstore = FAISS.load_local(
            self.index_path,
            self.embeddings_client,
            allow_dangerous_deserialization=True,  # required for pickle-based FAISS
        )
        print("FAISS vector store loaded.")

    def delete_store(self) -> None:
        """Remove the persisted vector store directory."""
        if os.path.exists(self.index_path):
            shutil.rmtree(self.index_path)
            print(f"Deleted FAISS vector store at '{self.index_path}'.")
        else:
            print(f"Path '{self.index_path}' does not exist—nothing to delete.")

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------
    def get_retriever(
        self,
        search_kwargs: Optional[Dict[str, Any]] = None,
    ) -> VectorStoreRetriever:
        """
        Obtain a retriever from the loaded vector store.

        Parameters
        ----------
        search_kwargs : dict, optional
            Additional search parameters (e.g., {'k': 5}).

        Returns
        -------
        VectorStoreRetriever
        """
        if self.vectorstore is None:
            raise ValueError(
                "Vector store not loaded. Call load_store or create_store_from_documents."
            )
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs or {})

    def store_exists(self) -> bool:
        """True if a FAISS index already exists on disk."""
        return os.path.isfile(os.path.join(self.index_path, "index.faiss"))
    
    
    # Optional: Add a function to force re-creation of the index
    def recreate_index(self):
        """Function to delete the existing FAISS index and create a new one."""
        self.delete_store()  # Delete old index
        # Re-run main to create new index (or call relevant parts)
        # For simplicity, we'll just print a message here.
        # You could call main() or the creation logic directly.
        print("Index deleted. Please run main() again to create a new one.")
        self.load_store()