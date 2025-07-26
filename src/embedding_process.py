
from __future__ import annotations

import logging
from typing import List

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_openai import OpenAIEmbeddings
except ImportError as e:
    raise ImportError(
        "langchain-google-genai is required. "
        "Install with: pip install langchain-google-genai"
    ) from e

from tenacity import retry, stop_after_attempt, wait_random_exponential, before_sleep_log

logger = logging.getLogger(__name__)

class EmbeddingHandler:
    """
    Handles the creation of embeddings using Google Generative AI.

    Example
    -------
    >>> handler = EmbeddingHandler()
    >>> vectors = handler.embed_documents(["hello world", "goodbye"])
    """

    def __init__(self, model_name: str = "models/gemini-embedding-001") -> None:
        # self.embeddings_client = GoogleGenerativeAIEmbeddings(model=model_name)
        self.embeddings_client = OpenAIEmbeddings(model="text-embedding-3-small")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(min=1, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of text strings.

        Parameters
        ----------
        texts : list[str]
            Texts to embed. Non-string items are coerced to ``str``.

        Returns
        -------
        list[list[float]]
            Embedding vectors, one per input text.
        """
        texts = [str(t) for t in texts]  # defensive
        logger.info("Embedding %d documents...", len(texts))
        return self.embeddings_client.embed_documents(texts)

    def embed_query(self, query_text: str) -> List[float]:
        """
        Embed a single query string.

        Parameters
        ----------
        query_text : str
            The query text. Non-string objects are coerced to ``str``.

        Returns
        -------
        list[float]
            The embedding vector.
        """
        return self.embeddings_client.embed_query(str(query_text))