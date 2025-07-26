from pathlib import Path

from bangla_pdf_ocr import process_pdf
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def bangla_process_pdf(pdf_path: str, output_file: str = "./data/output.txt"):
    extracted_text = process_pdf(pdf_path)
    with open(output_file, "w") as f:
        f.write(extracted_text)
    return extracted_text


def load_and_split_documents(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    max_chunks: int | None = None
):
    """
    Loads a PDF and splits specified pages into text chunks.

    max_chunks : int | None
        If provided, the function returns **at most** this many chunks
        (or exactly this many if `exact=True`).
    exact : bool
        When True, the list is padded with empty Documents or truncated
        so that **exactly** `max_chunks` items are returned.
    """
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    selected_docs = docs

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "।", " "]
    )
    chunks = text_splitter.split_documents(selected_docs)

    # ---- NEW: limit / pad the output ---------------------------------------
    if max_chunks is not None:
        chunks = chunks[:max_chunks]

    print(f"Returning {len(chunks)} chunks.")
    return chunks


def load_and_split_txt_documents(
    txt_path: Path | str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    max_chunks: int | None = None,
    exact: bool = False,
):
    """
    Loads a PDF and splits specified pages into text chunks.

    max_chunks : int | None
        If provided, the function returns **at most** this many chunks
        (or exactly this many if `exact=True`).
    exact : bool
        When True, the list is padded with empty Documents or truncated
        so that **exactly** `max_chunks` items are returned.
    """
    loader = TextLoader(txt_path)
    docs = loader.load()

    selected_docs = docs

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_overlap=chunk_overlap, chunk_size=chunk_size
    )
    chunks = text_splitter.split_documents(selected_docs)

    # ---- NEW: limit / pad the output ---------------------------------------
    if max_chunks is not None:
        if exact:
            from langchain_core.documents import Document

            # Trim or pad with empty Documents
            if len(chunks) > max_chunks:
                chunks = chunks[:max_chunks]
            else:
                empty = Document(page_content="", metadata={})
                chunks.extend([empty] * (max_chunks - len(chunks)))
        else:
            # simple truncate
            chunks = chunks[:max_chunks]

    print(f"Returning {len(chunks)} chunks.")
    return chunks
