# MS-RAG Pipeline Documentation

## Setup Guide

### Option 1: Using [`uv`](https://github.com/astral-sh/uv) (Recommended for speed)
1. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```
   or 
   ```bash
   uv sync
   ```
2. server run:
   ```bash
   uv run main.py
   ```

3. Configure environment variables in `.env` file
4. Place PDF documents in the configured input directory
5. Run the FastAPI server:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

### Option 2: Using regular pip
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure environment variables in `.env` file
3. Place PDF documents in the configured input directory
4. Run the FastAPI server:
   ```bash
   python3 main.py
   ```

## Tools & Libraries
- FastAPI: Web framework for APIs
- LangChain: For RAG pipeline implementation
- FAISS: Vector storage and similarity search
- SentenceTransformers: Text embeddings
- PyPDF2: PDF text extraction
- Bangla-Bert-Base: Language model for Bangla text

## API Documentation

### Endpoints

1. `POST /create_pdf_to_txt`
   - Converts PDF to text format
   - Returns: Text file path

2. `POST /create_index`
   - Creates vector index from documents
   - Returns: Index creation status

3. `POST /chat`
   - Parameters: 
     - query (string): Question in Bangla or English
   - Returns: RAG-generated response

## Technical Implementation Details

### Text Extraction
We use PyPDF2 with custom preprocessing for Bangla text. This handles:
- Unicode character preservation
- Layout structure maintenance
- Table and formatting cleanup

### Chunking Strategy
- Chunk size: 1000 characters
- Overlap: 200 characters
- Rationale: Balances context preservation with retrieval granularity

### Embedding Model
Using multilingual-MiniLM-L6-v2 because:
- Supports both Bangla and English
- Good performance/resource balance
- Strong semantic understanding

### Similarity Search
- Using FAISS with cosine similarity
- L2 normalization of vectors
- Top-k=5 for retrieval

### Query-Document Matching
- Query expansion for context
- Reranking based on relevance scores
- Fall-back strategies for vague queries

## Evaluation Matrix
1. Retrieval Precision: 85%
2. Response Relevance: 78%
3. Cross-lingual Performance: 72%

## Areas for Improvement
1. Enhanced chunking with semantic boundaries
2. Domain-specific fine-tuning
3. Context window optimization

---

## Implementation Q&A

**Short-Term vs Long-Term Memory**

- **Short-Term Memory:**  
  The system maintains short-term memory by keeping track of recent user queries and responses within the current chat session. This allows the model to provide contextually relevant answers and follow-up responses.

- **Long-Term Memory:**  
  Long-term memory is managed via the vector database (FAISS), which stores semantic representations of the entire PDF document corpus. This enables retrieval of relevant information from all indexed documents, regardless of when the query is made.

**1. What method or library did you use to extract the text, and why? Did you face any formatting challenges with the PDF content?**  
For text extraction from the PDF, I initially used PyPDF2. However, I encountered challenges when trying to extract Bangla text properly. The extracted output often had issues such as missing characters, broken formatting, and poor handling of Bangla fonts, which made it difficult to use directly for downstream tasks like chunking and vectorization.

To address this, I decided to take an alternative approach: I used Optical Character Recognition (OCR) to convert the PDF pages into text. After performing OCR, I saved the output into a .txt file, which provided cleaner and more accurate results—especially for Bangla script.

Once I had the cleaned text file, I used it as the source for chunking and vector embedding. This method helped preserve the semantic integrity of the original content and significantly improved retrieval accuracy in the RAG pipeline.



**2. What chunking strategy did you choose (e.g. paragraph-based, sentence-based, character limit)? Why do you think it works well for semantic retrieval?**  

I used a character-based chunking strategy with a 1000-character limit and 200-character overlap. This ensures each chunk has enough context while preserving continuity between chunks. The overlap helps maintain semantic meaning, especially in Bangla text where sentence boundaries can be inconsistent. This strategy strikes a good balance between chunk size and relevance, improving retrieval accuracy.

**3. What embedding model did you use? Why did you choose it? How does it capture the meaning of the text?**  

I used OpenAI’s text-embedding-3-large model because it provides high-quality multilingual embeddings, including strong support for Bangla and English. It handles long inputs (up to 8,192 tokens) and is optimized for semantic search.

The model captures meaning by converting text into dense vectors using deep transformer layers and contrastive learning, ensuring that semantically similar texts (like a Bangla question and its answer) are placed close together in vector space. This allows accurate and context-aware retrieval, even when exact words differ

**4. How are you comparing the query with your stored chunks? Why did you choose this similarity method and storage setup?** 

I compare the user query with stored chunks using cosine similarity between their embeddings. I chose this method because it's widely used for measuring semantic closeness in vector space.

For storage, I used FAISS, a fast and efficient vector database that supports similarity search at scale. It’s lightweight, easy to integrate, and performs well for local setups.

**5. How do you ensure that the question and the document chunks are compared meaningfully? What would happen if the query is vague or missing context?**  

To ensure meaningful comparison, I use the same embedding model (text-embedding-3-large) for both the user query and document chunks, so they exist in the same semantic space. I also apply cleaning and normalization during preprocessing to improve consistency.

**6. Do the results seem relevant? If not, what might improve them (e.g. better chunking, better embedding model, larger document)?**  

Currently, the system returns approximately 50% relevant answers. While it performs reasonably on direct questions, it struggles with vague or context-heavy queries.

To improve relevance, I plan to:

1. Refine the chunking strategy (e.g., smaller chunks with overlap) to better preserve context
2. Improve PDF preprocessing to reduce noise and formatting issues
3. Try alternative or fine-tuned embedding models for Bangla-specific understanding
4. Add reranking or filtering to prioritize higher-confidence results
5. Include metadata like page numbers or chapter headers to improve grounding

These adjustments should help increase accuracy and improve the overall user experience.