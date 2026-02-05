## Data Preparation
- Collected technical PDF documents
- Extracted raw text using PyPDF
- Verified text quality before embedding

## Text Chunking
- Applied recursive character-based chunking
- Chunk size: 500 characters, overlap: 50
- Prepared chunks for semantic embedding

## Embedding & Vector Search
- Generated semantic embeddings using sentence-transformers
- Stored embeddings in FAISS vector database
- Implemented cosine similarity-based semantic search

## Retrieval-Augmented Generation
- Implemented RAG pipeline combining FAISS retrieval and local LLM
- Used prompt constraints to reduce hallucination
- Supported interactive document-based Q&A

## GPU-Accelerated RAG
- Performed query embedding on GPU using sentence-transformers
- Implemented FAISS-based semantic retrieval
- Ran LLM inference on GPU with HuggingFace Transformers
