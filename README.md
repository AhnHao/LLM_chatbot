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