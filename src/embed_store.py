import json
import os
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = "data/processed/chunks.json"
VECTOR_DIR = "data/vector_store"
INDEX_PATH = os.path.join(VECTOR_DIR, "faiss.index")

if __name__ == "__main__":
    os.makedirs(VECTOR_DIR, exist_ok=True)

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device=device
    )

    embeddings = model.encode(
        chunks,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    dim = embeddings.shape[1]
    print(f"Embedding dimension: {dim}")

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    print(f"Stored {index.ntotal} vectors in FAISS")
