import faiss
import json
import torch
from sentence_transformers import SentenceTransformer

INDEX_PATH = "data/vector_store/faiss.index"
CHUNKS_PATH = "data/processed/chunks.json"

if __name__ == "__main__":
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    embed_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device=device
    )

    index = faiss.read_index(INDEX_PATH)

    query = "What is a large language model?"
    query_vec = embed_model.encode(
        [query],
        normalize_embeddings=True
    )

    k = 3
    _, indices = index.search(query_vec, k)

    print("\n Top results:\n")
    for i in indices[0]:
        print(chunks[i])
        print("-" * 80)
